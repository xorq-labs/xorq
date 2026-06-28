from __future__ import annotations

import datetime
import itertools
import json
import keyword
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache, partial, reduce
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import attr
import click

from xorq.catalog import constants as catalog_constants


if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from xorq.catalog.catalog import Catalog
    from xorq.catalog.content_store import ContentStoreConfig
    from xorq.common.utils.logging_utils import RunLogger
    from xorq.vendor.ibis.expr.types.core import Expr

from xorq.cli_options import (
    cache_dir_option,
    cache_strategy_options,
    code_option,
    content_store_option,
    entry_option,
    env_options,
    fuse_option,
    gcs_option,
    json_option,
    limit_option,
    output_options,
    params_option,
    rebind_backends_option,
    rename_params_option,
    serve_options,
    sync_option,
    unbind_options,
)
from xorq.ibis_yaml.enums import DumpFiles, ExprKind


def click_handler(e: Exception) -> None:
    raise click.ClickException(str(e)) from e


def _init_hint(ctx):
    """Return the `xorq catalog ... init` command the user should run."""
    group_ctx = ctx.parent or ctx
    match (group_ctx.params.get("name"), group_ctx.params.get("path")):
        case (str() as name, _):
            return f"xorq catalog --name {name} init"
        case (_, str() as path):
            return f"xorq catalog --path {path} init"
        case _:
            return "xorq catalog init"


def _pdb_active(ctx):
    """Return True when --pdb was passed to the top-level CLI group."""
    return ctx.find_root().params.get("use_pdb", False)


@contextmanager
def click_context(ctx: click.Context, *typs: type[Exception]) -> Iterator[None]:
    try:
        yield
    except click.ClickException:
        raise
    except typs as e:
        if _pdb_active(ctx):
            raise
        click_handler(e)


@contextmanager
def click_context_catalog(ctx: click.Context) -> Iterator[None]:
    from git import NoSuchPathError  # noqa: PLC0415

    try:
        yield
    except click.ClickException:
        raise
    except NoSuchPathError as e:
        hint = _init_hint(ctx)
        raise click.ClickException(
            f"Catalog not found: {e}\nRun `{hint}` to create it."
        ) from e
    except Exception as e:
        if _pdb_active(ctx):
            raise
        click_handler(e)


def click_context_default(ctx):
    return click_context(ctx, AssertionError, Exception)


@cache
def _make_catalog_for_completion(ctx):
    from xorq.catalog.catalog import Catalog  # noqa: PLC0415

    catalog_ctx = ctx.parent
    return Catalog.from_kwargs(
        name=catalog_ctx.params.get("name"),
        path=catalog_ctx.params.get("path"),
        url=catalog_ctx.params.get("url"),
        root_repo=catalog_ctx.params.get("root_repo"),
        init=False,
    )


def _complete_entry_names(ctx, param, incomplete):
    from click.shell_completion import CompletionItem  # noqa: PLC0415

    try:
        catalog = _make_catalog_for_completion(ctx)
        return [CompletionItem(n) for n in catalog.list() if n.startswith(incomplete)]
    except Exception:
        return []


def _complete_alias_names(ctx, param, incomplete):
    from click.shell_completion import CompletionItem  # noqa: PLC0415

    try:
        catalog = _make_catalog_for_completion(ctx)
        return [
            CompletionItem(a)
            for a in catalog.list_aliases()
            if a.startswith(incomplete)
        ]
    except Exception:
        return []


def _complete_entry_or_alias_names(ctx, param, incomplete):
    try:
        return sorted(
            _complete_entry_names(ctx, param, incomplete)
            + _complete_alias_names(ctx, param, incomplete)
        )
    except Exception:
        return []


@click.group(invoke_without_command=True)
@click.option(
    "-n", "--name", default=None, help="Catalog name (mutually exclusive with --path)."
)
@click.option(
    "-p",
    "--path",
    default=None,
    type=click.Path(file_okay=False),
    help="Catalog repo path (mutually exclusive with --name).",
)
@click.option(
    "-u",
    "--url",
    default=None,
    help="Remote repo URL to clone.",
)
@click.option(
    "-r",
    "--root-repo",
    default=None,
    type=click.Path(file_okay=False, exists=True),
    help="Repo root to add this catalog to as a submodule.",
)
@click.option(
    "--init/--no-init",
    default=None,
    show_default="auto",
    help="Initialize the repo.",
)
@click.pass_context
def cli(ctx, name, path, url, root_repo, init):
    """Manage Xorq build-artifact catalogs.

    A catalog is a versioned store of named, sharable, executable
    expressions: a git repository with an optional object-store backed
    annex for content. Group options select which catalog a subcommand
    targets and precede the subcommand name (for example,
    `xorq catalog -n analytics info`).
    """
    from xorq.catalog.catalog import Catalog  # noqa: PLC0415

    ctx.obj = SimpleNamespace(
        make_catalog=partial(
            Catalog.from_kwargs,
            name=name,
            path=path,
            url=url,
            root_repo=root_repo,
            init=init,
        )
    )
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option(
    "--refresh",
    default=10,
    type=float,
    show_default=True,
    help="Refresh interval in seconds.",
)
@click.pass_context
def tui(ctx, refresh):
    """Browse the catalog interactively in a terminal UI.

    Shows entries, aliases, schemas, and metadata, refreshing on an
    interval. The TUI exits on `q` or `Ctrl+C`.

    \b
    Examples:
      # Browse the default catalog
      xorq catalog tui
      # Browse a named catalog with a faster refresh
      xorq catalog --name analytics tui --refresh 2
    """
    with click_context_catalog(ctx):
        ctx.obj.make_catalog(init=False)  # validate catalog exists
    from xorq.catalog.tui import CatalogTUI  # noqa: PLC0415

    app = CatalogTUI(
        partial(ctx.obj.make_catalog, init=False), refresh_interval=refresh
    )
    app.run()


def _resolve_annex_option(env_file, env_prefix, gcs):
    """Return a RemoteConfig from CLI options, or None for plain git."""
    if env_file and env_prefix:
        raise click.UsageError("--env-file and --env-prefix are mutually exclusive.")
    if env_file:
        from xorq.catalog.annex import remote_config_from_env_file  # noqa: PLC0415

        return remote_config_from_env_file(env_file, gcs=gcs)
    elif env_prefix:
        from xorq.catalog.annex import remote_config_from_prefix  # noqa: PLC0415

        return remote_config_from_prefix(env_prefix, gcs=gcs)
    return None


def _resolve_content_store_option(
    content_store_type: str | None, gcs: bool
) -> ContentStoreConfig | None:
    """Return a ContentStoreConfig from CLI options, or None."""
    from xorq.catalog.content_store import (  # noqa: PLC0415
        DirectoryContentStoreConfig,
        S3ContentStoreConfig,
    )

    match content_store_type:
        case None:
            return None
        case "s3":
            if gcs:
                return S3ContentStoreConfig.from_env_gcs()
            return S3ContentStoreConfig.from_env()
        case "directory":
            return DirectoryContentStoreConfig.from_env()
        case _:
            raise click.UsageError(
                f"unknown content store type: {content_store_type!r}"
            )


def _check_backend_exclusive_cli(
    content_store_config: object | None, annex: object | None
) -> None:
    if content_store_config is not None and annex is not None:
        raise click.UsageError(
            "--content-store and annex options (--env-file/--env-prefix) "
            "are mutually exclusive."
        )


@cli.command()
@env_options
@gcs_option
@content_store_option
@click.option(
    "--remote-url",
    default=None,
    help="Git remote URL (sets origin).",
)
@click.pass_context
def init(
    ctx: click.Context,
    env_file: str | None,
    env_prefix: str | None,
    gcs: bool,
    content_store_type: str | None,
    remote_url: str | None,
) -> None:
    """Create a new catalog repository.

    The destination comes from the catalog group's selectors (`-n/--name`
    or `-p/--path`), supplied before the `init` subcommand. The command
    refuses to overwrite an existing catalog at the target path.

    \b
    Examples:
      # Create a catalog at the default location for the name
      xorq catalog --name analytics init
      # Create at an explicit path with a git remote
      xorq catalog --path ./catalogs/analytics init --remote-url git@github.com:acme/analytics-catalog.git
      # Create with an S3-backed annex remote sourced from an env file
      xorq catalog --name analytics init --env-file .env.catalog.s3
    """
    with click_context_catalog(ctx):
        annex = _resolve_annex_option(env_file, env_prefix, gcs)
        content_store_config = _resolve_content_store_option(content_store_type, gcs)
        _check_backend_exclusive_cli(content_store_config, annex)
        try:
            catalog = ctx.obj.make_catalog(
                init=True, annex=annex, content_store_config=content_store_config
            )
        except FileExistsError as err:
            probe = ctx.obj.make_catalog(init=False)
            raise click.ClickException(
                f"Catalog already exists at {probe.repo_path}"
            ) from err
        click.echo(f"Initialized catalog at {catalog.repo_path}")
        if remote_url:
            from xorq.catalog.constants import DEFAULT_REMOTE  # noqa: PLC0415

            remote = catalog.set_remote(DEFAULT_REMOTE, remote_url)
            click.echo(f"Set remote {remote.name} -> {remote_url}")


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@sync_option
@click.option(
    "-a",
    "--alias",
    "aliases",
    multiple=True,
    help="Alias to assign to each added entry (repeatable).",
)
@click.pass_context
def add(ctx, paths, sync, aliases):
    """Add entries from build directories or archive files.

    \b
    Arguments:
      PATHS  One or more paths; each is a build directory (typically
             produced by `xorq build`) or an archive file (.zip).

    \b
    Examples:
      # Add a single build directory
      xorq catalog add builds/f02d28198715
      # Add multiple builds in one invocation
      xorq catalog add builds/f02d28198715 builds/c5a981ab43cd
      # Assign aliases at add time (applied to each added entry)
      xorq catalog add builds/f02d28198715 -a penguins-prod -a current
      # Defer the push to a later `xorq catalog push`
      xorq catalog add builds/f02d28198715 --no-sync
    """
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        with catalog.maybe_synchronizing(sync):
            for path in map(Path, paths):
                entry = catalog.add(path, sync=False, aliases=aliases)
                click.echo(f"Added {entry.name}")


@cli.command()
@click.argument("names", nargs=-1, required=True, shell_complete=_complete_entry_names)
@sync_option
@click.pass_context
def remove(ctx, names, sync):
    """Remove entries by name.

    \b
    Arguments:
      NAMES  One or more entry names; use `xorq catalog list` to see
             available names.

    \b
    Examples:
      # Remove a single entry
      xorq catalog remove f02d28198715
      # Remove multiple entries in a single commit
      xorq catalog remove f02d28198715 c5a981ab43cd
      # Remove without syncing to the remote
      xorq catalog remove f02d28198715 --no-sync
    """
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        with catalog.maybe_synchronizing(sync):
            for name in names:
                entry = catalog.remove(name, sync=False)
                click.echo(f"Removed {entry.name}")


@cli.command("add-alias")
@click.argument("name", shell_complete=_complete_entry_names)
@click.argument("alias")
@sync_option
@click.pass_context
def add_alias(ctx, name, alias, sync):
    """Attach an alias to an existing entry.

    Aliases are stable, human-readable names accepted anywhere an entry
    name is. To assign an alias at the same time you add an entry, use
    `xorq catalog add -a`.

    \b
    Arguments:
      NAME   The existing entry's name.
      ALIAS  The alias to register.

    \b
    Examples:
      xorq catalog add-alias f02d28198715 penguins-prod
    """
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        catalog_alias = catalog.add_alias(name, alias, sync=sync)
        click.echo(f"Added alias {catalog_alias.alias!r} -> {name}")


@cli.command("remove-alias")
@click.argument(
    "aliases", nargs=-1, required=True, shell_complete=_complete_alias_names
)
@sync_option
@click.pass_context
def remove_alias(ctx, aliases, sync):
    """Remove one or more aliases.

    The underlying entries are left untouched—use `xorq catalog remove`
    to remove an entry itself. Unknown aliases produce an error and cancel
    the operation before any change is committed.

    \b
    Arguments:
      ALIASES  One or more alias names; use `xorq catalog list-aliases`
               to see registered aliases.

    \b
    Examples:
      xorq catalog remove-alias penguins-prod
      xorq catalog remove-alias penguins-prod current
    """
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        alias_map = {ca.alias: ca for ca in catalog.catalog_aliases}
        with catalog.maybe_synchronizing(sync):
            for alias in aliases:
                if alias not in alias_map:
                    raise click.UsageError(f"Unknown alias: {alias!r}")
                alias_map[alias].remove()
                click.echo(f"Removed alias {alias!r}")


@cli.command("list")
@click.option(
    "--kind/--no-kind",
    default=False,
    show_default=True,
    help="Print a second column showing each entry's kind.",
)
@click.pass_context
def list_entries(ctx: click.Context, kind: bool) -> None:
    """List all entries.

    \b
    Examples:
      # Names only (one per line)
      xorq catalog list
      # Names with a kind column (tab-separated)
      xorq catalog list --kind
    """
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)

        if not (entries := catalog.catalog_entries):
            click.echo("No entries.")
            return

        for entry in entries:
            click.echo(f"{entry.name}\t{entry.kind}" if kind else entry.name)


@cli.command("list-aliases")
@click.pass_context
def list_aliases(ctx):
    """List all aliases.

    Prints one alias per line, or `No aliases.` when the catalog has none.
    """
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        aliases = catalog.list_aliases() or ("No aliases.",)
        for alias in aliases:
            click.echo(alias)


@cli.command()
@click.pass_context
def info(ctx):
    """Show catalog summary metadata.

    Prints the filesystem path, current commit, configured remotes, and
    entry and alias counts. For per-entry metadata, use
    `xorq catalog show` or `xorq catalog schema`.

    \b
    Examples:
      # Inspect the default catalog
      xorq catalog info
      # Inspect a specific catalog via the group selectors
      xorq catalog --name my-catalog info
    """
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        click.echo(f"path:    {catalog.repo_path}")
        click.echo(f"commit:  {catalog.repo.head.commit.hexsha[:12]}")
        remotes = tuple(r.name for r in catalog.repo.remotes)
        click.echo(f"remotes: {', '.join(remotes) if remotes else '(none)'}")
        click.echo(f"entries: {len(catalog.list())}")
        click.echo(f"aliases: {len(catalog.list_aliases())}")


@cli.command("default")
@click.option("--set", "set_name", default=None, help="Set the default catalog name.")
@click.option("--unset", is_flag=True, help="Remove the persisted default.")
def default_catalog(set_name, unset):
    """Show or change the persisted default catalog name.

    The default catalog applies when `xorq catalog` invocations supply
    neither `-n/--name` nor `-p/--path`. Without options, prints the
    resolved name and the source that supplied it. `--set` and `--unset`
    are mutually exclusive.

    \b
    Resolution order:
    1. The `XORQ_DEFAULT_CATALOG` environment variable.
    2. The persisted config file (written by `--set`).
    3. The built-in default name (`default`).

    \b
    Examples:
      # Show the current default and where it came from
      xorq catalog default
      # Persist a new default
      xorq catalog default --set analytics
      # Remove the persisted default
      xorq catalog default --unset
    """
    from xorq.catalog.catalog import Catalog  # noqa: PLC0415
    from xorq.vendor.ibis.config import env_config  # noqa: PLC0415

    if set_name and unset:
        raise click.UsageError("--set and --unset are mutually exclusive.")

    if set_name:
        catalog_constants.DEFAULT_CATALOG_CONFIG.parent.mkdir(
            parents=True, exist_ok=True
        )
        catalog_constants.DEFAULT_CATALOG_CONFIG.write_text(set_name + "\n")
        click.echo(f"Default catalog set to {set_name!r}")
    elif unset:
        try:
            catalog_constants.DEFAULT_CATALOG_CONFIG.unlink()
            click.echo("Default catalog unset (reverted to 'default')")
        except FileNotFoundError:
            click.echo("No persisted default to unset.")
    else:
        name = Catalog._resolve_default_name()
        if env_config.XORQ_DEFAULT_CATALOG:
            source = "env (XORQ_DEFAULT_CATALOG)"
        elif catalog_constants.DEFAULT_CATALOG_CONFIG.exists():
            source = f"config ({catalog_constants.DEFAULT_CATALOG_CONFIG})"
        else:
            source = "built-in"
        click.echo(f"{name}  (source: {source})")


@cli.command()
@click.argument("name", shell_complete=_complete_entry_names)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(),
    show_default="current directory",
    help="Destination directory for the archive.",
)
@click.pass_context
def get(ctx, name, output):
    """Export an entry's archive to a directory.

    Writes the entry's zipped build directory to
    `<output_dir>/<name>.zip`, overwriting any existing file at that
    path—useful for shipping an entry to a system without catalog access.
    Re-add an exported archive with `xorq catalog add`.

    \b
    Arguments:
      NAME  Entry name to export.

    \b
    Examples:
      xorq catalog get f02d28198715 -o ./shipped-builds
    """
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        result = catalog.get_zip(name, dir_path=output)
        click.echo(f"Exported to {result}")


@cli.command()
@click.pass_context
def push(ctx):
    """Push the catalog's commits and annex content to its remotes.

    Pair with `--no-sync` on mutating commands (`add`, `remove`,
    `add-alias`, `remove-alias`) when batching changes for a single
    explicit push.

    \b
    Examples:
      xorq catalog add builds/f02d28198715 --no-sync
      xorq catalog add-alias f02d28198715 prod --no-sync
      xorq catalog push
    """
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        catalog.push()
        click.echo("Pushed.")


@cli.command()
@click.pass_context
def pull(ctx):
    """Pull the catalog's commits and annex content from its remotes."""
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        catalog.pull()
        click.echo("Pulled.")


@cli.command()
@click.pass_context
def sync(ctx):
    """Pull then push.

    Equivalent to running `xorq catalog pull` followed by
    `xorq catalog push`.
    """
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        catalog.sync()
        click.echo("Synced.")


@cli.command("set-remote")
@click.argument("url")
@click.option(
    "--name",
    default=None,
    show_default="origin",
    help="Remote name.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Replace the existing git remote (otherwise this command refuses to overwrite).",
)
@click.pass_context
def set_remote(ctx, url, name, force):
    """Configure the catalog's git remote.

    The catalog supports at most one git remote (ADR-0011). If no git
    remote is configured, this command sets one. If a git remote is
    already configured, this command refuses unless `--force` is passed—
    guarding against typos that would silently delete the configured
    remote.

    \b
    Arguments:
      URL  Git remote URL to configure.

    \b
    Examples:
      # First-time setup
      xorq catalog --path ~/flights-catalog set-remote git@github.com:me/flights-catalog.git
      # Replace an existing remote
      xorq catalog set-remote git@github.com:me/flights-catalog.git --force
    """
    from xorq.catalog.constants import DEFAULT_REMOTE  # noqa: PLC0415
    from xorq.catalog.exceptions import CatalogConfigurationError  # noqa: PLC0415

    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        try:
            remote = catalog.set_remote(name or DEFAULT_REMOTE, url, force=force)
        except CatalogConfigurationError as err:
            raise click.ClickException(str(err)) from err
        click.echo(f"Set remote {remote.name} -> {url}")


@cli.command()
@click.argument("url")
@click.option(
    "-n", "--name", "dest_name", default=None, help="Destination catalog name."
)
@click.option(
    "-p",
    "--path",
    "dest_path",
    default=None,
    type=click.Path(),
    help="Destination repo path.",
)
@click.pass_context
def clone(ctx, url, dest_name, dest_path):
    """Clone an existing catalog from a remote URL.

    `--name` and `--path` are mutually exclusive; with neither, a default
    location derives from the source URL.

    \b
    Arguments:
      URL  The git URL of the source catalog repository.

    \b
    Examples:
      # Clone to the default location
      xorq catalog clone git@github.com:acme/analytics-catalog.git
      # Clone under a specific name
      xorq catalog clone git@github.com:acme/analytics-catalog.git --name analytics
      # Clone to an explicit path
      xorq catalog clone git@github.com:acme/analytics-catalog.git --path ./catalogs/analytics
    """
    from xorq.catalog.catalog import Catalog  # noqa: PLC0415

    with click_context_default(ctx):
        match (dest_path, dest_name):
            case (None, None):
                repo_path = None
            case (_, None):
                repo_path = dest_path
            case (None, _):
                repo_path = Catalog.name_to_repo_path(dest_name)
            case (_, _):
                raise click.UsageError("--name and --path are mutually exclusive.")
        catalog = Catalog.clone_from(url, repo_path)
        click.echo(f"Cloned to {catalog.repo_path}")


@cli.command()
@click.argument("name", shell_complete=_complete_entry_or_alias_names)
@json_option
@click.pass_context
def schema(ctx, name, as_json):
    """Show the schemas for a catalog entry.

    Bound entries print a single output schema; unbound (partial) entries
    print both an input and an output schema.

    \b
    Arguments:
      NAME  Entry name or alias.

    \b
    Examples:
      # Formatted schema view
      xorq catalog schema penguins-prod
      # Machine-readable metadata
      xorq catalog schema penguins-prod --json
    """
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        entry = _get_catalog_entry(catalog, name)

        if as_json:
            click.echo(json.dumps(entry.metadata.to_dict(), indent=2))
            return

        type_label = (
            "Partial (unbound)"
            if entry.kind == ExprKind.UnboundExpr
            else "Source (bound)"
        )
        click.echo(f"Type: {type_label}")

        meta = entry.metadata
        for label, schema in (
            ("Schema In", meta.schema_in),
            ("Schema Out", meta.schema_out),
        ):
            if schema is not None:
                click.echo()
                click.echo(f"{label}:")
                for col, dtype in schema.items():
                    click.echo(f"  {col:<24} {dtype}")


@cli.command()
@click.argument("name", shell_complete=_complete_entry_or_alias_names)
@json_option
@click.option(
    "--raw",
    "as_raw",
    is_flag=True,
    default=False,
    help="Print the metadata sidecar file as-is (YAML).",
)
@click.pass_context
def show(ctx, name, as_json, as_raw):
    """Show full metadata for a catalog entry.

    Prints name, aliases, kind, backends, schemas, parameters, builders,
    cache key, and more. `--json` and `--raw` are mutually exclusive.

    \b
    Arguments:
      NAME  Entry name or alias.

    \b
    Examples:
      # Human-readable metadata
      xorq catalog show penguins-prod
      # Structured sidecar metadata as JSON
      xorq catalog show penguins-prod --json
    """
    if as_json and as_raw:
        raise click.UsageError("--json and --raw are mutually exclusive.")

    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        entry = _get_catalog_entry(catalog, name)

        if as_raw:
            click.echo(entry.metadata_path.read_text(), nl=False)
            return

        if as_json:
            click.echo(json.dumps(entry.sidecar_metadata, indent=2, default=str))
            return

        meta = entry.metadata
        type_label = {
            ExprKind.UnboundExpr: "Partial (unbound)",
            ExprKind.Source: "Source (bound)",
            ExprKind.Expr: "Expression",
            ExprKind.Composed: "Composed",
            ExprKind.ExprBuilder: "Expression Builder",
        }.get(entry.kind, str(entry.kind))
        click.echo(f"{'Name:':<15} {entry.name}")
        aliases = tuple(a.alias for a in entry.aliases)
        if aliases:
            click.echo(f"{'Aliases:':<15} {', '.join(aliases)}")
        click.echo(f"{'Type:':<15} {type_label}")
        if meta.root_tag:
            click.echo(f"{'Root tag:':<15} {meta.root_tag}")
        backends = entry.backends
        if backends:
            click.echo(f"{'Backends:':<15} {', '.join(backends)}")
        click.echo(
            f"{'Content local:':<15} {'yes' if entry.is_content_local else 'no'}"
        )
        if meta.composed_from:
            click.echo(f"{'Composed from:':<15} {len(meta.composed_from)}")
        if meta.projected_cache_key and meta.projected_cache_key.key:
            click.echo(f"{'Cache key:':<15} {meta.projected_cache_key.key}")
        if meta.params:
            click.echo()
            click.echo("Params:")
            for p in meta.params:
                tail = f" = {p['default']!r}" if "default" in p else ""
                click.echo(f"  {p['param_name']:<24} {p['type']}{tail}")
        if meta.builders:
            click.echo()
            click.echo("Builders:")
            for builder in meta.builders:
                click.echo(f"  Type: {builder.get('type', 'unknown')}")
                for key, value in builder.items():
                    if key == "type":
                        continue
                    if isinstance(value, (list, tuple)):
                        if value:
                            click.echo(f"    {key}: {', '.join(str(v) for v in value)}")
                    elif value is not None:
                        click.echo(f"    {key}: {value}")
        for label, schema in (
            ("Schema In", meta.schema_in),
            ("Schema Out", meta.schema_out),
        ):
            if schema is not None:
                click.echo()
                click.echo(f"{label}:")
                for col, dtype in schema.items():
                    click.echo(f"  {col:<24} {dtype}")


@cli.command()
@click.pass_context
def check(ctx):
    """Validate the catalog for consistency.

    Verifies that entries declared in the catalog manifest are present on
    disk, that aliases point to existing entries, and that recorded
    metadata matches actual content. Prints `OK` on success; otherwise
    exits non-zero with a description of the inconsistency.
    """
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        catalog.assert_consistency()
        click.echo("OK")


@cli.command()
@click.option(
    "--dry-run/--no-dry-run", default=True, help="List orphans without deleting."
)
@click.pass_context
def gc(ctx: click.Context, dry_run: bool) -> None:
    """Remove orphaned content store objects (pointer backend only)."""
    from xorq.catalog.backend import GitPointerBackend  # noqa: PLC0415

    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        if not isinstance(catalog.backend, GitPointerBackend):
            raise click.ClickException(
                "gc is only supported for pointer-backend catalogs"
            )
        orphans = catalog.backend.gc_content_store(dry_run=dry_run)
        if not orphans:
            click.echo("No orphaned content store objects found.")
        else:
            action = "Would delete" if dry_run else "Deleted"
            for key in orphans:
                click.echo(f"  {action}: {key}")
            click.echo(
                f"\n{len(orphans)} orphan(s) {'found' if dry_run else 'deleted'}."
            )


@cli.command()
@json_option
@click.pass_context
def log(ctx: click.Context, as_json: bool) -> None:
    """Show the catalog's history as structured operations.

    These are the same operations `xorq catalog replay` consumes when
    copying a catalog.

    \b
    Examples:
      xorq catalog log
      xorq catalog log --json | jq '.[] | select(.type == "Add")'
    """

    from xorq.catalog.replay import Replayer  # noqa: PLC0415

    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        replayer = Replayer(from_catalog=catalog)
        if as_json:
            click.echo(
                json.dumps(
                    [
                        {"type": type(op).__name__, **attr.asdict(op)}
                        for op in replayer.ops
                    ],
                    indent=2,
                )
            )
        else:
            replayer.print_plan()


@cli.command()
@click.argument("target_path", type=click.Path(file_okay=False))
@env_options
@gcs_option
@content_store_option
@click.option(
    "--remote-url",
    default=None,
    help="Git remote URL for the target catalog (sets origin and pushes).",
)
@click.option(
    "--preserve-commits/--no-preserve-commits",
    default=True,
    show_default=True,
    help="Preserve original commit authors and timestamps.",
)
@click.option("--force", is_flag=True, help="Force-push to the remote.")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be replayed without executing."
)
@click.option(
    "--rebuild",
    is_flag=True,
    help="Rebuild each entry under current code (refreshes build_metadata and entry hashes).",
)
@click.pass_context
def replay(
    ctx,
    target_path,
    env_file,
    env_prefix,
    gcs,
    content_store_type,
    remote_url,
    preserve_commits,
    force,
    dry_run,
    rebuild,
):
    """Replay the catalog's operation log into a target catalog.

    Useful for mirroring a catalog with a different remote backend, or
    for migrating between storage providers.

    With `--rebuild`, each entry is re-added under current code: entries
    with no catalog references are re-added from their stored expression;
    entries containing catalog references (Composed, or ExprBuilder
    wrapping a composition) have the catalog subtree recomposed against
    their already-rebuilt dependencies in the target catalog. Outer
    builder wrappings pass through untouched.

    \b
    Arguments:
      TARGET_PATH  Filesystem path where the target catalog is initialized.

    \b
    Examples:
      # Preview the operations that would be replayed
      xorq catalog replay ./mirrored-catalog --dry-run
      # Replay into a new catalog and push to a fresh remote
      xorq catalog replay ./mirrored-catalog --env-file .env.target.s3 --remote-url git@github.com:acme/mirror-catalog.git
    """
    from xorq.catalog.catalog import Catalog  # noqa: PLC0415
    from xorq.catalog.replay import Replayer  # noqa: PLC0415

    with click_context_catalog(ctx):
        source = ctx.obj.make_catalog(init=False)
        replayer = Replayer(from_catalog=source, rebuild=rebuild)
        if dry_run:
            replayer.print_plan()
            return
        annex = _resolve_annex_option(env_file, env_prefix, gcs)
        content_store_config = _resolve_content_store_option(content_store_type, gcs)
        _check_backend_exclusive_cli(content_store_config, annex)
        target = Catalog.from_repo_path(
            target_path, annex=annex, content_store_config=content_store_config
        )
        replayer.replay(target, preserve_commits=preserve_commits)
        click.echo(f"Replayed {len(replayer.ops)} operations into {target_path}")
        if remote_url:
            from xorq.catalog.constants import (  # noqa: PLC0415
                ANNEX_BRANCH,
                DEFAULT_REMOTE,
                MAIN_BRANCH,
            )

            origin = target.repo.create_remote(DEFAULT_REMOTE, remote_url)
            refspec_prefix = "+" if force else ""
            origin.push(f"{refspec_prefix}{MAIN_BRANCH}:{MAIN_BRANCH}")
            origin.push(f"{refspec_prefix}{ANNEX_BRANCH}:{ANNEX_BRANCH}")
            origin.fetch()
            target.repo.heads[MAIN_BRANCH].set_tracking_branch(origin.refs[MAIN_BRANCH])
            if ANNEX_BRANCH in target.repo.heads:
                target.repo.heads[ANNEX_BRANCH].set_tracking_branch(
                    origin.refs[ANNEX_BRANCH]
                )
            click.echo(f"Pushed to {remote_url}")


@cli.command("embed-readonly")
@env_options
@gcs_option
@click.pass_context
def embed_readonly(ctx, env_file, env_prefix, gcs):
    """Embed read-only S3 credentials into the catalog's git-annex branch.

    Lets consumers cloning the catalog fetch annexed content without
    supplying credentials of their own. One of `--env-file` or
    `--env-prefix` is required.

    Before embedding, the command probes the bucket by initiating (and
    canceling) a multipart upload with the supplied credentials. If that
    write succeeds, the credentials aren't read-only and the command
    raises an error instead of embedding them. Embedding sets
    `embedcreds=yes` on the S3 remote config and writes the result to
    `remote.log` on the git-annex branch.

    \b
    Examples:
      xorq catalog embed-readonly --env-file .env.catalog.readonly
      xorq catalog embed-readonly --env-prefix XORQ_CATALOG_RO_
    """
    with click_context_catalog(ctx):
        ro_config = _resolve_annex_option(env_file, env_prefix, gcs)
        if ro_config is None:
            raise click.UsageError(
                "Provide --env-file or --env-prefix for the read-only credentials."
            )
        catalog = ctx.obj.make_catalog(init=False)
        catalog.embed_readonly(ro_config)
        click.echo(f"Embedded read-only credentials into {catalog.repo_path}")


def _get_catalog_entry(catalog, name):
    """Look up an entry by name or alias, raising a helpful ClickException."""
    try:
        return catalog.get_catalog_entry(name, maybe_alias=True)
    except (ValueError, AssertionError) as err:
        raise click.ClickException(
            f"Entry {name!r} not found — run 'xorq catalog list' "
            f"or 'xorq catalog list-aliases' to see available entries and aliases."
        ) from err


def _eval_entry(catalog_entry, code, instream=None, cache_dir=None):
    """Evaluate a single catalog entry to an expression."""
    from xorq.catalog.bind import _eval_code, _make_source_expr  # noqa: PLC0415

    match catalog_entry.kind:
        case ExprKind.UnboundExpr:
            import pyarrow as pa  # noqa: PLC0415

            from xorq.common.utils.graph_utils import replace_unbound  # noqa: PLC0415
            from xorq.common.utils.io_utils import maybe_open  # noqa: PLC0415
            from xorq.expr.api import read_pyarrow_stream  # noqa: PLC0415

            loaded_expr = catalog_entry.load_expr(cache_dir=cache_dir)
            try:
                with maybe_open(instream, "rb") as stream:
                    input_expr = read_pyarrow_stream(stream)
            except (pa.ArrowInvalid, pa.ArrowException) as err:
                raise click.ClickException(
                    f"Entry {catalog_entry.name!r} is an unbound expression — "
                    f"pipe Arrow data into it or pass a source entry: "
                    f"'xorq catalog run SOURCE {catalog_entry.name} -o - -f csv'."
                ) from err
            expr = replace_unbound(loaded_expr, input_expr.op())
        case ExprKind.Source | ExprKind.Expr | ExprKind.Composed | ExprKind.ExprBuilder:
            expr = _make_source_expr(catalog_entry, cache_dir=cache_dir)
        case _:
            raise click.ClickException(
                f"Unsupported entry kind {catalog_entry.kind!r} for 'run'."
            )

    if code is not None:
        expr = _eval_code(code, expr)

    return expr


def _parse_rename_params(raw_rename_params):
    """Parse repeatable ``--rename-params entry,old,new`` strings.

    Returns ``{entry_name: {old_label: new_label}}``.
    """
    result = {}
    errors = []
    for spec in raw_rename_params:
        parts = tuple(p.strip() for p in spec.split(","))
        if len(parts) != 3:
            errors.append(f"Expected 'entry,old_name,new_name', got {spec!r}")
            continue
        entry_name, old, new = parts
        result.setdefault(entry_name, {})[old] = new
    if errors:
        raise click.BadParameter("\n".join(errors))
    return result


_RESERVED_BINDING_NAMES = frozenset({"bind", "xo", "ibis", "into_backend"})


def _parse_entry_bindings(raw_entries: tuple[str, ...]) -> dict[str, str]:
    """Parse repeatable ``-e NAME=ENTRY`` values into an ordered dict.

    Returns ``{}`` when no ``-e`` was given (single-root / linear mode).
    Insertion order is preserved so the first binding is the default
    rebind target.
    """
    bindings: dict[str, str] = {}
    for raw in raw_entries:
        name, sep, entry = raw.partition("=")
        if not sep or not name or not entry:
            raise click.BadParameter(
                f"--entry must be NAME=ENTRY, got {raw!r}",
                param_hint="-e/--entry",
            )
        if not name.isidentifier():
            raise click.BadParameter(
                f"binding name {name!r} is not a valid Python identifier",
                param_hint="-e/--entry",
            )
        if keyword.iskeyword(name):
            raise click.BadParameter(
                f"binding name {name!r} is a Python keyword",
                param_hint="-e/--entry",
            )
        if name.startswith("__") and name.endswith("__"):
            raise click.BadParameter(
                f"binding name {name!r} may not be a dunder",
                param_hint="-e/--entry",
            )
        if name in _RESERVED_BINDING_NAMES:
            raise click.BadParameter(
                f"binding name {name!r} is reserved "
                f"(reserved: {', '.join(sorted(_RESERVED_BINDING_NAMES))})",
                param_hint="-e/--entry",
            )
        if name in bindings:
            raise click.BadParameter(
                f"duplicate binding name {name!r}",
                param_hint="-e/--entry",
            )
        bindings[name] = entry
    return bindings


def _resolve_mode(
    entries: tuple[str, ...], raw_entries: tuple[str, ...], code: str | None
) -> dict[str, str]:
    """Validate and return the binding dict (empty in linear mode).

    Positional ENTRIES may be mixed with ``-e`` in two ways:
    ``-e source=ENTRY transform...`` applies transforms to that binding, while
    ``ENTRY transform... -e name=other`` exposes the positional chain as
    ``source`` alongside the extra bindings.
    """
    bindings = _parse_entry_bindings(raw_entries)
    if bindings and entries:
        if code is None and set(bindings) != {"source"}:
            raise click.UsageError(
                "Mixed mode with extra -e/--entry bindings "
                "requires inline code via -c/--code."
            )
    if bindings and not entries and code is None:
        raise click.UsageError(
            "Multi-root mode (-e/--entry) requires inline code via -c/--code."
        )
    return bindings


def _is_cross_backend(expr: Expr) -> bool:
    """True when *expr* reads from more than one distinct backend object."""
    from xorq.common.utils.graph_utils import find_all_sources  # noqa: PLC0415

    return len({id(s) for s in find_all_sources(expr)}) > 1


def _compose_linear_expr(
    catalog: Catalog,
    entries: tuple[str, ...],
    code: str | None,
    rename_map: dict[str, dict[str, str]] | None = None,
    cache_dir: str | Path | None = None,
    source_alias: str | None = None,
) -> Expr:
    """Build a linear source + transforms expression."""

    from xorq.catalog.bind import (  # noqa: PLC0415
        _eval_code,
        _resolve_source,
    )
    from xorq.common.utils.graph_utils import rename_params  # noqa: PLC0415

    if not entries:
        raise click.UsageError("At least one entry is required.")

    rename_map = rename_map or {}

    catalog_entries = {
        name: catalog.get_catalog_entry(name, maybe_alias=True) for name in entries
    }

    unknown = set(rename_map) - set(catalog_entries)
    if unknown:
        raise click.BadParameter(
            f"Unknown entry name(s) in --rename-params: {', '.join(sorted(unknown))}"
        )

    source_entry, *transform_entries = catalog_entries.values()

    source_name, *transform_names = entries

    # Fast path: delegate to ExprComposer when no per-entry transformations are needed
    if not rename_map and cache_dir is None:
        from xorq.catalog.composer import ExprComposer  # noqa: PLC0415

        return ExprComposer(
            source=source_entry,
            transforms=transform_entries,
            code=code,
            alias=source_alias,
        ).expr

    def _load(entry):
        return entry.load_expr(cache_dir=cache_dir)

    if source_name in rename_map:
        source_expr = rename_params(_load(source_entry), rename_map[source_name])
        source_tagged, resolved_con = _resolve_source(source_expr, None, source_alias)
    else:
        source_tagged, resolved_con = _resolve_source(
            source_entry, None, source_alias, cache_dir=cache_dir
        )

    if transform_entries:
        from xorq.catalog.bind import _ensure_remote  # noqa: PLC0415
        from xorq.common.utils.graph_utils import replace_unbound  # noqa: PLC0415
        from xorq.expr.relations import RemoteTable, gen_name  # noqa: PLC0415

        def _bind_one_with_rename(current_expr, entry_and_name):
            entry, name = entry_and_name
            if name in rename_map:
                transform_expr = rename_params(_load(entry), rename_map[name])
            else:
                transform_expr = _load(entry)
            source_node = _ensure_remote(current_expr.op(), resolved_con, current_expr)
            composed_expr = replace_unbound(transform_expr, source_node)
            return RemoteTable(
                name=gen_name(),
                schema=composed_expr.as_table().schema(),
                source=resolved_con,
                remote_expr=composed_expr,
            ).to_expr()

        current = reduce(
            _bind_one_with_rename,
            zip(transform_entries, transform_names),
            source_tagged,
        )
    else:
        current = source_tagged

    if code is not None:
        current = _eval_code(code, current)

    return current


def _compose_expr(
    catalog: Catalog,
    entries: tuple[str, ...],
    code: str | None,
    rename_map: dict[str, dict[str, str]] | None = None,
    cache_dir: str | Path | None = None,
    bindings: dict[str, str] | None = None,
    rebind_backends: bool = True,
) -> Expr:
    """Build a composed expression from catalog entries and/or inline code."""

    if bindings:
        if rename_map:
            raise click.UsageError(
                "--rename-params is not supported in multi-root (-e) mode."
            )
        import xorq.api as xo  # noqa: PLC0415
        from xorq.catalog.bind import compose_join  # noqa: PLC0415

        binding_exprs = {}
        if entries:
            primary = "source"
            linear_entries = (
                (bindings[primary], *entries) if primary in bindings else entries
            )
            binding_exprs[primary] = _compose_linear_expr(
                catalog,
                linear_entries,
                None,
                cache_dir=cache_dir,
                source_alias=primary,
            )
            if code is None and len(bindings) == 1:
                return binding_exprs[primary]

        # The CLI is the user-facing layer, so the ``xo`` facade is supplied
        # to the inline-code eval namespace here rather than imported inside
        # the catalog library (compose_join).
        return compose_join(
            catalog,
            bindings,
            code,
            cache_dir=cache_dir,
            rebind_backends=rebind_backends,
            eval_namespace={"xo": xo},
            binding_exprs=binding_exprs,
        )

    return _compose_linear_expr(catalog, entries, code, rename_map, cache_dir)


def _assert_requirements_identical(entry_reqs):
    distinct = {content for _, content in entry_reqs}
    if len(distinct) <= 1:
        return
    names = ", ".join(repr(name) for name, _ in entry_reqs)
    raise click.ClickException(
        "requirements.txt differs across entries; all entries must share an "
        f"identical requirements.txt. Mismatched entries: {names}."
    )


def _stage_bundle_into_build(bundle, build_path):
    for w in bundle.wheel_paths:
        dst = build_path / w.name
        if not dst.exists():
            shutil.copy2(w, dst)
    shutil.copy2(bundle.requirements_path, build_path / DumpFiles.requirements)


def _extract_wheel(zf, member, harvest_dir, seen_wheels, entry_name):
    base = Path(member).name
    if seen_wheels is not None:
        info = zf.getinfo(member)
        sig = (info.file_size, info.CRC)
        if base in seen_wheels:
            if seen_wheels[base] != sig:
                raise click.ClickException(
                    f"wheel collision: {base!r} differs in entry {entry_name!r}"
                )
            return None
        seen_wheels[base] = sig
    target = harvest_dir / base
    target.write_bytes(zf.read(member))
    return target


def _harvest_entry_from_zip(zf, harvest_dir, entry_name=None, seen_wheels=None):
    from xorq.ibis_yaml.packager import (  # noqa: PLC0415
        _python_minor_from_metadata_text,
    )

    members = sorted(zf.namelist())

    wheel_paths = [
        path
        for m in members
        if Path(m).name.endswith(".whl")
        if (path := _extract_wheel(zf, m, harvest_dir, seen_wheels, entry_name))
        is not None
    ]

    req_bytes = next(
        (zf.read(m) for m in members if Path(m).name == DumpFiles.requirements),
        None,
    )

    meta_member = next(
        (m for m in members if Path(m).name == DumpFiles.build_metadata),
        None,
    )
    python_pin = (
        _python_minor_from_metadata_text(zf.read(meta_member).decode())
        if meta_member
        else None
    )

    return wheel_paths, req_bytes, python_pin


@contextmanager
def _entry_run_bundle(catalog, entries):
    from xorq.common.utils.otel_utils import tracer  # noqa: PLC0415
    from xorq.ibis_yaml.packager import JointBundle  # noqa: PLC0415

    if not entries:
        raise click.ClickException("at least one entry is required")

    if len(entries) > 1:
        click.echo(f"Composing wheels from {len(entries)} entries...", err=True)

    with tempfile.TemporaryDirectory() as harvest_str:
        harvest_dir = Path(harvest_str)

        with tracer.start_as_current_span("catalog.compose_bundle") as span:
            span.set_attribute("entries", entries)
            seen_wheels: dict[str, tuple[int, int]] = {}
            all_wheel_paths = []
            req_contents = []
            python_pins = []

            for entry in entries:
                ce = _get_catalog_entry(catalog, entry)
                if not ce.is_content_local:
                    ce.fetch()
                with zipfile.ZipFile(ce.catalog_path) as zf:
                    wp, req_bytes, py_pin = _harvest_entry_from_zip(
                        zf, harvest_dir, entry, seen_wheels
                    )
                    all_wheel_paths.extend(wp)
                    if req_bytes is None:
                        raise click.ClickException(
                            f"entry {entry!r} has no requirements.txt"
                        )
                    req_contents.append((entry, req_bytes))
                    python_pins.append((entry, py_pin))

            if not all_wheel_paths:
                raise click.ClickException("no wheels found in entries")
            _assert_requirements_identical(req_contents)
            req_path = harvest_dir / DumpFiles.requirements
            req_path.write_bytes(req_contents[0][1])

            distinct_pins = {pin for _, pin in python_pins if pin is not None}
            unpinned = [e for e, pin in python_pins if pin is None]
            if len(distinct_pins) > 1:
                detail = ", ".join(f"{e!r}={pin}" for e, pin in python_pins)
                raise click.ClickException(
                    f"entries built on different Python minors: {detail}"
                )
            if distinct_pins and unpinned:
                joint = next(iter(distinct_pins))
                click.echo(
                    f"WARNING: entries {unpinned} lack a Python minor pin; "
                    f"running under {joint} but cloudpickled UDFs in those "
                    f"archives may SIGSEGV if built on a different minor.",
                    err=True,
                )
            joint_python = next(iter(distinct_pins), None)
            span.set_attribute("wheel_count", len(all_wheel_paths))
            span.set_attribute("python_version", joint_python or "")
            yield JointBundle(
                wheel_paths=tuple(all_wheel_paths),
                requirements_path=req_path,
                python_version=joint_python,
            )


@contextmanager
def _uv_reinvoke_xorq_cli(catalog, entries, *inner_args, **uv_kwargs):
    from xorq.ibis_yaml.packager import uv_tool_run  # noqa: PLC0415

    uv_kwargs.setdefault("capture_output", False)
    with _entry_run_bundle(catalog, entries) as bundle:
        result = uv_tool_run(
            *inner_args,
            python_version=bundle.python_version,
            with_=bundle.wheel_paths,
            with_requirements=bundle.requirements_path,
            **uv_kwargs,
        )
        yield result, bundle


def _forward_ctx_params(ctx, *, exclude=frozenset()):
    """Serialize non-default Click params back to CLI args.

    New options added to the command are automatically forwarded;
    only params in exclude (Python names) are suppressed.

    Handles bool flags, File options, and plain scalar/multiple options.
    Does not handle click.Choice(case_sensitive=False) or count params.
    """

    def _param_to_args(param):
        if param.name in exclude or isinstance(param, click.Argument):
            return
        value = ctx.params.get(param.name)
        if value is None:
            return
        if param.is_flag:
            if value != param.default:
                opts = param.opts if value else param.secondary_opts
                if opts:
                    yield opts[0]
            return
        if isinstance(param.type, click.File):
            name = getattr(value, "name", None)
            if not name or name == param.default or name.startswith("<"):
                return
            yield param.opts[0]
            yield str(Path(name).resolve())
            return
        if value == param.default:
            return
        opt = next((o for o in param.opts if o.startswith("--")), param.opts[0])
        for v in value if param.multiple else (value,):
            yield opt
            yield str(v)

    return tuple(
        itertools.chain.from_iterable(_param_to_args(p) for p in ctx.command.params)
    )


def _compose_via_reinvoke(
    ctx: click.Context,
    catalog: Catalog,
    entries: tuple[str, ...],
    involved_entries: tuple[str, ...] | None = None,
) -> Path | None:
    # entries are the positional argv tokens (empty in multi-root mode);
    # involved_entries are the catalog entries whose wheels to harvest.
    if involved_entries is None:
        involved_entries = entries
    dry_run = ctx.params.get("dry_run", False)
    forwarded = _forward_ctx_params(
        ctx, exclude={"use_this_venv", "emit_build_path_to", "alias"}
    )
    build_path_out_fd, build_path_out = tempfile.mkstemp(
        prefix="xorq-compose-build-path-", suffix=".txt"
    )
    os.close(build_path_out_fd)
    try:
        inner_cmd = (
            "xorq",
            "catalog",
            "--path",
            str(catalog.repo_path),
            "compose",
            *entries,
            *forwarded,
            "--use-this-venv",
            "--emit-build-path-to",
            build_path_out,
        )
        try:
            with _uv_reinvoke_xorq_cli(
                catalog, involved_entries, *inner_cmd, capture_output=True
            ) as (result, bundle):
                if dry_run:
                    click.echo(result.stdout, nl=False)
                    return None
                build_path_str = Path(build_path_out).read_text().strip()
                if not build_path_str:
                    raise click.ClickException(
                        f"inner compose did not write a build_path; "
                        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                    )
                build_path = Path(build_path_str)
                if not build_path.exists():
                    raise click.ClickException(
                        f"inner compose returned nonexistent build_path "
                        f"{build_path!r}; stdout:\n{result.stdout}"
                    )
                _stage_bundle_into_build(bundle, build_path)
                return build_path
        except subprocess.CalledProcessError as e:
            raise click.ClickException(
                f"inner compose failed (exit {e.returncode});\n"
                f"stdout:\n{e.stdout}\nstderr:\n{e.stderr}"
            ) from e
    finally:
        Path(build_path_out).unlink(missing_ok=True)


@cli.command("compose")
@click.argument("entries", nargs=-1, shell_complete=_complete_entry_or_alias_names)
@entry_option
@rebind_backends_option
@sync_option
@code_option
@click.option(
    "-a",
    "--alias",
    default=None,
    help="Also register this alias for the cataloged entry.",
)
@cache_dir_option
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show composition plan without building.",
)
@rename_params_option
@click.option(
    "--use-this-venv/--no-use-this-venv",
    default=False,
    help=(
        "Load and build the expression in the current Python environment "
        "instead of spawning `uv tool run` on the entries' joint bundle. "
        "Faster (no subprocess), but only correct when the calling venv "
        "already has every package each entry's wheel depends on. Default "
        "is the isolated `uv tool run` path."
    ),
)
@click.option(
    "--emit-build-path-to",
    type=click.Path(),
    default=None,
    hidden=True,
    help=(
        "Internal: after build_expr, write the build_path to the given file "
        "and exit without merging wheels or cataloging."
    ),
)
@click.pass_context
def compose(  # xorq-style: disable=type-annotations
    ctx,
    entries,
    raw_entries,
    rebind_backends,
    sync,
    code,
    alias,
    cache_dir,
    dry_run,
    raw_rename_params,
    use_this_venv,
    emit_build_path_to,
):
    """Compose entries into a new expression and persist it to the catalog.

    Assembles one or more catalog entries (and optionally inline Ibis
    code) into a new expression, builds it, and always catalogs the
    result. To execute a composed expression for data output without
    persisting, use `xorq catalog run` instead.

    \b
    Arguments:
      ENTRIES  One or more entries (names or aliases); the first is the
               source and subsequent entries apply as transforms.

    \b
    Examples:
      # Compose source + transform and catalog the result
      xorq catalog compose source-table transform-step --alias prod-pipeline
      # Add inline transformation code
      xorq catalog compose source-table -c "source.filter(source.amount > 100)" --alias high-value
      # Preview without building
      xorq catalog compose source-table transform-step --dry-run
      # Rename a parameter on one entry of the composition
      xorq catalog compose src trn --rename-params trn,threshold,cutoff
    """
    from xorq.common.utils.otel_utils import tracer  # noqa: PLC0415

    with tracer.start_as_current_span("catalog.compose") as span:
        span.set_attributes({"entries": entries, "has_code": code is not None})
        with click_context_catalog(ctx):
            catalog = ctx.obj.make_catalog(init=False)
            bindings = _resolve_mode(entries, raw_entries, code)
            involved_entries = _involved_entries(entries, bindings)
            if not involved_entries:
                raise click.UsageError("At least one entry is required.")

            if not use_this_venv:
                span.set_attribute("path", "uv-reinvoke")
                build_path = _compose_via_reinvoke(
                    ctx, catalog, entries, involved_entries=involved_entries
                )
                if build_path is None:
                    return
            else:
                span.set_attribute("path", "in-process")
                rename_map = (
                    _parse_rename_params(raw_rename_params)
                    if raw_rename_params
                    else None
                )
                expr = _compose_expr(
                    catalog,
                    entries,
                    code,
                    rename_map=rename_map,
                    cache_dir=cache_dir if bindings else None,
                    bindings=bindings,
                    rebind_backends=rebind_backends,
                )

                if dry_run:
                    click.echo("Dry run — composition plan:")
                    if bindings:
                        click.echo(
                            "  Bindings: "
                            + ", ".join(f"{k}={v}" for k, v in bindings.items())
                        )
                    else:
                        click.echo(f"  Entries: {' -> '.join(entries)}")
                    if code:
                        click.echo(f"  Code: {code}")
                    sch = expr.schema()
                    click.echo("  Schema:")
                    for col, dtype in sch.items():
                        click.echo(f"    {col:<24} {dtype}")
                    return

                from xorq.ibis_yaml.compiler import build_expr  # noqa: PLC0415

                build_kwargs = (
                    {} if cache_dir is None else {"cache_dir": Path(cache_dir)}
                )
                build_path = build_expr(expr, **build_kwargs)

                if emit_build_path_to:
                    Path(emit_build_path_to).write_text(str(build_path))
                    return

                with _entry_run_bundle(catalog, involved_entries) as bundle:
                    _stage_bundle_into_build(bundle, build_path)
            entry_name = build_path.name
            aliases = (alias,) if alias else ()
            alias_existed = alias and catalog.catalog_yaml.contains_alias(alias)
            entry_existed = catalog.contains(entry_name)
            catalog.add(build_path, sync=sync, aliases=aliases, exist_ok=True)
            label = alias or entry_name
            if entry_existed:
                if alias and not alias_existed:
                    click.echo(
                        f"Entry {entry_name!r} already exists; alias {alias!r} added",
                        err=True,
                    )
                elif alias:
                    click.echo(
                        f"Entry {entry_name!r} already exists;"
                        f" alias {alias!r} already set",
                        err=True,
                    )
                else:
                    click.echo(f"Entry {entry_name!r} already exists", err=True)
            else:
                click.echo(f"Cataloged as {label!r}", err=True)
            span.set_attribute("cataloged", label)


def _resolve_single_entry(
    catalog, entry, code, instream, rename_map, span, cache_dir=None
):
    """Resolve a single catalog entry to an expression."""
    catalog_entry = _get_catalog_entry(catalog, entry)

    span.set_attribute("kind", str(catalog_entry.kind))
    expr = _eval_entry(catalog_entry, code, instream, cache_dir=cache_dir)

    if rename_map and entry in rename_map:
        from xorq.common.utils.graph_utils import rename_params  # noqa: PLC0415

        expr = rename_params(expr, rename_map[entry])

    if catalog_entry.kind is ExprKind.UnboundExpr:
        span.set_attribute("piped_stdin", True)

    return expr


_EXPR_MODIFYING_PARAMS = frozenset({"code", "limit", "raw_params", "raw_rename_params"})


def _has_expr_modifications(ctx):
    """True when the user passed flags that modify the expression."""
    p = ctx.params
    if not p.get("fuse", True):
        return True
    if p.get("limit") is not None:
        return True
    return bool(
        p.get("code")
        or p.get("raw_entries")
        or p.get("raw_params")
        or p.get("raw_rename_params")
    )


def _involved_entries(
    entries: tuple[str, ...], bindings: dict[str, str]
) -> tuple[str, ...]:
    return (*bindings.values(), *entries) if bindings else entries


def _log_run_metrics(rl, span, prefix, elapsed, output_format, output_path):
    from xorq.common.utils.logging_utils import RunLogger  # noqa: PLC0415

    rl.log_span_event(
        span,
        f"{prefix}.done",
        {"elapsed_s": round(elapsed, 3), "output_format": str(output_format)},
    )
    file_metrics = RunLogger._compute_file_metrics(output_format, output_path)
    if file_metrics:
        rl.log_span_event(span, f"{prefix}.output_written", file_metrics)


def _reinvoke_and_log(
    ctx: click.Context,
    catalog: Catalog,
    entries: tuple[str, ...],
    span: Span,
    rl: RunLogger,
    involved_entries: tuple[str, ...] | None = None,
) -> None:
    """Reinvoke the current Click command in the entries' pinned env and log metrics."""
    from xorq.common.utils.profile_utils import timed  # noqa: PLC0415

    # entries are the positional argv tokens (empty in multi-root mode);
    # involved_entries are the catalog entries whose wheels to harvest.
    if involved_entries is None:
        involved_entries = entries
    span.set_attribute("path", "uv-reinvoke")
    forwarded = _forward_ctx_params(ctx, exclude={"use_this_venv"})
    inner_cmd = (
        "xorq",
        "catalog",
        "--path",
        str(catalog.repo_path),
        ctx.info_name,
        *entries,
        *forwarded,
        "--use-this-venv",
    )
    with (
        timed() as get_elapsed,
        _uv_reinvoke_xorq_cli(catalog, involved_entries, *inner_cmd),
    ):
        pass

    span_prefix = f"catalog_{ctx.info_name.replace('-', '_')}"
    _log_run_metrics(
        rl,
        span,
        span_prefix,
        get_elapsed(),
        ctx.params.get("output_format"),
        ctx.params.get("output_path"),
    )


def _resolve_and_execute(
    ctx, catalog, span, rl, span_prefix, *, expr_transform=None, cache_dir=None
):
    from xorq.cli import _apply_cli_params, arbitrate_output_format  # noqa: PLC0415
    from xorq.common.utils.profile_utils import timed  # noqa: PLC0415

    p = ctx.params
    entries = p["entries"]
    code = p.get("code")
    instream = p.get("instream")
    fuse = p.get("fuse", True)
    raw_rename_params = p.get("raw_rename_params", ())
    raw_params = p.get("raw_params", ())
    limit = p.get("limit")
    output_path = p.get("output_path")
    output_format = p.get("output_format")

    raw_entries = p.get("raw_entries", ())
    rebind_backends = p.get("rebind_backends", True)
    bindings = _resolve_mode(entries, raw_entries, code)

    rename_map = _parse_rename_params(raw_rename_params) if raw_rename_params else None

    span.set_attribute("path", "in-process")
    with timed() as get_elapsed:
        if bindings:
            expr = _compose_expr(
                catalog,
                entries,
                code,
                rename_map=rename_map,
                cache_dir=cache_dir,
                bindings=bindings,
                rebind_backends=rebind_backends,
            )
        elif len(entries) > 1:
            expr = _compose_expr(
                catalog, entries, code, rename_map=rename_map, cache_dir=cache_dir
            )
        else:
            (entry,) = entries
            expr = _resolve_single_entry(
                catalog,
                entry,
                code,
                instream,
                rename_map,
                span,
                cache_dir=cache_dir,
            )
        rl.log_span_event(
            span,
            f"{span_prefix}.expr_loaded",
            {"elapsed_s": round(get_elapsed(), 3)},
        )

    expr = _apply_cli_params(expr, raw_params)

    # Catalog fusion strips RemoteTable wrappers for single-backend
    # push-down, but doing so over an explicit cross-backend
    # into_backend(...) transport collapses the transport and breaks
    # execution. In multi-root mode, only fuse when the (post-rebind)
    # expression lives on a single backend.
    if fuse and not (bindings and _is_cross_backend(expr)):
        expr = expr.ls.fused
    if expr_transform is not None:
        expr = expr_transform(expr)
    if limit is not None:
        expr = expr.limit(limit)

    with timed() as get_elapsed:
        arbitrate_output_format(expr, output_path, output_format)

    _log_run_metrics(rl, span, span_prefix, get_elapsed(), output_format, output_path)


@cli.command("run")
@click.argument("entries", nargs=-1, shell_complete=_complete_entry_or_alias_names)
@entry_option
@rebind_backends_option
@code_option
@output_options
@limit_option
@click.option(
    "-i",
    "--instream",
    type=click.File("rb"),
    default="-",
    show_default="stdin",
    help="Stream to read Arrow IPC record batches from.",
)
@fuse_option
@rename_params_option
@params_option
@cache_dir_option
@click.option(
    "--use-this-venv/--no-use-this-venv",
    default=False,
    help=(
        "Execute in the current Python environment instead of spawning "
        "`uv tool run` on the entry's pinned env. Faster (no subprocess + "
        "uv venv lookup) but only correct when the calling venv already has "
        "every package the expression needs (Xorq itself plus any UDFs "
        "from the entries' wheels). Default is the isolated `uv tool run` "
        "path."
    ),
)
@click.pass_context
def run(  # xorq-style: disable=type-annotations
    ctx,
    entries,
    raw_entries,
    rebind_backends,
    code,
    output_path,
    output_format,
    limit,
    instream,
    fuse,
    raw_rename_params,
    raw_params,
    cache_dir,
    use_this_venv,
):
    """Compose and execute catalog entries, writing data to disk or stdout.

    A single entry runs directly; multiple entries compose as
    source + transforms. Unbound entries can have their input streamed in
    over Arrow IPC. To persist a composed expression back to the catalog
    instead of executing it, use `xorq catalog compose`.

    \b
    Arguments:
      ENTRIES  One or more entry names or aliases; multiple entries
               compose as source + transforms.

    \b
    Examples:
      # Run a single entry directly
      xorq catalog run src -o - -f csv
      # Compose source + transform and execute
      xorq catalog run src trn -o - -f csv
      # Add inline transformation code
      xorq catalog run src trn -c "source.filter(source.amount > 100)" -o - -f csv
      # Pipe Arrow data into an unbound entry
      xorq catalog run src -o - -f arrow | xorq catalog run trn -o - -f csv
      # Override an expression parameter
      xorq catalog run pipeline -p threshold=0.5 -o results.parquet
    """
    from xorq.cli import _get_cache_dir  # noqa: PLC0415
    from xorq.common.utils.logging_utils import RunLogger  # noqa: PLC0415
    from xorq.common.utils.otel_utils import tracer  # noqa: PLC0415
    from xorq.common.utils.profile_utils import timed  # noqa: PLC0415

    cache_dir = _get_cache_dir(cache_dir)

    with tracer.start_as_current_span("catalog.run") as span:
        span.set_attributes({"entries": entries, "has_code": code is not None})
        with click_context_catalog(ctx):
            bindings = _resolve_mode(entries, raw_entries, code)
            involved_entries = _involved_entries(entries, bindings)
            if not involved_entries:
                raise click.UsageError("At least one entry is required.")

            expr_hash = "+".join(involved_entries)
            run_params = (
                ("expr_hash", expr_hash),
                ("entries", ",".join(involved_entries)),
                ("has_code", str(code is not None)),
                ("output_format", str(output_format)),
                ("output_path", str(output_path)),
                ("fuse", str(fuse)),
                ("limit", limit),
                ("params", ",".join(raw_params)),
            )

            with RunLogger.from_expr_hash(
                expr_hash, params_tuple=run_params, span=span
            ) as rl:
                rl.log_span_event(span, "catalog_run.start", dict(run_params))

                catalog = ctx.obj.make_catalog(init=False)

                # Fast path: single unmodified entry run from archive.
                if (
                    not use_this_venv
                    and len(entries) == 1
                    and not _has_expr_modifications(ctx)
                ):
                    from xorq.catalog.zip_utils import (  # noqa: PLC0415
                        extract_build_zip_context,
                    )
                    from xorq.ibis_yaml.packager import (  # noqa: PLC0415
                        PackagedRunner,
                    )

                    catalog_entry = _get_catalog_entry(catalog, entries[0])
                    if catalog_entry.kind is not ExprKind.UnboundExpr:
                        if not catalog_entry.is_content_local:
                            catalog_entry.fetch()
                        span.set_attribute("kind", str(catalog_entry.kind))
                        span.set_attribute("path", "archive")
                        with (
                            timed() as get_elapsed,
                            extract_build_zip_context(
                                catalog_entry.catalog_path
                            ) as build_path,
                        ):
                            PackagedRunner(
                                build_path,
                                cache_dir=cache_dir,
                                output_path=output_path,
                                output_format=output_format,
                            ).run()
                        _log_run_metrics(
                            rl,
                            span,
                            "catalog_run",
                            get_elapsed(),
                            output_format,
                            output_path,
                        )
                        return

                if not use_this_venv:
                    _reinvoke_and_log(
                        ctx,
                        catalog,
                        entries,
                        span,
                        rl,
                        involved_entries=involved_entries,
                    )
                    return

                _resolve_and_execute(
                    ctx, catalog, span, rl, "catalog_run", cache_dir=cache_dir
                )


@cli.command("run-cached")
@click.argument("entries", nargs=-1, shell_complete=_complete_entry_or_alias_names)
@entry_option
@rebind_backends_option
@code_option
@output_options
@limit_option
@click.option(
    "-i",
    "--instream",
    type=click.File("rb"),
    default="-",
    show_default="stdin",
    help="Stream to read Arrow IPC record batches from.",
)
@fuse_option
@rename_params_option
@params_option
@cache_dir_option
@cache_strategy_options
@click.option(
    "--use-this-venv/--no-use-this-venv",
    default=False,
    help=(
        "Execute in the current Python environment instead of spawning "
        "`uv tool run` on the entry's pinned env. Faster (no subprocess + "
        "uv venv lookup) but only correct when the calling venv already has "
        "every package the expression needs (Xorq itself plus any UDFs "
        "from the entries' wheels). Default is the isolated `uv tool run` "
        "path."
    ),
)
@click.pass_context
def run_cached(  # xorq-style: disable=type-annotations
    ctx,
    entries,
    raw_entries,
    rebind_backends,
    code,
    output_path,
    output_format,
    limit,
    instream,
    fuse,
    raw_rename_params,
    raw_params,
    cache_dir,
    cache_type,
    ttl,
    use_this_venv,
):
    """Compose and execute catalog entries with a parquet cache wrapper.

    Same semantics as `xorq catalog run`, but wraps the resulting
    expression with a cache so subsequent invocations short-circuit when
    inputs haven't changed (ParquetCache by default; snapshot and TTL
    variants via `--cache-type` and `--ttl`—see `xorq run-cached` for the
    strategies).

    \b
    Arguments:
      ENTRIES  One or more entry names or aliases.

    \b
    Examples:
      # Default modification-time cache
      xorq catalog run-cached src trn --cache-dir ./cache -o results.parquet
      # Snapshot cache with a 1-hour TTL
      xorq catalog run-cached pipeline --cache-type snapshot --ttl 3600 -o results.parquet
    """
    from xorq.caching import (  # noqa: PLC0415
        ParquetCache,
        ParquetSnapshotCache,
        ParquetTTLSnapshotCache,
    )
    from xorq.cli import _get_cache_dir  # noqa: PLC0415
    from xorq.common.utils.logging_utils import RunLogger  # noqa: PLC0415
    from xorq.common.utils.otel_utils import tracer  # noqa: PLC0415

    with tracer.start_as_current_span("catalog.run_cached") as span:
        span.set_attributes({"entries": entries, "has_code": code is not None})
        with click_context_catalog(ctx):
            bindings = _resolve_mode(entries, raw_entries, code)
            involved_entries = _involved_entries(entries, bindings)
            if not involved_entries:
                raise click.UsageError("At least one entry is required.")

            resolved_cache_dir = _get_cache_dir(cache_dir)

            expr_hash = "+".join(involved_entries)
            run_params = (
                ("expr_hash", expr_hash),
                ("entries", ",".join(involved_entries)),
                ("has_code", str(code is not None)),
                ("output_format", str(output_format)),
                ("output_path", str(output_path)),
                ("fuse", str(fuse)),
                ("limit", limit),
                ("params", ",".join(raw_params)),
                ("cache_type", cache_type),
                ("ttl", ttl),
            )

            with RunLogger.from_expr_hash(
                expr_hash, params_tuple=run_params, span=span
            ) as rl:
                rl.log_span_event(span, "catalog_run_cached.start", dict(run_params))

                catalog = ctx.obj.make_catalog(init=False)

                if not use_this_venv:
                    _reinvoke_and_log(
                        ctx,
                        catalog,
                        entries,
                        span,
                        rl,
                        involved_entries=involved_entries,
                    )
                    return

                def _wrap_cache(expr):
                    match (cache_type, ttl):
                        case ("modification-time", None):
                            cache = ParquetCache.from_kwargs(
                                base_path=resolved_cache_dir
                            )
                        case (_, int(seconds)):
                            ttl_delta = datetime.timedelta(seconds=seconds)
                            cache = ParquetTTLSnapshotCache.from_kwargs(
                                base_path=resolved_cache_dir, ttl=ttl_delta
                            )
                        case ("snapshot", None):
                            cache = ParquetSnapshotCache.from_kwargs(
                                base_path=resolved_cache_dir
                            )
                        case _:
                            raise click.BadParameter(
                                f"Unknown cache type: {cache_type!r}. "
                                "Must be 'modification-time' or 'snapshot'."
                            )
                    return expr.cache(cache=cache)

                _resolve_and_execute(
                    ctx,
                    catalog,
                    span,
                    rl,
                    "catalog_run_cached",
                    expr_transform=_wrap_cache,
                    cache_dir=resolved_cache_dir,
                )


@cli.command("serve-unbound")
@click.argument("entry", shell_complete=_complete_entry_or_alias_names)
@unbind_options
@serve_options
@code_option
@fuse_option
@rename_params_option
@params_option
@cache_dir_option
@click.pass_context
def serve_unbound(
    ctx,
    entry,
    to_unbind_hash,
    to_unbind_tag,
    typ,
    host,
    port,
    prometheus_port,
    code,
    fuse,
    raw_rename_params,
    raw_params,
    cache_dir,
):
    """Resolve a catalog entry, unbind a node, and serve it via Flight.

    The catalog-aware counterpart to the top-level `xorq serve-unbound`:
    instead of pointing at a build directory, you reference an entry by
    name or alias and Xorq resolves it from the active catalog. Clients
    stream record batches to the endpoint to drive the computation.

    \b
    Arguments:
      ENTRY  Catalog entry name or alias to serve.

    \b
    Examples:
      # Serve an aliased entry, unbinding by tag
      xorq catalog serve-unbound flights-model --to-unbind-tag source_input --host 0.0.0.0 --port 8001
      # Bind a runtime parameter
      xorq catalog serve-unbound scorer --params threshold=0.5
    """
    import xorq.expr.relations  # noqa: PLC0415
    import xorq.flight  # noqa: PLC0415
    from xorq.caching.strategy import SnapshotStrategy  # noqa: PLC0415
    from xorq.cli import _apply_cli_params, _get_cache_dir  # noqa: PLC0415
    from xorq.common.utils.logging_utils import get_print_logger  # noqa: PLC0415
    from xorq.common.utils.node_utils import expr_to_unbound  # noqa: PLC0415
    from xorq.common.utils.otel_utils import tracer  # noqa: PLC0415

    logger = get_print_logger()
    cache_dir = _get_cache_dir(cache_dir)

    with tracer.start_as_current_span("catalog.serve_unbound") as span:
        span.set_attributes({"entry": entry, "has_code": code is not None})
        with click_context_catalog(ctx):
            catalog = ctx.obj.make_catalog(init=False)
            catalog_entry = _get_catalog_entry(catalog, entry)

            if not catalog_entry.is_content_local:
                catalog_entry.fetch()

            expr = catalog_entry.load_expr(cache_dir=cache_dir)

            if code is not None:
                from xorq.catalog.bind import _eval_code  # noqa: PLC0415

                expr = _eval_code(code, expr)

            rename_map = (
                _parse_rename_params(raw_rename_params) if raw_rename_params else None
            )
            if rename_map and entry in rename_map:
                from xorq.common.utils.graph_utils import rename_params  # noqa: PLC0415

                expr = rename_params(expr, rename_map[entry])

            expr = _apply_cli_params(expr, raw_params)

            if fuse:
                expr = expr.ls.fused

            try:
                from xorq.flight.metrics import setup_console_metrics  # noqa: PLC0415

                setup_console_metrics(prometheus_port=prometheus_port)
            except ImportError:
                logger.warning(
                    "Metrics support requires 'opentelemetry-sdk' and console exporter"
                )

            unbound_expr = expr_to_unbound(
                expr,
                hash=to_unbind_hash,
                tag=to_unbind_tag,
                typs=typ,
                strategy=SnapshotStrategy(),
            )
            flight_url = xorq.flight.FlightUrl(host=host, port=port)
            make_server = partial(xorq.flight.FlightServer, flight_url=flight_url)
            server, _ = xorq.expr.relations.flight_serve_unbound(
                unbound_expr, make_server=make_server
            )
            span.add_event(
                "server_started", {"flight_url": str(flight_url.to_location())}
            )

    # Span covers startup only; server.wait() runs outside to avoid a never-exported span.
    logger.info(f"Serving entry {entry!r} on {flight_url.to_location()}")
    server.wait()


if __name__ == "__main__":
    cli()
