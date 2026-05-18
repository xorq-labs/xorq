import datetime
import itertools
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from contextlib import contextmanager
from functools import cache, partial, reduce
from pathlib import Path
from types import SimpleNamespace

import click

from xorq.cli_options import (
    cache_dir_option,
    cache_strategy_options,
    code_option,
    env_options,
    fuse_option,
    gcs_option,
    json_option,
    limit_option,
    output_options,
    params_options,
    rename_params_option,
    sync_option,
)


def click_handler(e):
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
def click_context(ctx, *typs):
    try:
        yield
    except click.ClickException:
        raise
    except typs as e:
        if _pdb_active(ctx):
            raise
        click_handler(e)


@contextmanager
def click_context_catalog(ctx):
    from git import NoSuchPathError

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
    from xorq.catalog.catalog import Catalog

    catalog_ctx = ctx.parent
    return Catalog.from_kwargs(
        name=catalog_ctx.params.get("name"),
        path=catalog_ctx.params.get("path"),
        url=catalog_ctx.params.get("url"),
        root_repo=catalog_ctx.params.get("root_repo"),
        init=False,
    )


def _complete_entry_names(ctx, param, incomplete):
    from click.shell_completion import CompletionItem

    try:
        catalog = _make_catalog_for_completion(ctx)
        return [CompletionItem(n) for n in catalog.list() if n.startswith(incomplete)]
    except Exception:
        return []


def _complete_alias_names(ctx, param, incomplete):
    from click.shell_completion import CompletionItem

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
    help="Remote repo url to clone",
)
@click.option(
    "-r",
    "--root-repo",
    default=None,
    type=click.Path(file_okay=False, exists=True),
    help="Repo root to add this catalog to as a submodule",
)
@click.option(
    "--init/--no-init",
    default=None,
    help="Initialize the repo (default: auto).",
)
@click.pass_context
def cli(ctx, name, path, url, root_repo, init):
    """Manage xorq build-artifact catalogs."""
    from xorq.catalog.catalog import Catalog

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
@click.option("--refresh", default=10, type=float, help="Refresh interval in seconds.")
@click.pass_context
def tui(ctx, refresh):
    """Launch terminal UI."""
    with click_context_catalog(ctx):
        ctx.obj.make_catalog(init=False)  # validate catalog exists
    from xorq.catalog.tui import CatalogTUI

    app = CatalogTUI(
        partial(ctx.obj.make_catalog, init=False), refresh_interval=refresh
    )
    app.run()


def _resolve_annex_option(env_file, env_prefix, gcs):
    """Return a RemoteConfig from CLI options, or None for plain git."""
    if env_file and env_prefix:
        raise click.UsageError("--env-file and --env-prefix are mutually exclusive.")
    if env_file:
        from xorq.catalog.annex import remote_config_from_env_file

        return remote_config_from_env_file(env_file, gcs=gcs)
    elif env_prefix:
        from xorq.catalog.annex import remote_config_from_prefix

        return remote_config_from_prefix(env_prefix, gcs=gcs)
    return None


@cli.command()
@env_options
@gcs_option
@click.option(
    "--remote-url",
    default=None,
    help="Git remote URL (sets origin).",
)
@click.pass_context
def init(ctx, env_file, env_prefix, gcs, remote_url):
    """Initialize a new catalog."""
    with click_context_catalog(ctx):
        annex = _resolve_annex_option(env_file, env_prefix, gcs)
        try:
            catalog = ctx.obj.make_catalog(init=True, annex=annex)
        except AssertionError as err:
            # init_repo_path asserts the path does not already exist
            probe = ctx.obj.make_catalog(init=False)
            raise click.ClickException(
                f"Catalog already exists at {probe.repo_path}"
            ) from err
        click.echo(f"Initialized catalog at {catalog.repo_path}")
        if remote_url:
            from xorq.catalog.constants import DEFAULT_REMOTE

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
    help="Alias(es) to create for the added entry (repeatable).",
)
@click.pass_context
def add(ctx, paths, sync, aliases):
    """Add entries from archive files or build directories."""
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
    """Remove entries by name."""
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
    """Add an alias for an entry."""
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
    """Remove one or more aliases."""
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
@click.option("--kind/--no-kind", default=False, help="Show the kind column.")
@click.pass_context
def list_entries(ctx, kind):
    """List all entries."""
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
    """List all aliases."""
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        aliases = catalog.list_aliases() or ("No aliases.",)
        for alias in aliases:
            click.echo(alias)


@cli.command()
@click.pass_context
def info(ctx):
    """Show catalog metadata: path, remotes, entry/alias counts."""
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
    """Show or change the persisted default catalog name."""
    from xorq.catalog.catalog import Catalog
    from xorq.catalog.constants import DEFAULT_CATALOG_CONFIG
    from xorq.vendor.ibis.config import env_config

    if set_name and unset:
        raise click.UsageError("--set and --unset are mutually exclusive.")

    if set_name:
        DEFAULT_CATALOG_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_CATALOG_CONFIG.write_text(set_name + "\n")
        click.echo(f"Default catalog set to {set_name!r}")
    elif unset:
        try:
            DEFAULT_CATALOG_CONFIG.unlink()
            click.echo("Default catalog unset (reverted to 'default')")
        except FileNotFoundError:
            click.echo("No persisted default to unset.")
    else:
        name = Catalog._resolve_default_name()
        if env_config.XORQ_DEFAULT_CATALOG:
            source = "env (XORQ_DEFAULT_CATALOG)"
        elif DEFAULT_CATALOG_CONFIG.exists():
            source = f"config ({DEFAULT_CATALOG_CONFIG})"
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
    help="Output directory (default: current directory).",
)
@click.pass_context
def get(ctx, name, output):
    """Export an entry's archive to a directory."""
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        result = catalog.get_zip(name, dir_path=output)
        click.echo(f"Exported to {result}")


@cli.command()
@click.pass_context
def push(ctx):
    """Push catalog to the configured git remote."""
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        catalog.push()
        click.echo("Pushed.")


@cli.command()
@click.pass_context
def pull(ctx):
    """Pull catalog from the configured git remote."""
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        catalog.pull()
        click.echo("Pulled.")


@cli.command()
@click.pass_context
def sync(ctx):
    """Pull then push."""
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        catalog.sync()
        click.echo("Synced.")


@cli.command("set-remote")
@click.argument("url")
@click.option(
    "--name",
    default=None,
    help="Remote name (defaults to 'origin').",
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
    already configured, this command refuses unless ``--force`` is passed
    — guarding against typos that would silently delete the configured
    remote.
    """
    from xorq.catalog.constants import DEFAULT_REMOTE
    from xorq.catalog.exceptions import CatalogConfigurationError

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
    """Clone a catalog from a remote URL."""
    from xorq.catalog.catalog import Catalog

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
    """Show schema of a catalog entry (name or alias)."""
    from xorq.ibis_yaml.enums import ExprKind  # noqa: PLC0415

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
    """Show full metadata for a catalog entry (name or alias)."""
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

        from xorq.ibis_yaml.enums import ExprKind  # noqa: PLC0415

        meta = entry.metadata
        type_label = {
            ExprKind.UnboundExpr: "Partial (unbound)",
            ExprKind.Source: "Source (bound)",
            ExprKind.Expr: "Expression",
            ExprKind.Composed: "Composed",
            ExprKind.ExprBuilder: "Expression Builder",
        }.get(entry.kind, str(entry.kind))
        backends = entry.backends
        cache_key = (
            meta.projected_cache_key.key
            if meta.projected_cache_key and meta.projected_cache_key.key
            else None
        )
        aliases_str = ", ".join(a.alias for a in entry.aliases) or None
        fields = [
            ("Name:", entry.name),
            ("Aliases:", aliases_str),
            ("Type:", type_label),
            ("Root tag:", meta.root_tag if meta.root_tag else None),  # noqa: FURB110
            ("Backends:", ", ".join(backends) if backends else None),
            ("Content local:", "yes" if entry.is_content_local else "no"),
            (
                "Composed from:",
                str(len(meta.composed_from)) if meta.composed_from else None,
            ),
            ("Cache key:", cache_key),
        ]
        for label, value in fields:
            if value is not None:
                click.echo(f"{label:<15} {value}")
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
    """Validate catalog consistency."""
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        catalog.assert_consistency()
        click.echo("OK")


@cli.command()
@json_option
@click.pass_context
def log(ctx, as_json):
    """Show catalog history as structured operations."""
    import attr

    from xorq.catalog.replay import Replayer

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
@click.option(
    "--remote-url",
    default=None,
    help="Git remote URL for the target catalog (sets origin and pushes).",
)
@click.option(
    "--preserve-commits/--no-preserve-commits",
    default=True,
    help="Preserve original commit authors and timestamps (default: yes).",
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
    remote_url,
    preserve_commits,
    force,
    dry_run,
    rebuild,
):
    """Replay catalog operations into a target catalog.

    With ``--rebuild``, each entry is re-added under current code:
    entries with no catalog references are re-added from their stored
    expression; entries containing catalog references (Composed, or
    ExprBuilder wrapping a composition) have the catalog subtree
    recomposed against their already-rebuilt dependencies in the
    target catalog. Outer builder wrappings pass through untouched.
    """
    from xorq.catalog.catalog import Catalog
    from xorq.catalog.replay import Replayer

    with click_context_catalog(ctx):
        source = ctx.obj.make_catalog(init=False)
        replayer = Replayer(from_catalog=source, rebuild=rebuild)
        if dry_run:
            replayer.print_plan()
            return
        annex = _resolve_annex_option(env_file, env_prefix, gcs)
        target = Catalog.from_repo_path(target_path, annex=annex)
        replayer.replay(target, preserve_commits=preserve_commits)
        click.echo(f"Replayed {len(replayer.ops)} operations into {target_path}")
        if remote_url:
            from xorq.catalog.constants import ANNEX_BRANCH, DEFAULT_REMOTE, MAIN_BRANCH

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

    Verifies that the provided credentials cannot write to the bucket
    before embedding them.
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


def _eval_entry(catalog_entry, code, instream=None):
    """Evaluate a single catalog entry to an expression."""
    from xorq.catalog.bind import _eval_code, _make_source_expr
    from xorq.ibis_yaml.enums import ExprKind

    match catalog_entry.kind:
        case ExprKind.UnboundExpr:
            import pyarrow as pa

            from xorq.common.utils.graph_utils import replace_unbound
            from xorq.common.utils.io_utils import maybe_open
            from xorq.expr.api import read_pyarrow_stream

            try:
                with maybe_open(instream, "rb") as stream:
                    input_expr = read_pyarrow_stream(stream)
            except (pa.ArrowInvalid, pa.ArrowException) as err:
                raise click.ClickException(
                    f"Entry {catalog_entry.name!r} is an unbound expression — "
                    f"pipe Arrow data into it or pass a source entry: "
                    f"'xorq catalog run SOURCE {catalog_entry.name} -o - -f csv'."
                ) from err
            expr = replace_unbound(catalog_entry.expr, input_expr.op())
        case ExprKind.Source | ExprKind.Expr | ExprKind.Composed | ExprKind.ExprBuilder:
            expr = _make_source_expr(catalog_entry)
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


def _compose_expr(catalog, entries, code, rename_map=None):
    """Build a composed expression from catalog entries and/or inline code."""

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

    # Validate that rename_map keys refer to actual entry names
    unknown = set(rename_map) - set(catalog_entries)
    if unknown:
        raise click.BadParameter(
            f"Unknown entry name(s) in --rename-params: {', '.join(sorted(unknown))}"
        )

    source_entry, *transform_entries = catalog_entries.values()

    source_name, *transform_names = entries

    if not rename_map:
        # Fast path: no renames, use ExprComposer directly
        from xorq.catalog.composer import ExprComposer  # noqa: PLC0415

        return ExprComposer(
            source=source_entry,
            transforms=transform_entries,
            code=code,
        ).expr

    # Slow path: apply renames per-entry, then compose manually
    if source_name in rename_map:
        source_expr = rename_params(source_entry.expr, rename_map[source_name])
    else:
        source_expr = source_entry.expr

    source_tagged, resolved_con = _resolve_source(source_expr, None, None)

    if transform_entries:
        from xorq.catalog.bind import _ensure_remote  # noqa: PLC0415
        from xorq.common.utils.graph_utils import replace_unbound  # noqa: PLC0415
        from xorq.expr.relations import RemoteTable, gen_name  # noqa: PLC0415

        def _bind_one_with_rename(current_expr, entry_and_name):
            entry, name = entry_and_name
            if name in rename_map:
                transform_expr = rename_params(entry.expr, rename_map[name])
            else:
                transform_expr = entry.expr
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
    from xorq.ibis_yaml.enums import DumpFiles  # noqa: PLC0415

    for w in bundle.wheel_paths:
        dst = build_path / w.name
        if not dst.exists():
            shutil.copy2(w, dst)
    shutil.copy2(bundle.requirements_path, build_path / DumpFiles.requirements)


def _get_catalog_entry(catalog, entry):
    """Look up an entry by name or alias, raising a helpful ClickException."""
    try:
        return catalog.get_catalog_entry(entry, maybe_alias=True)
    except (ValueError, AssertionError) as err:
        raise click.ClickException(
            f"Entry {entry!r} not found — run 'xorq catalog list' "
            f"or 'xorq catalog list-aliases' to see available entries and aliases."
        ) from err


def _resolve_entry_for_run(catalog, entry):
    ce = _get_catalog_entry(catalog, entry)
    if not ce.is_content_local:
        ce.fetch()
    return ce


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
    from xorq.ibis_yaml.enums import DumpFiles  # noqa: PLC0415
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
    from xorq.ibis_yaml.enums import DumpFiles  # noqa: PLC0415
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
                ce = _resolve_entry_for_run(catalog, entry)
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


# Params on the ``run`` command that rewrite the expression.  The fast path
# (PackagedRunner from archive) is only valid when none of these are set.
# Update this set when adding new expression-rewriting options to ``run``.
_EXPR_MODIFYING_PARAMS = frozenset({"code", "limit", "raw_params", "raw_rename_params"})


def _has_expr_modifications(ctx):
    """True when the user passed flags that modify the expression."""
    p = ctx.params
    if not p.get("fuse", True):
        return True
    if p.get("limit") is not None:
        return True
    return bool(p.get("code") or p.get("raw_params") or p.get("raw_rename_params"))


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


def _reinvoke_and_log(ctx, catalog, entries, span, rl):
    """Reinvoke the current Click command in the entries' pinned env and log metrics."""
    from xorq.common.utils.profile_utils import timed  # noqa: PLC0415

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
    with timed() as get_elapsed, _uv_reinvoke_xorq_cli(catalog, entries, *inner_cmd):
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


def _compose_via_reinvoke(ctx, catalog, entries):
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
                catalog, entries, *inner_cmd, capture_output=True
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
        "and exit without merging wheels or cataloging. Out-of-band so library "
        "prints to stdout can't be mistaken for the build_path. The outer "
        "reinvoke uses this to pick up the inner's build_path."
    ),
)
@click.pass_context
def compose(
    ctx,
    entries,
    code,
    alias,
    cache_dir,
    dry_run,
    raw_rename_params,
    use_this_venv,
    emit_build_path_to,
):
    """Assemble expressions from catalog entries, build, and persist to catalog.

    Always catalogs the result. Use 'run' to execute an entry for data output.
    """
    from xorq.common.utils.otel_utils import tracer  # noqa: PLC0415

    with tracer.start_as_current_span("catalog.compose") as span:
        span.set_attributes({"entries": entries, "has_code": code is not None})
        with click_context_catalog(ctx):
            catalog = ctx.obj.make_catalog(init=False)
            if not entries:
                raise click.UsageError("At least one entry is required.")

            if not use_this_venv:
                span.set_attribute("path", "uv-reinvoke")
                build_path = _compose_via_reinvoke(ctx, catalog, entries)
                if build_path is None:
                    return
            else:
                span.set_attribute("path", "in-process")
                rename_map = (
                    _parse_rename_params(raw_rename_params)
                    if raw_rename_params
                    else None
                )
                expr = _compose_expr(catalog, entries, code, rename_map=rename_map)

                if dry_run:
                    click.echo("Dry run — composition plan:")
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

                with _entry_run_bundle(catalog, entries) as bundle:
                    _stage_bundle_into_build(bundle, build_path)
            entry_name = build_path.name
            aliases = (alias,) if alias else ()
            alias_existed = alias and catalog.catalog_yaml.contains_alias(alias)
            entry_existed = catalog.contains(entry_name)
            catalog.add(build_path, aliases=aliases, exist_ok=True)
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


def _resolve_single_entry(catalog, entry, code, instream, rename_map, span):
    """Resolve a single catalog entry to an expression."""
    catalog_entry = _resolve_entry_for_run(catalog, entry)

    from xorq.ibis_yaml.enums import ExprKind  # noqa: PLC0415

    span.set_attribute("kind", str(catalog_entry.kind))
    expr = _eval_entry(catalog_entry, code, instream)

    if rename_map and entry in rename_map:
        from xorq.common.utils.graph_utils import rename_params  # noqa: PLC0415

        expr = rename_params(expr, rename_map[entry])

    if catalog_entry.kind is ExprKind.UnboundExpr:
        span.set_attribute("piped_stdin", True)

    return expr


def _resolve_and_execute(ctx, catalog, span, rl, span_prefix, *, expr_transform=None):
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

    rename_map = _parse_rename_params(raw_rename_params) if raw_rename_params else None

    span.set_attribute("path", "in-process")
    with timed() as get_elapsed:
        if len(entries) > 1:
            expr = _compose_expr(catalog, entries, code, rename_map=rename_map)
        else:
            (entry,) = entries
            expr = _resolve_single_entry(
                catalog, entry, code, instream, rename_map, span
            )
        rl.log_span_event(
            span,
            f"{span_prefix}.expr_loaded",
            {"elapsed_s": round(get_elapsed(), 3)},
        )

    expr = _apply_cli_params(expr, raw_params)

    if fuse:
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
@code_option
@output_options
@limit_option
@click.option(
    "-i",
    "--instream",
    type=click.File("rb"),
    default="-",
    help="Stream to read Arrow IPC record batches from (default: stdin).",
)
@fuse_option
@rename_params_option
@params_options
@click.option(
    "--use-this-venv/--no-use-this-venv",
    default=False,
    help=(
        "Execute in the current Python environment instead of spawning "
        "`uv tool run` on the entry's pinned env. Faster (no subprocess + "
        "uv venv lookup) but only correct when the calling venv already has "
        "every package the expression needs (xorq itself plus any UDFs "
        "from the entries' wheels). Default is the isolated `uv tool run` "
        "path."
    ),
)
@click.pass_context
def run(
    ctx,
    entries,
    code,
    output_path,
    output_format,
    limit,
    instream,
    fuse,
    raw_rename_params,
    raw_params,
    use_this_venv,
):
    """Compose and execute catalog entries.

    One entry runs it directly; multiple entries compose source + transforms:

    \b
        xorq catalog run src -o - -f csv
        xorq catalog run src trn -o - -f csv
        xorq catalog run src trn -c "source.filter(source.amount > 100)" -o - -f csv

    Piped Arrow input for a single unbound entry:

    \b
        xorq catalog run src -o - -f arrow | xorq catalog run trn -o - -f csv

    To persist composed results, use 'compose'.
    """
    from xorq.common.utils.logging_utils import RunLogger  # noqa: PLC0415
    from xorq.common.utils.otel_utils import tracer  # noqa: PLC0415
    from xorq.common.utils.profile_utils import timed  # noqa: PLC0415

    with tracer.start_as_current_span("catalog.run") as span:
        span.set_attributes({"entries": entries, "has_code": code is not None})
        with click_context_catalog(ctx):
            if not entries:
                raise click.UsageError("At least one entry is required.")

            expr_hash = "+".join(entries)
            run_params = (
                ("expr_hash", expr_hash),
                ("entries", ",".join(entries)),
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
                    from xorq.ibis_yaml.enums import ExprKind  # noqa: PLC0415
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
                    _reinvoke_and_log(ctx, catalog, entries, span, rl)
                    return

                _resolve_and_execute(ctx, catalog, span, rl, "catalog_run")


@cli.command("run-cached")
@click.argument("entries", nargs=-1, shell_complete=_complete_entry_or_alias_names)
@code_option
@output_options
@limit_option
@click.option(
    "-i",
    "--instream",
    type=click.File("rb"),
    default="-",
    help="Stream to read Arrow IPC record batches from (default: stdin).",
)
@fuse_option
@rename_params_option
@params_options
@cache_dir_option
@cache_strategy_options
@click.option(
    "--use-this-venv/--no-use-this-venv",
    default=False,
    help=(
        "Execute in the current Python environment instead of spawning "
        "`uv tool run` on the entry's pinned env. Faster (no subprocess + "
        "uv venv lookup) but only correct when the calling venv already has "
        "every package the expression needs (xorq itself plus any UDFs "
        "from the entries' wheels). Default is the isolated `uv tool run` "
        "path."
    ),
)
@click.pass_context
def run_cached(
    ctx,
    entries,
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
    """Compose and execute catalog entries with a ParquetCache wrapping the expression.

    Same semantics as `run`, but wraps the resulting expression with a cache
    (ParquetCache by default; snapshot/TTL variants via --cache-type / --ttl).
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
            if not entries:
                raise click.UsageError("At least one entry is required.")

            resolved_cache_dir = _get_cache_dir(cache_dir)

            expr_hash = "+".join(entries)
            run_params = (
                ("expr_hash", expr_hash),
                ("entries", ",".join(entries)),
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
                    _reinvoke_and_log(ctx, catalog, entries, span, rl)
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
                )
