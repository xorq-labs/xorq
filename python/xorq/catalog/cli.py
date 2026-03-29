import os
from contextlib import contextmanager
from functools import cache, partial
from pathlib import Path
from types import SimpleNamespace

import click

from xorq.cli import OutputFormats


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


@contextmanager
def click_context(*typs):
    try:
        yield
    except click.ClickException:
        raise
    except typs as e:
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
        click_handler(e)


click_context_default = partial(click_context, AssertionError, Exception)


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


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize a new catalog."""
    with click_context_catalog(ctx):
        try:
            catalog = ctx.obj.make_catalog(init=True)
        except AssertionError as err:
            # init_repo_path asserts the path does not already exist
            probe = ctx.obj.make_catalog(init=False)
            raise click.ClickException(
                f"Catalog already exists at {probe.repo_path}"
            ) from err
    click.echo(f"Initialized catalog at {catalog.repo_path}")


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--sync/--no-sync", default=True)
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
@click.option("--sync/--no-sync", default=True)
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
@click.option("--sync/--no-sync", default=True)
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
@click.option("--sync/--no-sync", default=True)
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
    """Push catalog to remote(s)."""
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        catalog.push()
        click.echo("Pushed.")


@cli.command()
@click.pass_context
def pull(ctx):
    """Pull catalog from remote(s)."""
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
def clone(url, dest_name, dest_path):
    """Clone a catalog from a remote URL."""
    from xorq.catalog.catalog import Catalog

    with click_context_default():
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
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
@click.pass_context
def schema(ctx, name, as_json):
    """Show schema of a catalog entry (name or alias)."""
    import json as json_mod

    from xorq.ibis_yaml.enums import ExprKind  # noqa: PLC0415

    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        try:
            entry = catalog.get_catalog_entry(name, maybe_alias=True)
        except AssertionError as err:
            raise click.ClickException(
                f"Entry {name!r} not found — run 'xorq catalog list' or 'xorq catalog list-aliases' to see available entries and aliases."
            ) from err

        if as_json:
            click.echo(json_mod.dumps(entry.metadata.to_dict(), indent=2))
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
@click.pass_context
def check(ctx):
    """Validate catalog consistency."""
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        catalog.assert_consistency()
        click.echo("OK")


def _resolve_entries(catalog, entries):
    """Resolve entry names/aliases to CatalogEntry objects."""
    return tuple(catalog.get_catalog_entry(name, maybe_alias=True) for name in entries)


def _compose_expr(catalog, entries, code):
    """Build a composed expression from catalog entries and/or inline code."""

    from xorq.catalog.composer import ExprComposer  # noqa: PLC0415

    if not entries:
        raise click.UsageError("At least one entry is required.")

    resolved = _resolve_entries(catalog, entries)
    return ExprComposer(
        source=resolved[0],
        transforms=resolved[1:],
        code=code,
    ).expr


@cli.command("compose")
@click.argument("entries", nargs=-1, shell_complete=_complete_entry_or_alias_names)
@click.option(
    "-c",
    "--code",
    default=None,
    help="Inline Ibis code expression applied to `source`.",
)
@click.option(
    "-a",
    "--alias",
    default=None,
    help="Also register this alias for the cataloged entry.",
)
@click.option("--cache-dir", default=None, help="Directory for parquet cache files.")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show composition plan without building.",
)
@click.pass_context
def compose(ctx, entries, code, alias, cache_dir, dry_run):
    """Assemble expressions from catalog entries, build, and persist to catalog.

    Always catalogs the result. Use 'run' to execute an entry for data output.
    """
    from xorq.common.utils.otel_utils import tracer

    with tracer.start_as_current_span("catalog.compose") as span:
        span.set_attributes({"entries": entries, "has_code": code is not None})
        with click_context_catalog(ctx):
            catalog = ctx.obj.make_catalog(init=False)
            expr = _compose_expr(catalog, entries, code)

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

            build_kwargs = {}
            if cache_dir is not None:
                build_kwargs["cache_dir"] = Path(cache_dir)

            from xorq.ibis_yaml.compiler import build_expr

            build_path = build_expr(expr, **build_kwargs)
            entry_name = build_path.name
            aliases = (alias,) if alias else ()
            if catalog.contains(entry_name):
                if alias:
                    catalog.add_alias(entry_name, alias)
            else:
                catalog.add(build_path, aliases=aliases)
            label = alias or entry_name
            click.echo(f"Cataloged as {label!r}", err=True)
            span.set_attribute("cataloged", label)


@cli.command("run")
@click.argument("entries", nargs=-1, shell_complete=_complete_entry_or_alias_names)
@click.option(
    "-c",
    "--code",
    default=None,
    help="Inline Ibis code expression applied to `source`.",
)
@click.option(
    "-o",
    "--output-path",
    default=None,
    help=f"Path to write output (default: {os.devnull})",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice([f.value for f in OutputFormats]),
    default=OutputFormats.default,
    help="Output format (default: parquet).",
)
@click.option("--limit", type=int, default=None, help="Limit number of rows to output.")
@click.option(
    "-i",
    "--instream",
    type=click.File("rb"),
    default="-",
    help="Stream to read Arrow IPC record batches from (default: stdin).",
)
@click.pass_context
def run(ctx, entries, code, output_path, output_format, limit, instream):
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
    from xorq.catalog.bind import _eval_code, _make_source_expr
    from xorq.cli import arbitrate_output_format
    from xorq.common.utils.otel_utils import tracer
    from xorq.ibis_yaml.enums import ExprKind

    with tracer.start_as_current_span("catalog.run") as span:
        span.set_attributes({"entries": entries, "has_code": code is not None})
        with click_context_catalog(ctx):
            if not entries:
                raise click.UsageError("At least one entry is required.")

            catalog = ctx.obj.make_catalog(init=False)

            if len(entries) > 1:
                expr = _compose_expr(catalog, entries, code)
            else:
                entry = entries[0]
                try:
                    catalog_entry = catalog.get_catalog_entry(entry, maybe_alias=True)
                except AssertionError as err:
                    raise click.ClickException(
                        f"Entry {entry!r} not found — run 'xorq catalog list' "
                        f"or 'xorq catalog list-aliases' to see available entries."
                    ) from err

                span.set_attribute("kind", str(catalog_entry.kind))

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
                                f"Entry {entry!r} is an unbound expression — "
                                f"pipe Arrow data into it or pass a source entry: "
                                f"'xorq catalog run SOURCE {entry} -o - -f csv'."
                            ) from err
                        span.set_attribute("piped_stdin", True)
                        expr = replace_unbound(catalog_entry.expr, input_expr.op())
                    case ExprKind.Source | ExprKind.Expr | ExprKind.Composed:
                        expr = _make_source_expr(catalog_entry)
                    case _:
                        raise click.ClickException(
                            f"Unsupported entry kind {catalog_entry.kind!r} for 'run'."
                        )

                if code is not None:
                    expr = _eval_code(code, expr)

            if limit is not None:
                expr = expr.limit(limit)
            arbitrate_output_format(expr, output_path, output_format)
