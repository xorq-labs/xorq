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
@click.option(
    "--filter-kind",
    type=click.Choice(["expression", "builder"]),
    default=None,
    help="Filter by entry category (expression or builder).",
)
@click.pass_context
def list_entries(ctx, kind, filter_kind):
    """List all entries."""
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)

        if not (entries := catalog.catalog_entries):
            click.echo("No entries.")
            return

        for entry in entries:
            category = "builder" if entry.is_builder else "expression"
            if filter_kind and category != filter_kind:
                continue
            match (kind, entry.is_builder):
                case (True, True):
                    builder_type = (entry.builder_meta or {}).get("type", "unknown")
                    click.echo(f"{entry.name}\tbuilder\t{builder_type}")
                case (True, False):
                    click.echo(f"{entry.name}\texpression\t{entry.kind}")
                case _:
                    click.echo(entry.name)


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
        except (ValueError, AssertionError) as err:
            raise click.ClickException(
                f"Entry {name!r} not found — run 'xorq catalog list' or 'xorq catalog list-aliases' to see available entries and aliases."
            ) from err

        # Builder entries: show builder_meta.json content
        if entry.is_builder:
            bmeta = entry.builder_meta
            if as_json:
                click.echo(json_mod.dumps(bmeta, indent=2))
                return

            builder_type = bmeta.get("type", "unknown")
            click.echo(f"Kind: builder ({builder_type})")
            if desc := bmeta.get("description"):
                click.echo(f"Description: {desc}")

            if dims := bmeta.get("available_dimensions"):
                click.echo()
                click.echo("Available Dimensions:")
                click.echo(f"  {', '.join(dims)}")
            if measures := bmeta.get("available_measures"):
                click.echo()
                click.echo("Available Measures:")
                click.echo(f"  {', '.join(measures)}")
            if steps := bmeta.get("steps"):
                click.echo()
                click.echo("Pipeline Steps:")
                for step in steps:
                    click.echo(
                        f"  {step.get('name', '?'):<20} {step.get('estimator', '?')}"
                    )
            return

        # Expression entries
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
        case ExprKind.Source | ExprKind.Expr | ExprKind.Composed:
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
        from functools import reduce  # noqa: PLC0415

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
@click.option(
    "--rename-params",
    "raw_rename_params",
    multiple=True,
    help="Rename a parameter: entry,old_name,new_name (repeatable).",
)
@click.pass_context
def compose(ctx, entries, code, alias, cache_dir, dry_run, raw_rename_params):
    """Assemble expressions from catalog entries, build, and persist to catalog.

    Always catalogs the result. Use 'run' to execute an entry for data output.
    """
    from xorq.common.utils.otel_utils import tracer

    with tracer.start_as_current_span("catalog.compose") as span:
        span.set_attributes({"entries": entries, "has_code": code is not None})
        with click_context_catalog(ctx):
            catalog = ctx.obj.make_catalog(init=False)
            rename_map = (
                _parse_rename_params(raw_rename_params) if raw_rename_params else None
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

            from xorq.ibis_yaml.compiler import build_expr

            build_kwargs = {} if cache_dir is None else {"cache_dir": Path(cache_dir)}
            build_path = build_expr(expr, **build_kwargs)
            entry_name = build_path.name
            aliases = (alias,) if alias else ()
            alias_existed = alias and catalog.catalog_yaml.contains_alias(alias)
            catalog_entry = catalog.add(build_path, aliases=aliases, exist_ok=True)
            label = alias or entry_name
            if catalog_entry is None:
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
@click.option(
    "--fuse/--no-fuse",
    default=True,
    help="Enable/disable catalog source fusion (default: enabled).",
)
@click.option(
    "--rename-params",
    "raw_rename_params",
    multiple=True,
    help="Rename a parameter: entry,old_name,new_name (repeatable).",
)
@click.pass_context
def run(ctx, entries, code, output_path, output_format, limit, instream, fuse, raw_rename_params):
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
    from xorq.cli import arbitrate_output_format
    from xorq.common.utils.otel_utils import tracer

    with tracer.start_as_current_span("catalog.run") as span:
        span.set_attributes({"entries": entries, "has_code": code is not None})
        with click_context_catalog(ctx):
            if not entries:
                raise click.UsageError("At least one entry is required.")

            catalog = ctx.obj.make_catalog(init=False)
            rename_map = (
                _parse_rename_params(raw_rename_params) if raw_rename_params else None
            )

            if len(entries) > 1:
                expr = _compose_expr(catalog, entries, code, rename_map=rename_map)
            else:
                (entry,) = entries
                try:
                    catalog_entry = catalog.get_catalog_entry(entry, maybe_alias=True)
                except (ValueError, AssertionError) as err:
                    raise click.ClickException(
                        f"Entry {entry!r} not found — run 'xorq catalog list' "
                        f"or 'xorq catalog list-aliases' to see available entries."
                    ) from err
                from xorq.ibis_yaml.enums import ExprKind

                if catalog_entry.is_builder:
                    raise click.ClickException(
                        f"Entry {entry!r} is a builder, not an expression — "
                        f"use 'xorq catalog schema {entry}' to inspect it, "
                        f"or use the Python API to build an expression from it."
                    )

                span.set_attribute("kind", str(catalog_entry.kind))
                expr = _eval_entry(catalog_entry, code, instream)
                if rename_map and entry in rename_map:
                    from xorq.common.utils.graph_utils import (
                        rename_params,  # noqa: PLC0415
                    )

                    expr = rename_params(expr, rename_map[entry])
                if catalog_entry.kind is ExprKind.UnboundExpr:
                    span.set_attribute("piped_stdin", True)

            if fuse:
                expr = expr.ls.fused

            if limit is not None:
                expr = expr.limit(limit)
            arbitrate_output_format(expr, output_path, output_format)
