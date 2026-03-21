from contextlib import contextmanager
from functools import cache, partial
from pathlib import Path
from types import SimpleNamespace

import click


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
@click.option(
    "--filter-kind",
    default=None,
    help="Show only entries of the given kind (source, expr, unbound_expr, composed).",
)
@click.pass_context
def list_entries(ctx, kind, filter_kind):
    """List all entries."""
    from xorq.ibis_yaml.enums import ExprKind

    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)

        if not (entries := catalog.catalog_entries):
            click.echo("No entries.")
            return

        if filter_kind:
            try:
                target_kind = ExprKind(filter_kind)
            except ValueError:
                raise click.UsageError(
                    f"Unknown kind: {filter_kind!r}. "
                    f"Valid kinds: {', '.join(k.value for k in ExprKind)}"
                ) from None
            entries = tuple(e for e in entries if e.kind == target_kind)
            if not entries:
                click.echo(f"No entries with kind={filter_kind}.")
                return

        for entry in entries:
            if kind:
                parts = [entry.name, str(entry.kind)]
                if entry.kind == ExprKind.Composed:
                    source_names = ", ".join(
                        s.get("alias") or s.get("entry_name", "?")
                        for s in entry.sources
                    )
                    parts.append(source_names or "-")
                else:
                    parts.append("-")
                click.echo("\t".join(parts))
            else:
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
            click.echo(json_mod.dumps(entry.metadata, indent=2))
            return

        type_label = (
            "Partial (unbound)"
            if entry.kind == ExprKind.UnboundExpr
            else "Source (bound)"
        )
        click.echo(f"Type: {type_label}")

        for label, key in (("Schema In", "schema_in"), ("Schema Out", "schema_out")):
            if (sch := entry.metadata.get(key)) is not None:
                click.echo()
                click.echo(f"{label}:")
                for col, dtype in sch.items():
                    click.echo(f"  {col:<24} {dtype}")


@cli.command()
@click.pass_context
def check(ctx):
    """Validate catalog consistency."""
    with click_context_catalog(ctx):
        catalog = ctx.obj.make_catalog(init=False)
        catalog.assert_consistency()
        click.echo("OK")


def _complete_entry_or_alias(ctx, param, incomplete):
    from click.shell_completion import CompletionItem

    try:
        catalog = _make_catalog_for_completion(ctx)
        names = set(catalog.list()) | set(catalog.list_aliases())
        return [CompletionItem(n) for n in sorted(names) if n.startswith(incomplete)]
    except Exception:
        return []


def _resolve_entries(catalog, entries):
    """Resolve entry names/aliases to CatalogEntry objects."""
    return tuple(catalog.get_catalog_entry(name, maybe_alias=True) for name in entries)


def _eval_code(code, source):
    """Evaluate an inline Ibis expression with a restricted namespace.

    Only xorq, vendored ibis, and the bound ``source`` expression are
    available — arbitrary builtins (open, exec, __import__, …) are removed.
    """
    import xorq.api as xo  # noqa: PLC0415
    from xorq.vendor import ibis  # noqa: PLC0415

    restricted_globals = {"__builtins__": {}, "xo": xo, "ibis": ibis, "source": source}
    return eval(code, restricted_globals)  # noqa: S307


def _compose_expr(catalog, entries, code):
    """Build a composed expression from catalog entries and/or inline code."""

    from xorq.catalog.bind import bind  # noqa: PLC0415

    match (entries, code):
        case ((), str()):
            raise click.UsageError("--code requires at least one entry as source.")
        case (_, str()):
            transforms = _resolve_entries(catalog, entries[1:])
            source = (
                bind(catalog.source(entries[0]), *transforms)
                if transforms
                else catalog.source(entries[0])
            )
            return _eval_code(code, source)
        case _ if len(entries) < 2:
            raise click.UsageError(
                "At least two entries required: SOURCE TRANSFORM [TRANSFORM ...]\n"
                "Or one entry with --code."
            )
        case _:
            resolved = _resolve_entries(catalog, entries)
            return bind(resolved[0], *resolved[1:])


@cli.command("run")
@click.argument("entries", nargs=-1, shell_complete=_complete_entry_or_alias)
@click.option(
    "-c",
    "--code",
    default=None,
    help="Inline Ibis code expression applied to `source`.",
)
@click.option(
    "-a", "--alias", default=None, help="Catalog the result under this alias."
)
@click.option(
    "--catalog/--no-catalog",
    "do_catalog",
    default=True,
    help="Catalog the result (default: yes).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Write output to file.",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    default="csv",
    type=click.Choice(["csv", "parquet", "json"]),
    help="Output format.",
)
@click.option("--limit", type=int, default=None, help="Limit rows.")
@click.pass_context
def run(ctx, entries, code, alias, do_catalog, output_path, output_format, limit):
    """Run catalog entries through each other and print results.

    By default the result is added to the catalog (with --alias, or hash-only).
    Use --no-catalog to skip.
    """
    import sys

    from xorq.common.utils.otel_utils import tracer

    with tracer.start_as_current_span("catalog.run") as span:
        span.set_attributes({"entries": entries, "has_code": code is not None})
        with click_context_catalog(ctx):
            catalog = ctx.obj.make_catalog(init=False)
            expr = _compose_expr(catalog, entries, code)

            if do_catalog:
                from xorq.ibis_yaml.compiler import build_expr

                build_path = build_expr(expr)
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

            if limit is not None:
                expr = expr.limit(limit)
            result = expr.execute()

            match (output_path, output_format):
                case (None, "csv"):
                    result.to_csv(sys.stdout, index=False)
                case (None, "json"):
                    click.echo(result.to_json(orient="records", lines=True))
                case (None, "parquet"):
                    raise click.UsageError(
                        "Parquet cannot be written to stdout — use -o to specify an output file."
                    )
                case (str(), "parquet"):
                    result.to_parquet(output_path)
                    click.echo(f"Written to {output_path}")
                case (str(), "csv"):
                    result.to_csv(output_path, index=False)
                    click.echo(f"Written to {output_path}")
                case (str(), "json"):
                    result.to_json(output_path, orient="records", lines=True)
                    click.echo(f"Written to {output_path}")


@cli.command("build")
@click.argument("entries", nargs=-1, shell_complete=_complete_entry_or_alias)
@click.option(
    "-c",
    "--code",
    default=None,
    help="Inline Ibis code expression applied to `source`.",
)
@click.option(
    "--builds-dir", default="builds", help="Directory for generated artifacts."
)
@click.option("--debug", is_flag=True, help="Output debug artifacts.")
@click.pass_context
def build(ctx, entries, code, builds_dir, debug):
    """Build a composed artifact from catalog entries.

    First entry is the source, remaining are transforms.
    Use --code for an inline transform applied to `source`.
    """
    from xorq.common.utils.otel_utils import tracer
    from xorq.ibis_yaml.compiler import build_expr

    with tracer.start_as_current_span("catalog.build") as span:
        span.set_attributes({"entries": entries, "has_code": code is not None})
        with click_context_catalog(ctx):
            catalog = ctx.obj.make_catalog(init=False)
            expr = _compose_expr(catalog, entries, code)
            build_path = build_expr(expr, builds_dir=builds_dir, debug=debug)
            span.set_attribute("build_path", str(build_path))
            click.echo(build_path)
