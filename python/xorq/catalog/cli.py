from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import click

from xorq.catalog.catalog import Catalog


def click_handler(e):
    raise click.ClickException(str(e)) from e


@contextmanager
def click_context(*typs):
    try:
        yield
    except click.ClickException:
        raise
    except typs as e:
        click_handler(e)


click_context_default = partial(click_context, AssertionError, Exception)


def _complete_entry_names(ctx, param, incomplete):
    from click.shell_completion import CompletionItem

    try:
        # During completion, Click calls make_context but never invokes group
        # callbacks, so ctx.obj is None.  Read the parsed params directly from
        # the catalog group's context (our immediate parent).
        catalog_ctx = ctx.parent
        catalog = Catalog.from_kwargs(
            name=catalog_ctx.params.get("name"),
            path=catalog_ctx.params.get("path"),
            url=catalog_ctx.params.get("url"),
            root_repo=catalog_ctx.params.get("root_repo"),
            init=False,
        )
        return [CompletionItem(n) for n in catalog.list() if n.startswith(incomplete)]
    except Exception:
        return []


@click.group()
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


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize a new catalog."""
    with click_context_default():
        catalog = ctx.obj.make_catalog(init=True)
    click.echo(f"Initialized catalog at {catalog.repo_path}")


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--sync/--no-sync", default=True)
@click.pass_context
def add(ctx, paths, sync):
    """Add entries from tgz files or build directories."""
    with click_context_default():
        catalog = ctx.obj.make_catalog(init=False)
        with catalog.maybe_synchronizing(sync):
            for path in map(Path, paths):
                entry = catalog.add(path, sync=False)
                click.echo(f"Added {entry.name}")


@cli.command()
@click.argument("names", nargs=-1, required=True, shell_complete=_complete_entry_names)
@click.option("--sync/--no-sync", default=True)
@click.pass_context
def remove(ctx, names, sync):
    """Remove entries by name."""
    with click_context_default():
        catalog = ctx.obj.make_catalog(init=False)
        with catalog.maybe_synchronizing(sync):
            for name in names:
                entry = catalog.remove(name, sync=False)
                click.echo(f"Removed {entry.name}")


@cli.command("list")
@click.pass_context
def list_entries(ctx):
    """List all entries."""
    with click_context_default():
        catalog = ctx.obj.make_catalog(init=False)
        names = catalog.list() or ("No entries.",)
        for name in names:
            click.echo(name)


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
    """Export an entry's tgz to a directory."""
    with click_context_default():
        catalog = ctx.obj.make_catalog(init=False)
        result = catalog.get_tgz(name, dir_path=output)
        click.echo(f"Exported to {result}")


@cli.command()
@click.pass_context
def push(ctx):
    """Push catalog to remote(s)."""
    with click_context_default():
        catalog = ctx.obj.make_catalog(init=False)
        catalog.push()
        click.echo("Pushed.")


@cli.command()
@click.pass_context
def pull(ctx):
    """Pull catalog from remote(s)."""
    with click_context_default():
        catalog = ctx.obj.make_catalog(init=False)
        catalog.pull()
        click.echo("Pulled.")


@cli.command()
@click.pass_context
def sync(ctx):
    """Pull then push."""
    with click_context_default():
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
@click.pass_context
def check(ctx):
    """Validate catalog consistency."""
    with click_context_default():
        catalog = ctx.obj.make_catalog(init=False)
        catalog.assert_consistency()
        click.echo("OK")
