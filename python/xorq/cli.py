import os
import pdb as pdb_module
import sys
import traceback
from functools import partial, wraps
from pathlib import Path

import click

from xorq.init_templates import InitTemplates


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class OutputFormats(StrEnum):
    csv = "csv"
    json = "json"
    parquet = "parquet"
    arrow = "arrow"


OutputFormats.default = OutputFormats.parquet


def _lazy_span(name):
    """Decorator that wraps a function in an OpenTelemetry span, importing lazily."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            from xorq.common.utils.otel_utils import tracer

            with tracer.start_as_current_span(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def _get_cache_dir(cache_dir):
    if cache_dir is None:
        from xorq.common.utils.caching_utils import get_xorq_cache_dir

        cache_dir = get_xorq_cache_dir()
    return cache_dir


def ensure_build_dir(expr_path):
    build_dir = Path(expr_path)
    if not build_dir.exists() or not build_dir.is_dir():
        print(f"Build target not found: {expr_path}")
        sys.exit(2)
    return build_dir


@_lazy_span("cli.uv_build_command")
def uv_build_command(
    script_path,
    project_path=None,
    sys_argv=(),
):
    from xorq.ibis_yaml.packager import SdistBuilder

    sdist_builder = SdistBuilder.from_script_path(
        script_path, project_path=project_path, args=sys_argv
    )
    # should we execv here instead?
    # ensure we do copy_sdist
    sdist_builder.build_path
    popened = sdist_builder._uv_tool_run_xorq_build
    print(popened.stderr, file=sys.stderr, end="")
    print(popened.stdout, file=sys.stdout, end="")
    return popened


@_lazy_span("cli.uv_run_command")
def uv_run_command(
    expr_path,
    sys_argv=(),
):
    from xorq.ibis_yaml.packager import SdistRunner

    sdist_runner = SdistRunner(expr_path, args=sys_argv)
    popened = sdist_runner._uv_tool_run_xorq_run
    return popened


@_lazy_span("cli.build_command")
def build_command(
    script_path,
    expr_name,
    builds_dir="builds",
    cache_dir=None,
    debug: bool = False,
):
    """
    Generate artifacts from an expression in a given Python script

    Parameters
    ----------
    script_path : Path to the Python script
    expr_name : The name of the expression to build
    builds_dir : Directory where artifacts will be generated
    cache_dir : Directory where the parquet cache files will be generated

    Returns
    -------

    """
    from opentelemetry import trace

    import xorq.common.utils.pickle_utils  # noqa: F401
    from xorq.common.utils.import_utils import import_from_path
    from xorq.ibis_yaml.compiler import build_expr
    from xorq.vendor.ibis import Expr

    cache_dir = _get_cache_dir(cache_dir)

    span = trace.get_current_span()
    span.add_event(
        "build.params",
        {
            "script_path": str(script_path),
            "expr_name": expr_name,
        },
    )

    if not os.path.exists(script_path):
        raise ValueError(f"Error: Script not found at {script_path}")
    print(f"Building {expr_name} from {script_path}", file=sys.stderr)
    vars_module = import_from_path(script_path, module_name="__main__")
    match expr := getattr(vars_module, expr_name, None):
        case Expr():
            pass
        case None:
            raise ValueError(f"Expression {expr_name} not found")
        case _:
            raise ValueError(
                f"The object {expr_name} must be an instance of {Expr.__module__}.{Expr.__name__}"
            )

    build_path = build_expr(
        expr, builds_dir=builds_dir, cache_dir=Path(cache_dir), debug=debug
    )
    expr_hash = build_path.name
    span.add_event("build.outputs", {"expr_hash": expr_hash})
    print(
        f"Written '{expr_name}' to {build_path}",
        file=sys.stderr,
    )
    print(build_path)


@_lazy_span("cli.run_command")
def run_command(
    expr_path,
    output_path=None,
    output_format=OutputFormats.default,
    cache_dir=None,
    limit=None,
):
    """
    Execute an artifact

    Parameters
    ----------
    expr_path : str
        Path to the expr in the builds dir
    output_path : str
        Path to write output. Defaults to os.devnull
    output_format : OutputFormats | str, optional
        Output format, either "csv", "json", "arrow", or "parquet". Defaults to "parquet"
    cache_dir : Path, optional
        Directory where the parquet cache files will be generated
    limit : int, optional
        Limit number of rows to output. Defaults to None (no limit).

    Returns
    -------

    """
    from opentelemetry import trace

    from xorq.ibis_yaml.compiler import load_expr

    cache_dir = _get_cache_dir(cache_dir)

    span = trace.get_current_span()
    span.add_event(
        "run.params",
        {
            "expr_path": str(expr_path),
            "output_path": str(output_path),
            "output_format": output_format,
        },
    )

    expr = load_expr(expr_path, cache_dir=cache_dir)
    if limit is not None:
        expr = expr.limit(limit)
    arbitrate_output_format(expr, output_path, output_format)


def arbitrate_output_format(expr, output_path, output_format):
    match (output_path, output_format):
        case (None, _):
            output_path = os.devnull
        case ("-", OutputFormats.json):
            # FIXME: deal with windows
            output_path = sys.stdout
        case ("-", _):
            output_path = sys.stdout.buffer
        case _:
            pass
    match output_format:
        case OutputFormats.csv:
            expr.to_csv(output_path)
        case OutputFormats.json:
            expr.to_json(output_path)
        case OutputFormats.arrow:
            from xorq.expr.api import to_pyarrow_stream

            to_pyarrow_stream(expr, output_path)
        case OutputFormats.parquet:
            expr.to_parquet(output_path)
        case _:
            raise ValueError(f"Unknown output_format: {output_format}")


@_lazy_span("cli.run_unbound_command")
def run_unbound_command(
    expr_path,
    to_unbind_hash=None,
    to_unbind_tag=None,
    output_path=None,
    output_format=OutputFormats.default,
    cache_dir=None,
    limit=None,
    typ=None,
    instream=sys.stdin.buffer,
):
    """
    Execute an unbound expression by reading Arrow IPC from stdin and binding it

    Parameters
    ----------
    expr_path : str
        Path to the expr in the builds dir
    to_unbind_hash : str, optional
        Hash of the node to unbind
    to_unbind_tag : str, optional
        Tag of the node to unbind
    output_path : str, optional
        Path to write output. Defaults to stdout for arrow, os.devnull otherwise
    output_format : str, optional
        Output format, either "csv", "json", "parquet", or "arrow". Defaults to "parquet"
    cache_dir : Path, optional
        Directory where the parquet cache files will be generated
    limit : int, optional
        Limit number of rows to output. Defaults to None (no limit).
    typ : str, optional
        Type of the node to unbind

    Returns
    -------

    """
    from opentelemetry import trace

    from xorq.common.utils.io_utils import maybe_open
    from xorq.common.utils.node_utils import expr_to_unbound
    from xorq.expr.api import read_pyarrow_stream
    from xorq.flight.exchanger import replace_one_unbound
    from xorq.ibis_yaml.compiler import load_expr

    cache_dir = _get_cache_dir(cache_dir)

    span = trace.get_current_span()
    span.add_event(
        "run_unbound.params",
        {
            "expr_path": str(expr_path),
            "to_unbind_hash": str(to_unbind_hash),
            "to_unbind_tag": str(to_unbind_tag),
            "output_format": str(output_format),
        },
    )

    # Resolve build identifier
    expr_path = ensure_build_dir(expr_path)
    # Log to stderr to avoid polluting Arrow streams
    print(f"[run-unbound] Loading expression from {expr_path}", file=sys.stderr)

    # Load the expression and make it unbound
    expr = load_expr(expr_path, cache_dir=cache_dir)
    unbound_expr = expr_to_unbound(
        expr, hash=to_unbind_hash, tag=to_unbind_tag, typs=typ
    ).to_expr()

    # Read Arrow IPC from instream
    print("[run-unbound] Reading Arrow IPC from instream...", file=sys.stderr)
    # Create a connection and register the input table
    with maybe_open(instream, "rb") as stream:
        input_expr = read_pyarrow_stream(stream)
        # Replace the unbound node with the input table
        bound_expr = replace_one_unbound(unbound_expr, input_expr)

        if limit is not None:
            bound_expr = bound_expr.limit(limit)
        arbitrate_output_format(bound_expr, output_path, output_format)


@_lazy_span("cli.unbind_and_serve_command")
def unbind_and_serve_command(
    expr_path,
    to_unbind_hash=None,
    to_unbind_tag=None,
    host=None,
    port=None,
    prometheus_port=None,
    cache_dir=None,
    typ=None,
):
    import xorq
    import xorq.expr.relations
    from xorq.caching.strategy import SnapshotStrategy
    from xorq.common.utils.logging_utils import get_print_logger
    from xorq.common.utils.node_utils import expr_to_unbound
    from xorq.ibis_yaml.compiler import load_expr

    logger = get_print_logger()
    cache_dir = _get_cache_dir(cache_dir)

    # Resolve build path to an actual build directory
    expr_path = ensure_build_dir(expr_path)
    logger.info(f"Loading expression from {expr_path}")
    try:
        # initialize console and optional Prometheus metrics
        from xorq.flight.metrics import setup_console_metrics

        setup_console_metrics(prometheus_port=prometheus_port)
    except ImportError:
        logger.warning(
            "Metrics support requires 'opentelemetry-sdk' and console exporter"
        )

    expr = load_expr(expr_path, cache_dir=cache_dir)
    unbound_expr = expr_to_unbound(
        expr,
        hash=to_unbind_hash,
        tag=to_unbind_tag,
        typs=typ,
        strategy=SnapshotStrategy(),
    )
    flight_url = xorq.flight.FlightUrl(host=host, port=port)
    make_server = partial(
        xorq.flight.FlightServer,
        flight_url=flight_url,
    )
    logger.info(f"Serving expression from '{expr_path}' on {flight_url.to_location()}")
    server, _ = xorq.expr.relations.flight_serve_unbound(
        unbound_expr, make_server=make_server
    )
    server.wait()


@_lazy_span("cli.serve_command")
def serve_command(
    expr_path,
    host=None,
    port=None,
    duckdb_path=None,
    prometheus_port=None,
    cache_dir=None,
):
    """
    Serve a built expression via Flight Server

    Parameters
    ----------
    expr_path : str
        Path to the expression directory (output of xorq build)
    host : str
        Host to bind Flight Server
    port : int or None
        Port to bind Flight Server (None for random)
    duckdb_path : str or None
        Path to duckdb cache DB file
    prometheus_port : int or None
        Port to connect to the prometheus server
    cache_dir : str or None
        Path to the dir to store the parquet cache files
    """
    from opentelemetry import trace

    from xorq.common.utils.logging_utils import get_print_logger
    from xorq.flight import FlightServer
    from xorq.ibis_yaml.compiler import load_expr
    from xorq.loader import load_backend

    logger = get_print_logger()
    cache_dir = _get_cache_dir(cache_dir)

    # Resolve build path to an actual build directory
    expr_path = ensure_build_dir(expr_path)
    span = trace.get_current_span()
    params = {
        "build_path": expr_path,
        "host": host,
        "port": port,
    }
    if duckdb_path is not None:
        params["duckdb_path"] = duckdb_path
    span.add_event("serve.params", params)

    expr_path = Path(expr_path)
    logger.info(f"Loading expression from {expr_path}")
    expr = load_expr(expr_path, cache_dir=cache_dir)

    db_path = Path(duckdb_path or "xorq_serve.db")  # FIXME what should be the default?

    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using duckdb at {db_path}")

    try:
        from xorq.flight.metrics import setup_console_metrics

        setup_console_metrics(prometheus_port=prometheus_port)
    except ImportError:
        logger.warning(
            "Metrics support requires 'opentelemetry-sdk' and console exporter"
        )

    server = FlightServer.from_udxf(
        expr,
        make_connection=partial(load_backend("duckdb").connect, str(db_path)),
        port=port,
        host=host,
    )
    location = server.flight_url.to_location()
    logger.info(f"Serving expression '{expr_path.stem}' on {location}")
    server.serve(block=True)


@_lazy_span("cli.init_command")
def init_command(
    path="./xorq-template",
    template=InitTemplates.default,
    branch=None,
):
    from xorq.common.utils.download_utils import download_unpacked_xorq_template

    path = download_unpacked_xorq_template(path, template, branch=branch)
    print(f"initialized xorq template `{template}` to {path}")
    return path


class PdbGroup(click.Group):
    def invoke(self, ctx):
        try:
            if ctx.params.get("pdb_runcall"):
                return pdb_module.runcall(super().invoke, ctx)
            return super().invoke(ctx)
        except (click.ClickException, click.exceptions.Exit, SystemExit):
            raise
        except Exception as e:
            if ctx.params.get("use_pdb"):
                traceback.print_exception(e)
                pdb_module.post_mortem(e.__traceback__)
            else:
                traceback.print_exc()
            sys.exit(1)


@click.group(cls=PdbGroup)
@click.option("--pdb", "use_pdb", is_flag=True, help="Drop into pdb on failure")
@click.option(
    "--pdb-runcall", "pdb_runcall", is_flag=True, help="Invoke with pdb.runcall"
)
def cli(use_pdb, pdb_runcall):
    pass


@cli.command("uv-build")
@click.argument("script_path")
@click.option(
    "-e",
    "--expr-name",
    default="expr",
    help="Name of the expression variable in the Python script",
)
@click.option(
    "--builds-dir", default="builds", help="Directory for all generated artifacts"
)
@click.option(
    "--cache-dir",
    default=None,
    help="Directory for all generated parquet files cache",
)
def uv_build(script_path, expr_name, builds_dir, cache_dir):
    """Build an expression with a custom Python environment."""
    sys_argv = tuple(el if el != "uv-build" else "build" for el in sys.argv)
    uv_build_command(script_path, None, sys_argv)


@cli.command("uv-run")
@click.argument("build_path")
@click.option(
    "--cache-dir",
    default=None,
    help="Directory for all generated parquet files cache",
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
    help="Output format (default: parquet)",
)
def uv_run(build_path, cache_dir, output_path, output_format):
    """Run an expression with a custom Python environment."""
    sys_argv = tuple(el if el != "uv-run" else "run" for el in sys.argv)
    uv_run_command(build_path, sys_argv)


@cli.command("build")
@click.argument("script_path")
@click.option(
    "-e",
    "--expr-name",
    default="expr",
    help="Name of the expression variable in the Python script",
)
@click.option(
    "--builds-dir", default="builds", help="Directory for all generated artifacts"
)
@click.option(
    "--cache-dir",
    default=None,
    help="Directory for all generated parquet files cache",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Output SQL files and other debug artifacts",
)
def build(script_path, expr_name, builds_dir, cache_dir, debug):
    """Generate artifacts from an expression."""
    build_command(script_path, expr_name, builds_dir, cache_dir, debug)


@cli.command("run")
@click.argument("build_path")
@click.option(
    "--cache-dir",
    default=None,
    help="Directory for all generated parquet files cache",
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
    help="Output format (default: parquet)",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of rows to output",
)
def run(build_path, cache_dir, output_path, output_format, limit):
    """Run a build from a builds directory."""
    run_command(build_path, output_path, output_format, cache_dir, limit)


@cli.command("run-unbound")
@click.argument("build_path")
@click.option("--to_unbind_hash", default=None, help="Hash of the node to unbind")
@click.option("--to_unbind_tag", default=None, help="Tag of the node to unbind")
@click.option("--typ", default=None, help="Type of the node to unbind")
@click.option(
    "-o",
    "--output-path",
    default=None,
    help=f"Path to write output (default: stdout for arrow, {os.devnull} otherwise)",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice([f.value for f in OutputFormats]),
    default=OutputFormats.default,
    help=f"Output format (default: {OutputFormats.default})",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of rows to output",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for Arrow streaming output (default: use table default)",
)
@click.option(
    "--cache-dir",
    default=None,
    help="Directory for all generated parquet files cache",
)
@click.option(
    "-i",
    "--instream",
    type=click.File("rb"),
    default="-",
    help="Stream to read record batches from",
)
def run_unbound(
    build_path,
    to_unbind_hash,
    to_unbind_tag,
    typ,
    output_path,
    output_format,
    limit,
    batch_size,
    cache_dir,
    instream,
):
    """Run an unbound expr by reading Arrow IPC from stdin."""
    run_unbound_command(
        build_path,
        to_unbind_hash,
        to_unbind_tag,
        output_path,
        output_format,
        cache_dir,
        limit,
        typ,
        instream,
    )


@cli.command("serve-unbound")
@click.argument("build_path")
@click.option("--to_unbind_hash", default=None, help="Hash of the expr to replace")
@click.option("--to_unbind_tag", default=None, help="Tag of the expr to replace")
@click.option(
    "--host",
    default="localhost",
    help="Host to bind Flight Server (default: localhost)",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to bind Flight Server (default: random)",
)
@click.option(
    "--cache-dir",
    default=None,
    help="Directory for all generated parquet files cache",
)
@click.option("--typ", default=None, help="Type of the node to unbind")
@click.option(
    "--prometheus-port",
    type=int,
    default=None,
    help="Port to expose Prometheus metrics (default: disabled)",
)
def serve_unbound(
    build_path,
    to_unbind_hash,
    to_unbind_tag,
    host,
    port,
    cache_dir,
    typ,
    prometheus_port,
):
    """Serve an unbound expr via Flight Server."""
    unbind_and_serve_command(
        build_path,
        to_unbind_hash,
        to_unbind_tag,
        host,
        port,
        prometheus_port,
        cache_dir,
        typ,
    )


@cli.command("serve-flight-udxf")
@click.argument("build_path")
@click.option(
    "--host",
    default="localhost",
    help="Host to bind Flight Server (default: localhost)",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to bind Flight Server (default: random)",
)
@click.option(
    "--duckdb-path",
    default=None,
    help="Path to duckdb DB (default: <build_path>/xorq_serve.db)",
)
@click.option(
    "--prometheus-port",
    type=int,
    default=None,
    help="Port to expose Prometheus metrics (default: disabled)",
)
@click.option(
    "--cache-dir",
    default=None,
    help="Directory for all generated parquet files cache",
)
def serve_flight_udxf(build_path, host, port, duckdb_path, prometheus_port, cache_dir):
    """Serve a build via Flight Server."""
    serve_command(build_path, host, port, duckdb_path, prometheus_port, cache_dir)


@cli.command("init")
@click.option(
    "-p",
    "--path",
    default="./xorq-template",
    help="Path to initialize the template",
)
@click.option(
    "-t",
    "--template",
    type=click.Choice([str(t) for t in InitTemplates]),
    default=str(InitTemplates.default),
    help="Template to use",
)
@click.option(
    "-b",
    "--branch",
    default=None,
    help="Branch to use for the template",
)
def init(path, template, branch):
    """Initialize a xorq project."""
    init_command(path, template, branch)


_COMPLETION_INSTALL_PATHS = {
    "bash": Path("~/.local/share/bash-completion/completions/xorq").expanduser(),
    "zsh": Path("~/.zfunc/_xorq").expanduser(),
    "fish": Path("~/.config/fish/completions/xorq.fish").expanduser(),
}


def _get_completion_source(shell):
    from click.shell_completion import get_completion_class

    prog_name = "xorq"
    complete_var = "_XORQ_COMPLETE"
    comp_cls = get_completion_class(shell)
    comp = comp_cls(cli, {}, prog_name, complete_var)
    return comp.source()


def _detect_shell():
    shell_bin = Path(os.environ.get("SHELL", "")).name
    if shell_bin not in _COMPLETION_INSTALL_PATHS:
        raise click.UsageError(
            f"Cannot detect shell from $SHELL={os.environ.get('SHELL')!r}. "
            "Pass the shell name explicitly: bash, zsh, or fish."
        )
    return shell_bin


@cli.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), required=False)
def completion(shell):
    """Output shell completion script.

    SHELL defaults to the value of $SHELL if not provided.

    \b
    Add to your shell config:
      bash:  eval "$(xorq completion bash)"
      zsh:   eval "$(xorq completion zsh)"
      fish:  xorq completion fish | source
    """
    if shell is None:
        shell = _detect_shell()
    click.echo(_get_completion_source(shell), nl=False)


@cli.command("install-completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), required=False)
def install_completion(shell):
    """Install shell completion script to the standard location.

    SHELL defaults to the value of $SHELL if not provided.

    \b
    Install paths:
      bash:  ~/.local/share/bash-completion/completions/xorq
      zsh:   ~/.zfunc/_xorq  (requires ~/.zfunc in fpath)
      fish:  ~/.config/fish/completions/xorq.fish
    """
    if shell is None:
        shell = _detect_shell()

    install_path = _COMPLETION_INSTALL_PATHS[shell]
    install_path.parent.mkdir(parents=True, exist_ok=True)
    install_path.write_text(_get_completion_source(shell))
    click.echo(f"Installed {shell} completion to {install_path}")
    click.echo(f"Restart your shell or run: source {install_path}")


def _load_catalog_cli():
    from xorq.catalog.cli import cli as _catalog_cli

    cli.add_command(_catalog_cli, "catalog")


_load_catalog_cli()


def main():
    cli()


if __name__ == "__main__":
    main()
