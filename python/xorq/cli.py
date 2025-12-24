import argparse
import os
import pdb
import sys
import traceback
from functools import partial
from pathlib import Path

from opentelemetry import trace

import xorq
import xorq.common.utils.pickle_utils  # noqa: F401
from xorq.catalog import (
    ServerRecord,
    catalog_command,
    lineage_command,
    ps_command,
    resolve_build_dir,
)
from xorq.common.utils.caching_utils import get_xorq_cache_dir
from xorq.common.utils.import_utils import import_from_path
from xorq.common.utils.logging_utils import get_print_logger
from xorq.common.utils.node_utils import expr_to_unbound
from xorq.common.utils.otel_utils import tracer
from xorq.flight import FlightServer
from xorq.ibis_yaml.compiler import (
    BuildManager,
    load_expr,
)
from xorq.ibis_yaml.packager import (
    SdistBuilder,
    SdistRunner,
)
from xorq.init_templates import InitTemplates
from xorq.loader import load_backend
from xorq.vendor.ibis import Expr


logger = get_print_logger()


@tracer.start_as_current_span("cli.uv_build_command")
def uv_build_command(
    script_path,
    project_path=None,
    sys_argv=(),
):
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


@tracer.start_as_current_span("cli.uv_run_command")
def uv_run_command(
    expr_path,
    sys_argv=(),
):
    sdist_runner = SdistRunner(expr_path, args=sys_argv)
    popened = sdist_runner._uv_tool_run_xorq_run
    return popened


@tracer.start_as_current_span("cli.build_command")
def build_command(
    script_path,
    expr_name,
    builds_dir="builds",
    cache_dir=get_xorq_cache_dir(),
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

    build_manager = BuildManager(
        builds_dir,
        cache_dir=Path(cache_dir),
        debug=debug,
    )

    vars_module = import_from_path(script_path, module_name="__main__")

    if not hasattr(vars_module, expr_name):
        raise ValueError(f"Expression {expr_name} not found")

    expr = getattr(vars_module, expr_name)

    if not isinstance(expr, Expr):
        raise ValueError(
            f"The object {expr_name} must be an instance of {Expr.__module__}.{Expr.__name__}"
        )

    expr_hash = build_manager.compile_expr(expr)
    span.add_event("build.outputs", {"expr_hash": expr_hash})
    print(
        f"Written '{expr_name}' to {build_manager.artifact_store.get_path(expr_hash)}",
        file=sys.stderr,
    )
    print(build_manager.artifact_store.get_path(expr_hash))


@tracer.start_as_current_span("cli.run_command")
def run_command(
    expr_path,
    output_path=None,
    output_format="parquet",
    cache_dir=get_xorq_cache_dir(),
    limit=None,
    input_format="none",
    batch_size=None,
):
    """
    Execute an artifact

    Parameters
    ----------
    expr_path : str
        Path to the expr in the builds dir
    output_path : str
        Path to write output. Defaults to os.devnull
    output_format : str, optional
        Output format, either "csv", "json", "parquet", or "arrow". Defaults to "parquet"
    cache_dir : Path, optional
        Directory where the parquet cache files will be generated
    limit : int, optional
        Limit number of rows to output. Defaults to None (no limit).
    input_format : str, optional
        Input format for reading from stdin, either "arrow" or "none". Defaults to "none"
    batch_size : int, optional
        Batch size for Arrow streaming output. Defaults to None (use table default)

    Returns
    -------

    """
    from xorq.cli.io import read_arrow_stream, write_arrow_stream

    span = trace.get_current_span()
    span.add_event(
        "run.params",
        {
            "expr_path": str(expr_path),
            "output_path": str(output_path),
            "output_format": output_format,
            "input_format": input_format,
        },
    )

    if output_path is None:
        output_path = os.devnull

    expr_path = Path(expr_path)
    build_manager = BuildManager(expr_path.parent, cache_dir=cache_dir)
    expr = build_manager.load_expr(expr_path.name)

    # Handle input from stdin if specified
    if input_format == "arrow":
        input_table = read_arrow_stream()
        # Bind the input table to the expression
        # This assumes the expression has an unbound parameter named "input"
        expr = expr.bind(input=input_table)

    if limit is not None:
        expr = expr.limit(limit)

    match output_format:
        case "arrow":
            # For arrow output, write to stdout.buffer or the specified path
            if output_path == os.devnull:
                # If no output specified, write to stdout
                output_stream = sys.stdout.buffer
            elif hasattr(output_path, "write"):
                # Already a file-like object
                output_stream = output_path
            else:
                # Open the file for writing
                output_stream = open(output_path, "wb")

            try:
                # Use to_pyarrow_batches if available (Ibis expression)
                # Otherwise convert result to Arrow

                if batch_size is not None:
                    batches = expr.to_pyarrow_batches(chunk_size=batch_size)
                else:
                    batches = expr.to_pyarrow_batches()

                write_arrow_stream(batches, out=output_stream)
            finally:
                if not hasattr(output_path, "write") and output_path != os.devnull:
                    output_stream.close()
        case "csv":
            expr.to_csv(output_path)
        case "json":
            expr.to_json(output_path)
        case "parquet":
            expr.to_parquet(output_path)
        case _:
            raise ValueError(f"Unknown output_format: {output_format}")


@tracer.start_as_current_span("cli.run_unbound_command")
def run_unbound_command(
    expr_path,
    to_unbind_hash=None,
    to_unbind_tag=None,
    output_path=None,
    output_format="parquet",
    cache_dir=get_xorq_cache_dir(),
    limit=None,
    batch_size=None,
    typ=None,
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
    batch_size : int, optional
        Batch size for Arrow streaming output. Defaults to None (use table default)
    typ : str, optional
        Type of the node to unbind

    Returns
    -------

    """
    from xorq.cli.io import read_arrow_stream

    span = trace.get_current_span()
    span.add_event(
        "run_unbound.params",
        {
            "expr_path": str(expr_path),
            "to_unbind_hash": to_unbind_hash,
            "to_unbind_tag": to_unbind_tag,
            "output_format": output_format,
        },
    )

    if output_path is None:
        output_path = os.devnull

    # Resolve build identifier
    build_dir = resolve_build_dir(expr_path)
    if build_dir is None or not build_dir.exists() or not build_dir.is_dir():
        print(f"Build target not found: {expr_path}")
        sys.exit(2)

    expr_path = Path(build_dir)
    # Log to stderr to avoid polluting Arrow streams
    print(f"[run-unbound] Loading expression from {expr_path}", file=sys.stderr)

    # Load the expression and make it unbound
    expr = load_expr(expr_path)
    unbound_op = expr_to_unbound(expr, hash=to_unbind_hash, tag=to_unbind_tag, typs=typ)
    unbound_expr = unbound_op.to_expr()

    # Read Arrow IPC from stdin
    print("[run-unbound] Reading Arrow IPC from stdin...", file=sys.stderr)
    input_table = read_arrow_stream()
    print(
        f"[run-unbound] Received table: {input_table.num_rows} rows, {input_table.num_columns} columns",
        file=sys.stderr,
    )

    # Bind the input table to the unbound expression
    import xorq.api as xo
    from xorq.flight.exchanger import replace_one_unbound

    # Create a connection and register the input table
    con = xo.connect()
    input_ibis_table = con.read_record_batches(input_table)

    # Replace the unbound node with the input table
    bound_expr = replace_one_unbound(unbound_expr, input_ibis_table)

    if limit is not None:
        bound_expr = bound_expr.limit(limit)

    # Execute and output using the same logic as run_command
    match output_format:
        case "arrow":
            if output_path == os.devnull:
                output_stream = sys.stdout.buffer
            elif hasattr(output_path, "write"):
                output_stream = output_path
            else:
                output_stream = open(output_path, "wb")

            try:
                if batch_size is not None:
                    batches = bound_expr.to_pyarrow_batches(chunk_size=batch_size)
                else:
                    batches = bound_expr.to_pyarrow_batches()

                from xorq.cli.io import write_arrow_stream

                write_arrow_stream(batches, out=output_stream)
            finally:
                if not hasattr(output_path, "write") and output_path != os.devnull:
                    output_stream.close()
        case "csv":
            bound_expr.to_csv(output_path)
        case "json":
            bound_expr.to_json(output_path)
        case "parquet":
            bound_expr.to_parquet(output_path)
        case _:
            raise ValueError(f"Unknown output_format: {output_format}")


@tracer.start_as_current_span("cli.tee_command")
def tee_command(
    output_paths,
    input_format="arrow",
    append=False,
):
    """
    Read Arrow IPC from stdin and write to both stdout and file(s) (like Unix tee)

    Parameters
    ----------
    output_paths : list[str]
        List of file paths to write to (in addition to stdout)
    input_format : str, optional
        Input format, currently only "arrow" is supported
    append : bool, optional
        Append to files instead of overwriting (default: False)

    Returns
    -------

    """
    import pyarrow as pa

    if input_format != "arrow":
        raise ValueError("tee command currently only supports Arrow IPC streams")

    # Read Arrow stream from stdin (streaming, not buffered)
    # Note: No logging here to avoid polluting stdout Arrow stream
    reader = pa.ipc.open_stream(sys.stdin.buffer)
    schema = reader.schema

    # Open all output files
    mode = "ab" if append else "wb"
    output_files = [open(path, mode) for path in output_paths]

    # Create writers for stdout and all files
    stdout_writer = pa.ipc.new_stream(sys.stdout.buffer, schema)
    file_writers = [pa.ipc.new_stream(f, schema) for f in output_files]
    all_writers = [stdout_writer] + file_writers

    try:
        # Stream batches incrementally
        batch_count = 0
        for batch in reader:
            # Write the same batch to all outputs simultaneously
            for writer in all_writers:
                writer.write_batch(batch)
            batch_count += 1

        # Log to stderr after all Arrow data is written
        print(
            f"[tee] Streamed {batch_count} batches to {len(output_paths) + 1} destination(s)",
            file=sys.stderr,
        )
    finally:
        # Close all writers
        for writer in all_writers:
            writer.close()
        # Close all file handles
        for f in output_files:
            f.close()


@tracer.start_as_current_span("cli.unbind_and_serve_command")
def unbind_and_serve_command(
    expr_path,
    to_unbind_hash=None,
    to_unbind_tag=None,
    host=None,
    port=None,
    prometheus_port=None,
    cache_dir=get_xorq_cache_dir(),
    typ=None,
):
    import functools

    # Preserve original target token for server listing
    orig_target = expr_path
    # Resolve build identifier (alias, entry_id, build_id, or path) to an actual build directory
    build_dir = resolve_build_dir(expr_path)
    if build_dir is None or not build_dir.exists() or not build_dir.is_dir():
        print(f"Build target not found: {expr_path}")
        sys.exit(2)
    expr_path = Path(build_dir)
    logger.info(f"Loading expression from {expr_path}")
    try:
        # initialize console and optional Prometheus metrics
        from xorq.flight.metrics import setup_console_metrics

        setup_console_metrics(prometheus_port=prometheus_port)
    except ImportError:
        logger.warning(
            "Metrics support requires 'opentelemetry-sdk' and console exporter"
        )

    expr = load_expr(expr_path)
    unbound_expr = expr_to_unbound(
        expr, hash=to_unbind_hash, tag=to_unbind_tag, typs=typ
    )
    flight_url = xorq.flight.FlightUrl(host=host, port=port)
    make_server = functools.partial(
        xorq.flight.FlightServer,
        flight_url=flight_url,
    )
    logger.info(f"Serving expression from '{expr_path}' on {flight_url.to_location()}")
    server, _ = xorq.expr.relations.flight_serve_unbound(
        unbound_expr, make_server=make_server
    )
    # Record server metadata
    rec = ServerRecord(
        pid=os.getpid(),
        command="serve-unbound",
        target=orig_target,
        port=flight_url.port,
        node_hash=to_unbind_hash,
    )
    rec.save(Path(cache_dir) / "servers")
    server.wait()


@tracer.start_as_current_span("cli.serve_command")
def serve_command(
    expr_path,
    host=None,
    port=None,
    duckdb_path=None,
    prometheus_port=None,
    cache_dir=get_xorq_cache_dir(),
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

    # Preserve original target token for server listing
    orig_target = expr_path
    # Resolve build identifier (alias, entry_id, build_id, or path) to an actual build directory
    build_dir = resolve_build_dir(expr_path)
    if build_dir is None or not build_dir.exists() or not build_dir.is_dir():
        print(f"Build target not found: {expr_path}")
        sys.exit(2)
    expr_path = build_dir
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
    expr = load_expr(expr_path)

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
    # Record server metadata
    rec = ServerRecord(
        pid=os.getpid(),
        command="serve-flight-udxf",
        target=orig_target,
        port=server.flight_url.port,
    )
    rec.save(Path(cache_dir) / "servers")
    location = server.flight_url.to_location()
    logger.info(f"Serving expression '{expr_path.stem}' on {location}")
    server.serve(block=True)


@tracer.start_as_current_span("cli.init_command")
def init_command(
    path="./xorq-template",
    template=InitTemplates.default,
    branch=None,
):
    from xorq.common.utils.download_utils import download_unpacked_xorq_template

    path = download_unpacked_xorq_template(path, template, branch=branch)
    print(f"initialized xorq template `{template}` to {path}")
    return path


def parse_args(override=None):
    parser = argparse.ArgumentParser(
        description="xorq - build, run, and serve expressions"
    )
    parser.add_argument("--pdb", action="store_true", help="Drop into pdb on failure")
    parser.add_argument(
        "--pdb-runcall", action="store_true", help="Invoke with pdb.runcall"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    uv_build_parser = subparsers.add_parser(
        "uv-build", help="Build an expression with a custom Python environment"
    )
    uv_build_parser.add_argument("script_path", help="Path to the Python script")
    uv_build_parser.add_argument(
        "-e",
        "--expr-name",
        default="expr",
        help="Name of the expression variable in the Python script",
    )
    uv_build_parser.add_argument(
        "--builds-dir", default="builds", help="Directory for all generated artifacts"
    )
    uv_build_parser.add_argument(
        "--cache-dir",
        required=False,
        default=get_xorq_cache_dir(),
        help="Directory for all generated parquet files cache",
    )

    uv_run_parser = subparsers.add_parser(
        "uv-run", help="Run an expression with a custom Python environment"
    )
    uv_run_parser.add_argument("build_path", help="Path to the build script")
    uv_run_parser.add_argument(
        "--cache-dir",
        required=False,
        default=get_xorq_cache_dir(),
        help="Directory for all generated parquet files cache",
    )
    uv_run_parser.add_argument(
        "-o",
        "--output-path",
        default=None,
        help=f"Path to write output (default: {os.devnull})",
    )
    uv_run_parser.add_argument(
        "-f",
        "--format",
        choices=["csv", "json", "parquet"],
        default="parquet",
        help="Output format (default: parquet)",
    )

    build_parser = subparsers.add_parser(
        "build", help="Generate artifacts from an expression"
    )
    build_parser.add_argument("script_path", help="Path to the Python script")
    build_parser.add_argument(
        "-e",
        "--expr-name",
        default="expr",
        help="Name of the expression variable in the Python script",
    )
    build_parser.add_argument(
        "--builds-dir", default="builds", help="Directory for all generated artifacts"
    )
    build_parser.add_argument(
        "--cache-dir",
        required=False,
        default=get_xorq_cache_dir(),
        help="Directory for all generated parquet files cache",
    )
    build_parser.add_argument(
        "--debug",
        action="store_true",
        help="Output SQL files and other debug artifacts",
    )

    run_parser = subparsers.add_parser(
        "run", help="Run a build from a builds directory"
    )
    run_parser.add_argument("build_path", help="Path to the build script")
    run_parser.add_argument(
        "--cache-dir",
        required=False,
        default=get_xorq_cache_dir(),
        help="Directory for all generated parquet files cache",
    )
    run_parser.add_argument(
        "-o",
        "--output-path",
        default=None,
        help=f"Path to write output (default: {os.devnull})",
    )
    run_parser.add_argument(
        "-i",
        "--input",
        choices=["arrow", "none"],
        default="none",
        help="Input format for reading from stdin (default: none)",
    )
    run_parser.add_argument(
        "-f",
        "--format",
        choices=["csv", "json", "parquet", "arrow"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to output",
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for Arrow streaming output (default: use table default)",
    )

    run_unbound_parser = subparsers.add_parser(
        "run-unbound", help="Run an unbound expr by reading Arrow IPC from stdin"
    )
    run_unbound_parser.add_argument(
        "build_path",
        help="Build target: alias, entry_id, build_id, or path to build dir",
    )
    run_unbound_parser.add_argument(
        "--to_unbind_hash", default=None, help="Hash of the node to unbind"
    )
    run_unbound_parser.add_argument(
        "--to_unbind_tag", default=None, help="Tag of the node to unbind"
    )
    run_unbound_parser.add_argument(
        "--typ",
        required=False,
        default=None,
        help="Type of the node to unbind",
    )
    run_unbound_parser.add_argument(
        "-o",
        "--output-path",
        default=None,
        help=f"Path to write output (default: stdout for arrow, {os.devnull} otherwise)",
    )
    run_unbound_parser.add_argument(
        "-f",
        "--format",
        choices=["csv", "json", "parquet", "arrow"],
        default="arrow",
        help="Output format (default: arrow)",
    )
    run_unbound_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to output",
    )
    run_unbound_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for Arrow streaming output (default: use table default)",
    )
    run_unbound_parser.add_argument(
        "--cache-dir",
        required=False,
        default=get_xorq_cache_dir(),
        help="Directory for all generated parquet files cache",
    )

    tee_parser = subparsers.add_parser(
        "tee", help="Read Arrow IPC from stdin and write to both stdout and file(s)"
    )
    tee_parser.add_argument(
        "output_paths",
        nargs="+",
        help="File path(s) to write to (in addition to stdout)",
    )
    tee_parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append to files instead of overwriting",
    )

    serve_unbound_parser = subparsers.add_parser(
        "serve-unbound", help="Serve an an unbound expr via Flight Server"
    )
    serve_unbound_parser.add_argument(
        "build_path",
        help="Build target: alias, entry_id, build_id, or path to build dir",
    )
    serve_unbound_parser.add_argument(
        "--to_unbind_hash", default=None, help="hash of the expr to replace"
    )
    serve_unbound_parser.add_argument(
        "--to_unbind_tag", default=None, help="tag of the expr to replace"
    )
    serve_unbound_parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind Flight Server (default: localhost)",
    )
    serve_unbound_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind Flight Server (default: random)",
    )
    serve_unbound_parser.add_argument(
        "--cache-dir",
        required=False,
        default=get_xorq_cache_dir(),
        help="Directory for all generated parquet files cache",
    )
    serve_unbound_parser.add_argument(
        "--typ",
        required=False,
        default=None,
        help="type of the node to unbind",
    )
    serve_unbound_parser.add_argument(
        "--prometheus-port",
        type=int,
        default=None,
        help="Port to expose Prometheus metrics (default: disabled)",
    )
    serve_parser = subparsers.add_parser(
        "serve-flight-udxf", help="Serve a build via Flight Server"
    )
    serve_parser.add_argument(
        "build_path",
        help="Build target: alias, entry_id, build_id, or path to build dir",
    )
    serve_parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind Flight Server (default: localhost)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind Flight Server (default: random)",
    )
    serve_parser.add_argument(
        "--duckdb-path",
        default=None,
        help="Path to duckdb DB (default: <build_path>/xorq_serve.db)",
    )
    serve_parser.add_argument(
        "--prometheus-port",
        type=int,
        default=None,
        help="Port to expose Prometheus metrics (default: disabled)",
    )
    serve_parser.add_argument(
        "--cache-dir",
        required=False,
        default=get_xorq_cache_dir(),
        help="Directory for all generated parquet files cache",
    )

    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a xorq project",
    )
    init_parser.add_argument(
        "-p",
        "--path",
        type=Path,
        default="./xorq-template",
    )
    init_parser.add_argument(
        "-t",
        "--template",
        choices=tuple(InitTemplates),
        default=InitTemplates.cached_fetcher,
    )
    init_parser.add_argument(
        "-b",
        "--branch",
        default=None,
    )
    lineage_parser = subparsers.add_parser(
        "lineage",
        help="Print lineage trees of all columns for a build",
    )
    lineage_parser.add_argument(
        "target",
        help="Build target: alias, entry_id, build_id, or path to build dir",
    )
    ps_parser = subparsers.add_parser(
        "ps",
        help="List running xorq servers",
    )
    ps_parser.add_argument(
        "--cache-dir",
        required=False,
        default=get_xorq_cache_dir(),
        help="Directory for server state records",
    )
    catalog_parser = subparsers.add_parser("catalog", help="Manage build catalog")
    catalog_subparsers = catalog_parser.add_subparsers(
        dest="subcommand", help="Catalog commands"
    )
    catalog_subparsers.required = True

    catalog_add = catalog_subparsers.add_parser(
        "add", help="Add a build to the catalog"
    )
    catalog_add.add_argument("build_path", help="Path to the build directory")
    catalog_add.add_argument(
        "-a", "--alias", help="Optional alias for this entry", default=None
    )
    catalog_subparsers.add_parser("ls", help="List catalog entries")

    catalog_subparsers.add_parser("info", help="Show catalog information")
    catalog_rm = catalog_subparsers.add_parser(
        "rm", help="Remove a build entry or alias from the catalog"
    )
    catalog_rm.add_argument("entry", help="Entry ID or alias to remove")
    catalog_diff_builds = catalog_subparsers.add_parser(
        "diff-builds", help="Compare two build artifacts via git diff --no-index"
    )
    catalog_diff_builds.add_argument(
        "left",
        help="Left build target: alias, entry_id, build_id, or path to build dir",
    )
    catalog_diff_builds.add_argument(
        "right",
        help="Right build target: alias, entry_id, build_id, or path to build dir",
    )
    catalog_diff_builds.add_argument(
        "--all",
        action="store_true",
        help="Diff all known build files plus all .sql files",
    )
    catalog_diff_builds.add_argument(
        "--files",
        nargs="+",
        help="Explicit list of relative files to diff (overrides --all)",
        default=None,
    )

    args = parser.parse_args(override)
    if getattr(args, "output_path", None) == "-":
        if args.format == "json":
            # FIXME: deal with windows
            args.output_path = sys.stdout
        else:
            args.output_path = sys.stdout.buffer
    return args


def main():
    """Main entry point for the xorq CLI."""
    args = parse_args()

    try:
        match args.command:
            case "uv-build":
                sys_argv = tuple(el if el != "uv-build" else "build" for el in sys.argv)
                f, f_args = (
                    uv_build_command,
                    (args.script_path, None, sys_argv),
                )
            case "uv-run":
                sys_argv = tuple(el if el != "uv-run" else "run" for el in sys.argv)
                f, f_args = (
                    uv_run_command,
                    (args.build_path, sys_argv),
                )
            case "build":
                f, f_args = (
                    build_command,
                    (
                        args.script_path,
                        args.expr_name,
                        args.builds_dir,
                        args.cache_dir,
                        args.debug,
                    ),
                )
            case "run":
                f, f_args = (
                    run_command,
                    (
                        args.build_path,
                        args.output_path,
                        args.format,
                        args.cache_dir,
                        args.limit,
                        args.input,
                        getattr(args, "batch_size", None),
                    ),
                )
            case "run-unbound":
                f, f_args = (
                    run_unbound_command,
                    (
                        args.build_path,
                        args.to_unbind_hash,
                        args.to_unbind_tag,
                        args.output_path,
                        args.format,
                        args.cache_dir,
                        args.limit,
                        getattr(args, "batch_size", None),
                        args.typ,
                    ),
                )
            case "tee":
                f, f_args = (
                    tee_command,
                    (
                        args.output_paths,
                        "arrow",
                        args.append,
                    ),
                )
            case "serve-unbound":
                f, f_args = (
                    unbind_and_serve_command,
                    (
                        args.build_path,
                        args.to_unbind_hash,
                        args.to_unbind_tag,
                        args.host,
                        args.port,
                        args.prometheus_port,
                        args.cache_dir,
                        args.typ,
                    ),
                )
            case "serve-flight-udxf":
                # Serve a Flight UDXF build
                f, f_args = (
                    serve_command,
                    (
                        args.build_path,
                        args.host,
                        args.port,
                        args.duckdb_path,
                        args.prometheus_port,
                        args.cache_dir,
                    ),
                )
            case "init":
                f, f_args = (
                    init_command,
                    (args.path, args.template, args.branch),
                )
            case "lineage":
                f, f_args = (
                    lineage_command,
                    (args.target,),
                )
            case "catalog":
                f, f_args = (
                    catalog_command,
                    (args,),
                )
            case "ps":
                f, f_args = (
                    ps_command,
                    (args.cache_dir,),
                )
            case _:
                raise ValueError(f"Unknown command: {args.command}")
        match args.pdb_runcall:
            case True:
                pdb.runcall(f, *f_args)
            case False:
                f(*f_args)
            case _:
                raise ValueError(f"Unknown value for pdb_runcall: {args.pdb_runcall}")
    except Exception as e:
        if args.pdb:
            traceback.print_exception(e)
            pdb.post_mortem(e.__traceback__)
        else:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
