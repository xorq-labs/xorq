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
from xorq.agent.onboarding import bootstrap_agent_docs
from xorq.agent.templates import (
    get_template,
    iter_templates,
    list_template_names,
    scaffold_template,
)
from xorq.caching.strategy import SnapshotStrategy
from xorq.catalog import (
    ServerRecord,
    catalog_command,
    lineage_command,
    load_catalog,
    ps_command,
    resolve_build_dir,
)
from xorq.common.utils import classproperty
from xorq.common.utils.caching_utils import get_xorq_cache_dir
from xorq.common.utils.import_utils import import_from_path
from xorq.common.utils.logging_utils import get_print_logger
from xorq.common.utils.node_utils import expr_to_unbound
from xorq.common.utils.otel_utils import tracer
from xorq.flight import FlightServer
from xorq.ibis_yaml.compiler import (
    build_expr,
    load_expr,
)
from xorq.ibis_yaml.packager import (
    SdistBuilder,
    SdistRunner,
)
from xorq.init_templates import InitTemplates
from xorq.loader import load_backend
from xorq.vendor.ibis import Expr


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class OutputFormats(StrEnum):
    csv = "csv"
    json = "json"
    parquet = "parquet"
    arrow = "arrow"

    @classproperty
    def default(self):
        return self.parquet


logger = get_print_logger()


def ensure_build_dir(expr_path):
    catalog = load_catalog()
    build_dir = resolve_build_dir(expr_path, catalog)
    if build_dir is None or not build_dir.exists() or not build_dir.is_dir():
        print(f"Build target not found: {expr_path}")
        sys.exit(2)
    return build_dir


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


@tracer.start_as_current_span("cli.run_command")
def run_command(
    expr_path,
    output_path=None,
    output_format=OutputFormats.default,
    cache_dir=get_xorq_cache_dir(),
    limit=None,
):
    """
    Execute an artifact

    Parameters
    ----------
    expr_path : str
        Build target: alias, alias@revision, entry_id, build_id, or path to build dir
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

    span = trace.get_current_span()
    span.add_event(
        "run.params",
        {
            "expr_path": str(expr_path),
            "output_path": str(output_path),
            "output_format": output_format,
        },
    )

    # Resolve build identifier (alias, entry_id, build_id, or path) to an actual build directory
    expr_path = ensure_build_dir(expr_path)
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


@tracer.start_as_current_span("cli.run_unbound_command")
def run_unbound_command(
    expr_path,
    to_unbind_hash=None,
    to_unbind_tag=None,
    output_path=None,
    output_format=OutputFormats.default,
    cache_dir=get_xorq_cache_dir(),
    limit=None,
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
    typ : str, optional
        Type of the node to unbind

    Returns
    -------

    """
    from xorq.expr.api import read_pyarrow_stream
    from xorq.flight.exchanger import replace_one_unbound

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

    # Read Arrow IPC from stdin
    print("[run-unbound] Reading Arrow IPC from stdin...", file=sys.stderr)
    # Create a connection and register the input table
    input_expr = read_pyarrow_stream(sys.stdin.buffer)
    # Replace the unbound node with the input table
    bound_expr = replace_one_unbound(unbound_expr, input_expr)

    if limit is not None:
        bound_expr = bound_expr.limit(limit)
    arbitrate_output_format(bound_expr, output_path, output_format)


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
    # Preserve original target token for server listing
    orig_target = expr_path
    # Resolve build identifier (alias, entry_id, build_id, or path) to an actual build directory
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


def agents_command(args):
    match args.agents_subcommand:
        case "init":
            return agents_init_command(args)
        case "onboard":
            return agent_onboard_command(args)
        case "land":
            return agent_land_command(args)
        case "templates":
            return agent_templates_command(args)
        case _:
            raise ValueError(f"Unknown agents subcommand: {args.agents_subcommand}")


def agents_init_command(args):
    path = Path(args.path)
    if not path.exists():
        print(
            f"Error: Path {path} does not exist. Please initialize a xorq project first with 'xorq init'"
        )
        return None

    # Parse comma-separated agent list
    agents = [a.strip().lower() for a in args.agents.split(",")]
    valid_agents = {"claude", "codex"}
    invalid = set(agents) - valid_agents
    if invalid:
        print(f"Warning: Unknown agents {invalid}. Valid options: {valid_agents}")
        agents = [a for a in agents if a in valid_agents]

    if not agents:
        print("No valid agents specified. Skipping agent setup.")
        return None

    created_files = bootstrap_agent_docs(path, agents=agents)
    if created_files:
        rel_paths = ", ".join(
            str(Path(file).relative_to(path)) for file in created_files
        )
        print(f"wrote agent onboarding files: {rel_paths}")
    else:
        print("agent onboarding files already present, skipping")
    return path


def agent_onboard_command(args):
    from xorq.agent.onboarding import render_onboarding_summary

    summary = render_onboarding_summary(step=args.step)
    print(summary.rstrip())


def agent_land_command(args):
    from xorq.agent.land import agent_land_command as land_cmd

    limit = getattr(args, "limit", 10)
    return land_cmd(args, limit=limit)


def agent_templates_command(args):
    match args.templates_command:
        case "list":
            return agent_templates_list_command()
        case "show":
            return agent_templates_show_command(args.name)
        case "scaffold":
            return agent_templates_scaffold_command(
                args.name, args.dest, args.overwrite
            )
        case _:
            raise ValueError(
                f"Unknown agent templates command: {args.templates_command}"
            )


def agent_templates_list_command():
    print(f"{'NAME':20} {'INIT TEMPLATE':18} DESCRIPTION")
    for template in iter_templates():
        print(f"{template.name:20} {template.template.value:18} {template.description}")


def agent_templates_show_command(name):
    template = get_template(name)
    lines = [
        f"# Template: {template.name}",
        f"Init template: {template.template.value}",
        f"Catalog alias hint: {template.catalog_hint}",
        f"Default table: {template.default_table}",
        "",
        "Prompts:",
    ]
    lines.extend(f"- {prompt}" for prompt in template.prompts)
    lines.extend(
        [
            "",
            "Suggested workflow:",
            f"- Run `xorq init --template {template.template.value}` if the project lacks this template",
            "- Customize the scaffolded expression and build it",
            f"- Register with `xorq catalog add ... --alias {template.catalog_hint}`",
        ]
    )
    print("\n".join(lines))


def agent_templates_scaffold_command(name, dest, overwrite):
    template = get_template(name)
    dest_path = Path(dest) if dest else Path("skills") / f"{template.name}.py"
    try:
        written = scaffold_template(template, dest_path, overwrite=overwrite)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(f"Wrote template scaffold to {written}")


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
        "--builds-dir",
        default=".xorq/builds",
        help="Directory for all generated artifacts",
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
        choices=OutputFormats,
        default=OutputFormats.default,
        type=OutputFormats,
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
        "--builds-dir",
        default=".xorq/builds",
        help="Directory for all generated artifacts",
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
    run_parser.add_argument(
        "build_path",
        help="Build target: alias, alias@revision, entry_id, build_id, or path to build dir",
    )
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
        "-f",
        "--format",
        choices=OutputFormats,
        default=OutputFormats.default,
        type=OutputFormats,
        help="Output format (default: parquet)",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows to output",
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
        choices=OutputFormats,
        # why was this arrow before and why did we fail when it was arrow?
        default=OutputFormats.default,
        type=OutputFormats,
        help=f"Output format (default: {OutputFormats.default})",
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
    catalog_parser.add_argument(
        "--namespace",
        help="Path to catalog namespace (default: .xorq/catalog.yaml or ~/.config/xorq/catalog.yaml)",
        default=None,
    )
    catalog_subparsers = catalog_parser.add_subparsers(
        dest="subcommand", help="Catalog commands"
    )
    catalog_subparsers.required = True

    catalog_subparsers.add_parser(
        "init",
        help="Initialize a catalog namespace (creates .xorq/catalog.yaml by default)",
    )

    catalog_add = catalog_subparsers.add_parser(
        "add", help="Add a build to the catalog"
    )
    catalog_add.add_argument("build_path", help="Path to the build directory")
    catalog_add.add_argument(
        "-a", "--alias", help="Optional alias for this entry", default=None
    )
    catalog_ls = catalog_subparsers.add_parser("ls", help="List catalog entries")
    catalog_ls.add_argument(
        "--quiet", "-q", action="store_true", help="Only show alias names"
    )
    catalog_ls.add_argument("--json", action="store_true", help="Output in JSON format")

    catalog_subparsers.add_parser("info", help="Show catalog information")
    catalog_rm = catalog_subparsers.add_parser(
        "rm", help="Remove a build entry or alias from the catalog"
    )
    catalog_rm.add_argument("entry", help="Entry ID or alias to remove")

    catalog_export = catalog_subparsers.add_parser(
        "export", help="Export catalog and builds to a directory"
    )
    catalog_export.add_argument(
        "output_path", help="Directory to export catalog and builds"
    )

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

    # New catalog composition helper commands
    catalog_sources = catalog_subparsers.add_parser(
        "sources", help="List source nodes in an expression (for composition)"
    )
    catalog_sources.add_argument("alias", help="Catalog alias or entry to inspect")
    catalog_sources.add_argument(
        "--show-schema",
        action="store_true",
        help="Show schema details for each source node",
    )

    catalog_schema = catalog_subparsers.add_parser(
        "schema", help="Show output schema of a cataloged expression"
    )
    catalog_schema.add_argument("alias", help="Catalog alias or entry to inspect")
    catalog_schema.add_argument(
        "--json", action="store_true", help="Output schema as JSON"
    )

    catalog_search = catalog_subparsers.add_parser(
        "search-source-schema",
        help="Search catalog for transforms accepting a schema (reads JSON from stdin)",
    )
    catalog_search.add_argument(
        "--exact-only", action="store_true", help="Only show exact schema matches"
    )

    agents_parser = subparsers.add_parser(
        "agents",
        help="Agent-native helpers built on top of xorq primitives",
    )
    agents_subparsers = agents_parser.add_subparsers(
        dest="agents_subcommand",
        help="Agent helper commands",
    )
    agents_subparsers.required = True

    agents_init_parser = agents_subparsers.add_parser(
        "init",
        help="Bootstrap agent guides (claude, codex, or both)",
    )
    agents_init_parser.add_argument(
        "-p",
        "--path",
        type=Path,
        default=".",
        help="Path to the xorq project directory",
    )
    agents_init_parser.add_argument(
        "--agents",
        type=str,
        default="claude,codex",
        help="Comma-separated list of agents to bootstrap (claude, codex, or both)",
    )

    onboard_parser = agents_subparsers.add_parser(
        "onboard",
        help="Guided onboarding summary for xorq agents",
    )
    onboard_parser.add_argument(
        "--step",
        choices=("init", "templates", "build", "catalog", "explore", "compose", "land"),
        default=None,
        help="Filter onboarding instructions to a specific step",
    )

    land_parser = agents_subparsers.add_parser(
        "land",
        help="Show session summary and landing checklist",
    )
    land_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of recent builds to show (default: 10)",
    )

    templates_parser = agents_subparsers.add_parser(
        "templates",
        help="Template registry commands",
    )
    templates_subparsers = templates_parser.add_subparsers(
        dest="templates_command",
        help="Templates commands",
    )
    templates_subparsers.required = True

    templates_subparsers.add_parser("list", help="List registered templates")

    templates_show = templates_subparsers.add_parser(
        "show", help="Show details for a specific template"
    )
    templates_show.add_argument(
        "name",
        choices=list_template_names(),
        help="Template identifier",
    )

    templates_scaffold = templates_subparsers.add_parser(
        "scaffold", help="Scaffold an expression file for a template"
    )
    templates_scaffold.add_argument(
        "name",
        choices=list_template_names(),
        help="Template identifier",
    )
    templates_scaffold.add_argument(
        "--dest",
        default=None,
        help="Destination path for the scaffold (default: skills/<name>.py)",
    )
    templates_scaffold.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination file if it exists",
    )

    args = parser.parse_args(override)
    return args


def main():
    """Main entry point for the xorq CLI."""
    args = parse_args()

    try:
        match args.command:
            case "uv-build":
                # Convert builds-dir to absolute path so it works correctly in uv environment
                sys_argv = list(el if el != "uv-build" else "build" for el in sys.argv)
                # Find and convert --builds-dir to absolute path
                builds_dir_abs = str(Path(args.builds_dir).resolve())
                if "--builds-dir" in sys_argv:
                    idx = sys_argv.index("--builds-dir")
                    sys_argv[idx + 1] = builds_dir_abs
                else:
                    # Add --builds-dir with absolute path
                    sys_argv.extend(["--builds-dir", builds_dir_abs])
                sys_argv = tuple(sys_argv)
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
                        args.typ,
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
            case "agents":
                f, f_args = (
                    agents_command,
                    (args,),
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
