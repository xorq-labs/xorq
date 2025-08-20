import argparse
import os
import pdb
import sys
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path

from opentelemetry import trace

import xorq as xo
import xorq.common.utils.pickle_utils  # noqa: F401

# Helper functions for diff-builds subcommand
from xorq.catalog import (
    cache_command,
    catalog_command,
    do_save_server_record,
    lineage_command,
    load_catalog,
    make_server_record,
    profile_command,
    ps_command,
    resolve_build_dir,
)
from xorq.common.utils import classproperty
from xorq.common.utils.caching_utils import get_xorq_cache_dir
from xorq.common.utils.import_utils import import_from_path
from xorq.common.utils.logging_utils import get_print_logger
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
from xorq.vendor.ibis import Expr


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


logger = get_print_logger()


class InitTemplates(StrEnum):
    cached_fetcher = "cached-fetcher"
    sklearn = "sklearn"
    penguins = "penguins"

    @classproperty
    def default(self):
        return self.cached_fetcher


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
    script_path, expr_name, builds_dir="builds", cache_dir=get_xorq_cache_dir()
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

    build_manager = BuildManager(builds_dir, cache_dir=Path(cache_dir))

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
    expr_path, output_path=None, output_format="parquet", cache_dir=get_xorq_cache_dir()
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
        Output format, either "csv", "json", or "parquet". Defaults to "parquet"

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

    if output_path is None:
        output_path = os.devnull

    expr_path = Path(expr_path)
    build_manager = BuildManager(expr_path.parent, cache_dir=cache_dir)
    expr = build_manager.load_expr(expr_path.name)

    match output_format:
        case "csv":
            expr.to_csv(output_path)
        case "json":
            expr.to_json(output_path)
        case "parquet":
            expr.to_parquet(output_path)
        case _:
            raise ValueError(f"Unknown output_format: {output_format}")


@tracer.start_as_current_span("cli.unbind_and_serve_command")
def unbind_and_serve_command(
    expr_path,
    to_unbind_hash,
    host=None,
    port=None,
    prometheus_port=None,
    cache_dir=get_xorq_cache_dir(),
    typ=None,
):
    import functools

    from xorq.common.utils.graph_utils import (
        find_all_sources,
    )
    from xorq.common.utils.node_utils import (
        find_by_expr_hash,
    )

    # Preserve original target token for server listing
    orig_target = expr_path
    # Resolve build identifier (alias, entry_id, build_id, or path) to an actual build directory
    build_dir = resolve_build_dir(expr_path, load_catalog())
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

    def expr_to_unbound(expr, to_unbind_hash):
        """create an unbound expr that only needs to have a source of record batches fed in"""
        from xorq.common.utils.graph_utils import (
            replace_nodes,
            walk_nodes,
        )
        from xorq.common.utils.node_utils import (
            elide_downstream_cached_node,
            replace_by_expr_hash,
        )
        from xorq.vendor.ibis.expr.operations import UnboundTable

        found_cons = find_all_sources(expr)
        found = find_by_expr_hash(expr, to_unbind_hash, typs=typ)

        if len(found_cons) == 0:
            raise ValueError(
                f"No sources found to unbind for expression hash: {to_unbind_hash}"
            )
        elif len(found_cons) == 1:
            (found_con,) = found_cons
        else:
            (found_con,) = find_all_sources(found)
        # Log the node being replaced and its dask hash
        try:
            import dask

            node_hash = dask.base.tokenize(found.to_expr())
            logger.info(f"Replacing node {type(found).__name__} with hash {node_hash}")
        except Exception:
            logger.info(f"Replacing node {type(found).__name__}")
            subtree_cons = find_all_sources(found)
            if len(subtree_cons) == 1:
                found_con = subtree_cons[0]
            else:
                raise ValueError(
                    f"Multiple sources found for expr hash {to_unbind_hash}: "
                    + ", ".join(f"[{i}]: {src}" for i, src in enumerate(subtree_cons))
                )

        import dask

        node_hash = dask.base.tokenize(found.to_expr())
        logger.info(f"Unbinding with node {type(found).__name__} with hash {node_hash}")

        unbound_table = UnboundTable("unbound", found.schema)
        replace_with = unbound_table.to_expr().into_backend(found_con).op()
        replaced = replace_by_expr_hash(
            expr, to_unbind_hash, replace_with, typs=(type(found),)
        )
        (found,) = walk_nodes(UnboundTable, replaced)
        elided = replace_nodes(elide_downstream_cached_node(replaced, found), replaced)
        return elided

    unbound_expr = expr_to_unbound(expr, to_unbind_hash)
    flight_url = xo.flight.FlightUrl(host=host, port=port)
    make_server = functools.partial(
        xo.flight.FlightServer,
        flight_url=flight_url,
    )
    logger.info(f"Serving expression from '{expr_path}' on {flight_url.to_location()}")
    server, _ = xo.expr.relations.flight_serve_unbound(
        unbound_expr, make_server=make_server
    )
    # Record server metadata
    rec = make_server_record(
        pid=os.getpid(),
        command="serve-unbound",
        target=orig_target,
        port=flight_url.port,
        start_time=datetime.now(),
        node_hash=to_unbind_hash,
    )
    do_save_server_record(rec, Path(cache_dir) / "servers")
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
    catalog = load_catalog()
    build_dir = resolve_build_dir(expr_path, catalog)
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
        make_connection=partial(xo.duckdb.connect, str(db_path)),
        port=port,
        host=host,
    )
    # Record server metadata
    rec = make_server_record(
        pid=os.getpid(),
        command="serve-flight-udxf",
        target=orig_target,
        port=server.flight_url.port,
        start_time=datetime.now(),
    )
    do_save_server_record(rec, Path(cache_dir) / "servers")
    location = server.flight_url.to_location()
    logger.info(f"Serving expression '{expr_path.stem}' on {location}")
    server.serve(block=True)


@tracer.start_as_current_span("cli.init_command")
def init_command(
    path="./xorq-template",
    template=InitTemplates.default,
):
    from xorq.common.utils.download_utils import download_unpacked_xorq_template

    path = download_unpacked_xorq_template(path, template)
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
    ls_parser = subparsers.add_parser("ls", help="List catalog entries")
    ls_parser.set_defaults(subcommand="ls")

    uv_build_parser = subparsers.add_parser(
        "uv-build",
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
        "uv-run",
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
        "-f",
        "--format",
        choices=["csv", "json", "parquet"],
        default="parquet",
        help="Output format (default: parquet)",
    )

    serve_unbound_parser = subparsers.add_parser(
        "serve-unbound", help="Serve an an unbound expr via Flight Server"
    )
    serve_unbound_parser.add_argument(
        "build_path",
        help="Build target: alias, entry_id, build_id, or path to build dir",
    )
    serve_unbound_parser.add_argument(
        "to_unbind_hash", help="hash of the expr to replace"
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
        "serve", help="Serve a build via Flight Server"
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
    # Top-level lineage command
    lineage_parser = subparsers.add_parser(
        "lineage",
        help="Print lineage trees of all columns for a build",
    )
    lineage_parser.add_argument(
        "target",
        help="Build target: alias, entry_id, build_id, or path to build dir",
    )
    # Cache a built expression to Parquet via CachedNode
    cache_parser = subparsers.add_parser(
        "cache", help="Cache a build's expression output to Parquet"
    )
    cache_parser.add_argument(
        "target",
        help="Build target: alias, entry_id, build_id, or path to build dir",
    )
    cache_parser.add_argument(
        "--cache-dir",
        required=False,
        default=get_xorq_cache_dir(),
        help="Directory to store parquet cache files",
    )
    # List running servers
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
    # Connection profile commands
    profile_parser = subparsers.add_parser("profile", help="Manage connection profiles")
    profile_subparsers = profile_parser.add_subparsers(
        dest="subcommand", help="Profile commands"
    )
    profile_subparsers.required = True
    # Add profile
    profile_add = profile_subparsers.add_parser("add", help="Add a connection profile")
    profile_add.add_argument("alias", help="Profile alias name")
    profile_add.add_argument(
        "--con-name",
        required=True,
        help="Connection backend name (e.g. 'postgres', 'duckdb')",
    )
    profile_add.add_argument(
        "-p",
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Connection parameter KEY=VALUE",
    )
    # Catalog commands
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
    # List catalog entries
    catalog_subparsers.add_parser("ls", help="List catalog entries")

    catalog_inspect = catalog_subparsers.add_parser(
        "inspect",
        help="Inspect a catalog entry",
        description="Inspect build catalog entries with optional detail sections",
    )
    catalog_inspect.add_argument(
        "entry", help="Entry ID, alias, or entry@revision to inspect"
    )
    catalog_inspect.add_argument(
        "-r",
        "--revision",
        help="Revision ID to inspect (overrides alias)",
        default=None,
    )
    catalog_inspect.add_argument(
        "--schema", action="store_true", help="Show schema section"
    )
    catalog_inspect.add_argument(
        "--plan", action="store_true", help="Show plan section"
    )
    catalog_inspect.add_argument(
        "--profiles", action="store_true", help="Show profiles section"
    )
    catalog_inspect.add_argument(
        "--hashes", dest="hashes", action="store_true", help="Show node hashes section"
    )
    catalog_inspect.add_argument(
        "--full", action="store_true", help="Show all available sections"
    )
    # Deprecated flags (ignored)
    catalog_inspect.add_argument(
        "--pretty", dest="pretty", action="store_true", help=argparse.SUPPRESS
    )
    catalog_inspect.add_argument(
        "--no-pretty", dest="pretty", action="store_false", help=argparse.SUPPRESS
    )
    # Deprecated/removed: --caches is now redundant and no longer supported
    # Note: JSON/YAML output, raw names, color toggles, and --print-nodes removed
    catalog_inspect.set_defaults(
        full=False, schema=False, plan=False, profiles=False, hashes=False, pretty=None
    )
    # diff-builds: compare two build artifacts via git diff --no-index
    # diff-builds: compare two build artifacts via git diff --no-index
    catalog_diff_builds = catalog_subparsers.add_parser(
        "diff-builds", help="Compare two build artifacts via git diff --no-index"
    )
    # Show top-level catalog information
    catalog_subparsers.add_parser("info", help="Show catalog information")
    # Remove an entry or alias from the catalog
    catalog_rm = catalog_subparsers.add_parser(
        "rm", help="Remove a build entry or alias from the catalog"
    )
    catalog_rm.add_argument("entry", help="Entry ID or alias to remove")
    # Export catalog and builds to a directory
    catalog_export = catalog_subparsers.add_parser(
        "export", help="Export catalog and all builds to a target directory"
    )
    catalog_export.add_argument(
        "output_path", help="Directory path to export catalog.yaml and builds subfolder"
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
                    (args.script_path, args.expr_name, args.builds_dir, args.cache_dir),
                )
            case "run":
                f, f_args = (
                    run_command,
                    (args.build_path, args.output_path, args.format, args.cache_dir),
                )
            case "serve-unbound":
                f, f_args = (
                    unbind_and_serve_command,
                    (
                        args.build_path,
                        args.to_unbind_hash,
                        args.host,
                        args.port,
                        args.prometheus_port,
                        args.cache_dir,
                        args.typ,
                    ),
                )
            case "serve":
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
                    (args.path, args.template),
                )
            case "lineage":
                f, f_args = (
                    lineage_command,
                    (args.target,),
                )
            case "cache":
                f, f_args = (
                    cache_command,
                    (args,),
                )
            case "profile":
                f, f_args = (
                    profile_command,
                    (args,),
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
