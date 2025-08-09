import argparse
from typing import List
import os
import pdb
import sys
import traceback
from functools import partial
from pathlib import Path

from opentelemetry import trace

import xorq as xo
# Ensure custom Dask normalization handlers are registered
import xorq.common.utils.dask_normalize  # noqa: F401
import xorq.common.utils.pickle_utils  # noqa: F401
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
import yaml
import uuid
from datetime import datetime, timezone
from xorq.catalog import _dict_load_catalog as load_catalog, _dict_save_catalog as save_catalog, DEFAULT_CATALOG_PATH
from xorq.ibis_yaml.packager import (
    SdistBuilder,
    SdistRunner,
)
import subprocess
# Helper functions for diff-builds subcommand
from xorq.catalog import _dict_resolve_build_dir as resolve_build_dir

def maybe_resolve_build_dirs(left: str, right: str, catalog) -> tuple[Path, Path] | None:
    try:
        left_dir = resolve_build_dir(left, catalog)
        right_dir = resolve_build_dir(right, catalog)
    except ValueError as e:
        print(f"Error: {e}")
        return None
    # Ensure resolution succeeded
    if left_dir is None:
        print(f"Build target not found: {left}")
        return None
    if right_dir is None:
        print(f"Build target not found: {right}")
        return None
    # Ensure paths exist and are directories
    if not left_dir.exists() or not left_dir.is_dir():
        print(f"Build directory not found: {left_dir}")
        return None
    if not right_dir.exists() or not right_dir.is_dir():
        print(f"Build directory not found: {right_dir}")
        return None
    return left_dir, right_dir

def get_diff_file_list(left_dir: Path, right_dir: Path, files: list[str] | None, all_flag: bool) -> tuple[str, ...]:
    default = ("expr.yaml",)
    if files is not None:
        return tuple(files)
    if all_flag:
        # Exclude node_hashes.yaml as node hashes are not used here
        default_files = ("expr.yaml", "deferred_reads.yaml", "profiles.yaml", "sql.yaml", "metadata.json")
        sqls = {p.relative_to(left_dir).as_posix() for p in left_dir.rglob("*.sql")} | {p.relative_to(right_dir).as_posix() for p in right_dir.rglob("*.sql")}
        return default_files + tuple(sorted(sqls))
    return default

def get_keep_files(file_list: tuple[str, ...], left_dir: Path, right_dir: Path) -> tuple[str, ...]:
    return tuple(f for f in file_list if (left_dir / f).exists() or (right_dir / f).exists())

def run_diffs(left_dir: Path, right_dir: Path, keep_files: tuple[str, ...]) -> int:
    exit_code = 0
    for f in keep_files:
        print(f"## Diff: {f}")
        lf = left_dir / f
        rf = right_dir / f
        ret = subprocess.call(["git", "diff", "--no-index", "--", str(lf), str(rf)])
        if ret == 1:
            exit_code = 1
        elif ret != 0:
            return ret
    return exit_code

def do_diff_builds(
    left: str,
    right: str,
    files: list[str] | None,
    all_flag: bool,
) -> int:
    catalog = load_catalog()
    resolved = maybe_resolve_build_dirs(left, right, catalog)
    if not resolved:
        return 2
    left_dir, right_dir = resolved
    file_list = get_diff_file_list(left_dir, right_dir, files, all_flag)
    keep_files = get_keep_files(file_list, left_dir, right_dir)
    if not keep_files:
        print("No files to diff")
        return 2
    return run_diffs(left_dir, right_dir, keep_files)

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
            raise ValueError
        elif len(found_cons) == 1:
            (found_con,) = found_cons
        else:
            (found_con,) = find_all_sources(found)

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

@tracer.start_as_current_span("cli.lineage_command")
def lineage_command(
    target: str,
):
    """
    Print per-column lineage trees for a single build.
    """
    catalog = load_catalog()
    build_dir = resolve_build_dir(target, catalog)
    if build_dir is None or not build_dir.exists() or not build_dir.is_dir():
        print(f"Build target not found: {target}")
        sys.exit(2)
    # Load serialized expression
    expr = load_expr(build_dir)
    # Build and print lineage trees
    from xorq.common.utils.lineage_utils import build_column_trees, print_tree

    trees = build_column_trees(expr)
    for column, tree in trees.items():
        print(f"Lineage for column '{column}':")
        print_tree(tree)
        print()

@tracer.start_as_current_span("cli.catalog_command")
def catalog_command(args):
    """
    Manage build catalog subcommands: add, ls, inspect.
    """
    if args.subcommand == "add":
        # Use absolute path for build directory
        build_path = Path(args.build_path).resolve()
        alias = args.alias
        # Validate build and extract metadata
        build_id, expr_hashes, meta_digest, metadata_preview = BuildManager.validate_build(build_path)
        build_path_str = str(build_path)
        catalog = load_catalog()
        now = datetime.now(timezone.utc).isoformat()
        # If alias exists, append a new revision to that entry
        if alias and alias in (catalog.get("aliases") or {}):
            mapping = catalog["aliases"][alias]
            entry_id = mapping["entry_id"]
            entry = next(e for e in catalog.get("entries", []) if e.get("entry_id") == entry_id)
            # Determine next revision number
            existing = [r.get("revision_id", "r0") for r in entry.get("history", [])]
            nums = [int(r[1:]) for r in existing if r.startswith("r")]
            next_num = max(nums, default=0) + 1
            revision_id = f"r{next_num}"
            revision = {
                "revision_id": revision_id,
                "created_at": now,
                "build": {"build_id": build_id, "path": build_path_str},
                "expr_hashes": expr_hashes,
                "meta_digest": meta_digest,
            }
            if metadata_preview:
                revision["metadata"] = metadata_preview
            # Update entry
            entry.setdefault("history", []).append(revision)
            entry["current_revision"] = revision_id
            # Update alias mapping timestamp
            mapping["revision_id"] = revision_id
            mapping["updated_at"] = now
        else:
            # New entry and optional alias
            entry_id = str(uuid.uuid4())
            revision_id = "r1"
            revision = {
                "revision_id": revision_id,
                "created_at": now,
                "build": {"build_id": build_id, "path": build_path_str},
                "expr_hashes": expr_hashes,
                "meta_digest": meta_digest,
            }
            if metadata_preview:
                revision["metadata"] = metadata_preview
            entry = {
                "entry_id": entry_id,
                "created_at": now,
                "current_revision": revision_id,
                "history": [revision],
            }
            catalog.setdefault("entries", []).append(entry)
            if alias:
                catalog.setdefault("aliases", {})[alias] = {
                    "entry_id": entry_id,
                    "revision_id": revision_id,
                    "updated_at": now,
                }
        save_catalog(catalog)
        print(f"Added build {build_id} as entry {entry_id} revision {revision_id}")

    elif args.subcommand == "ls":
        catalog = load_catalog()
        aliases = catalog.get("aliases", {})
        if aliases:
            print("Aliases:")
            for al, mapping in aliases.items():
                print(f"{al}\t{mapping['entry_id']}\t{mapping['revision_id']}")
        print("Entries:")
        for entry in catalog.get("entries", []):
            ent_id = entry.get("entry_id")
            curr_rev = entry.get("current_revision")
            build_id = None
            for rev in entry.get("history", []):
                if rev.get("revision_id") == curr_rev:
                    build_id = rev.get("build", {}).get("build_id")
                    break
            print(f"{ent_id}\t{curr_rev}\t{build_id}")

    elif args.subcommand == "inspect":
        catalog = load_catalog()
        # Resolve alias or entry@revision or entry
        from xorq.catalog import resolve_target
        target = resolve_target(args.entry, catalog)
        if target is None:
            print(f"Entry {args.entry} not found in catalog")
            return
        entry_id = target.entry_id
        # Use explicit --revision or target.rev (from alias@rev or entry@rev)
        revision_id = args.revision or target.rev
        # Find entry
        entry = next((e for e in catalog.get("entries", []) if e.get("entry_id") == entry_id), None)
        if entry is None:
            print(f"Entry {entry_id} not found in catalog")
            return
        # Determine revision
        if not revision_id:
            revision_id = entry.get("current_revision")
        revision = next((r for r in entry.get("history", []) if r.get("revision_id") == revision_id), None)
        if revision is None:
            print(f"Revision {revision_id} not found for entry {entry_id}")
            return
        # Prepare output data
        output = {
            "entry_id": entry_id,
            "revision_id": revision_id,
            "entry": entry,
            "revision": revision,
        }
        # JSON/YAML output
        if args.as_json:
            import json
            print(json.dumps(output, indent=2, default=str))
            return
        if args.as_yaml:
            print(yaml.safe_dump(output, sort_keys=False))
            return
        # Print summary info and catalog entry details
        print("Summary:")
        print(f"  Entry ID: {entry_id}")
        entry_created = entry.get("created_at")
        if entry_created:
            print(f"  Entry Created: {entry_created}")
        print(f"  Revision ID: {revision_id}")
        revision_created = revision.get("created_at")
        if revision_created:
            print(f"  Revision Created: {revision_created}")
        expr_hash = (revision.get("expr_hashes") or {}).get("expr") or revision.get("build", {}).get("build_id")
        print(f"  Expr Hash: {expr_hash}")
        meta_digest = revision.get("meta_digest")
        if meta_digest:
            print(f"  Meta Digest: {meta_digest}")
        # Sections
        build_path = revision.get("build", {}).get("path")
        expr = None
        schema = None
        # Load expression and schema if needed
        if build_path and (args.full or args.plan or args.schema or args.profiles or args.caches):
            from xorq.ibis_yaml.compiler import load_expr
            try:
                expr = load_expr(Path(build_path))
                schema = expr.schema()
            except Exception as e:
                print(f"Error loading expression or schema: {e}")
        # Plan section
        if args.full or args.plan:
            print("\nPlan:")
            if expr is not None:
                # Print the Ibis expression representation
                print(expr)
            else:
                print("  No plan available.")
        # Schema section
        if args.full or args.schema:
            print("\nSchema:")
            if schema:
                for name, dtype in schema.items():
                    print(f"  {name}: {dtype}")
            else:
                print("  No schema available.")
        # Profiles section
        if args.full or args.profiles:
            profiles_file = Path(build_path) / "profiles.yaml" if build_path else None
            print("\nProfiles:")
            if profiles_file and profiles_file.exists():
                try:
                    text = profiles_file.read_text()
                    data = yaml.safe_load(text) or {}
                    for name, profile in data.items():
                        print(f"  {name}: {profile}")
                except Exception as e:
                    print(f"  Error loading profiles: {e}")
            else:
                print("  No profiles available.")
        # Caches section
        if args.full or args.caches:
            print("\nCaches:")
            if expr is not None:
                from xorq.common.utils.graph_utils import walk_nodes
                from xorq.expr.relations import CachedNode
                import dask
                try:
                    nodes = walk_nodes((CachedNode,), expr)
                    if nodes:
                        for node in nodes:
                            h = dask.base.tokenize(node.to_expr())
                            print(f"  {h}: {node}")
                    else:
                        print("  No caches recorded.")
                except Exception as e:
                    print(f"  Error computing caches: {e}")
            else:
                print("  No caches available.")
        return
    elif args.subcommand == "diff-builds":
        # Compare two build artifacts via git diff --no-index
        code = do_diff_builds(args.left, args.right, args.files, args.all)
        sys.exit(code)
    else:
        print(f"Unknown catalog subcommand: {args.subcommand}")



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
        "build_path", help="Path to the build directory (output of xorq build)"
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
        "serve-flight-udxf", help="Serve a build via Flight Server"
    )
    serve_parser.add_argument(
        "build_path", help="Path to the build directory (output of xorq build)"
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
    # Catalog commands
    catalog_parser = subparsers.add_parser("catalog", help="Manage build catalog")
    catalog_subparsers = catalog_parser.add_subparsers(dest="subcommand", help="Catalog commands")
    catalog_subparsers.required = True

    catalog_add = catalog_subparsers.add_parser("add", help="Add a build to the catalog")
    catalog_add.add_argument("build_path", help="Path to the build directory")
    catalog_add.add_argument("-a", "--alias", help="Optional alias for this entry", default=None)

    catalog_ls = catalog_subparsers.add_parser("ls", help="List catalog entries")

    catalog_inspect = catalog_subparsers.add_parser(
        "inspect", help="Inspect a catalog entry",
        description="Inspect build catalog entries with optional detail sections",
    )
    catalog_inspect.add_argument(
        "entry", help="Entry ID, alias, or entry@revision to inspect"
    )
    catalog_inspect.add_argument(
        "-r", "--revision", help="Revision ID to inspect (overrides alias)", default=None
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
        "--caches", action="store_true", help="Show caches section"
    )
    catalog_inspect.add_argument(
        "--full", action="store_true", help="Show all available sections"
    )
    catalog_inspect.add_argument(
        "--json", dest="as_json", action="store_true", help="Output result as JSON"
    )
    catalog_inspect.add_argument(
        "--yaml", dest="as_yaml", action="store_true", help="Output result as YAML"
    )
    catalog_inspect.add_argument(
        "--no-color", action="store_true", help="Disable colorized output"
    )
    catalog_inspect.add_argument(
        "--raw-names", action="store_true", help="Display full object names without truncation"
    )
    # Deprecated flags (ignored)
    catalog_inspect.add_argument(
        "--pretty", dest="pretty", action="store_true", help=argparse.SUPPRESS
    )
    catalog_inspect.add_argument(
        "--no-pretty", dest="pretty", action="store_false", help=argparse.SUPPRESS
    )
    catalog_inspect.set_defaults(
        as_json=False, as_yaml=False, full=False, schema=False,
        plan=False, profiles=False, caches=False, pretty=None
    )
    # diff-builds: compare two build artifacts via git diff --no-index
    # diff-builds (alias: diff): compare two build artifacts via git diff --no-index
    catalog_diff_builds = catalog_subparsers.add_parser(
        "diff-builds", aliases=["diff"], help="Compare two build artifacts via git diff --no-index"
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
                    (args.path, args.template),
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
