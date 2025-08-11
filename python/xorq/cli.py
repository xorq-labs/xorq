import argparse
import json
import os
import pdb
import shutil

# JSON handling
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import yaml
from opentelemetry import trace

import xorq as xo

# Ensure custom Dask normalization handlers are registered
import xorq.common.utils.dask_normalize  # noqa: F401
import xorq.common.utils.pickle_utils  # noqa: F401

# Helper functions for diff-builds subcommand
from xorq.catalog import (
    get_catalog_path,
    load_catalog,
    resolve_build_dir,
    save_catalog,
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


def maybe_resolve_build_dirs(
    left: str, right: str, catalog
) -> tuple[Path, Path] | None:
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


# === Server Recording Utilities ===
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ServerRecord:
    pid: int
    command: str
    target: str
    port: Optional[int]
    start_time: datetime
    node_hash: Optional[str] = None


def maybe_make_server_record(data: dict) -> Optional[ServerRecord]:
    try:
        pid = data["pid"]
        command = data["command"]
        target = data.get("target", "")
        port = data.get("port")
        node_hash = data.get("to_unbind_hash")
        start_time = datetime.fromisoformat(data["start_time"])
        return ServerRecord(
            pid=pid,
            command=command,
            target=target,
            port=port,
            start_time=start_time,
            node_hash=node_hash,
        )
    except Exception:
        return None


def get_server_records(record_dir: Path) -> Tuple[ServerRecord, ...]:
    if not record_dir.exists():
        return ()
    records: list[ServerRecord] = []
    for f in record_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        rec = maybe_make_server_record(data)
        if rec is not None:
            records.append(rec)
    return tuple(records)


def filter_running(records: Tuple[ServerRecord, ...]) -> Tuple[ServerRecord, ...]:
    running: list[ServerRecord] = []
    for rec in records:
        try:
            os.kill(rec.pid, 0)
            running.append(rec)
        except Exception:
            continue
    return tuple(running)


def format_server_table(
    records: Tuple[ServerRecord, ...],
) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
    headers = ("TARGET", "STATE", "COMMAND", "HASH", "PID", "PORT", "UPTIME")
    rows: list[tuple[str, ...]] = []
    now = datetime.now()
    for rec in records:
        state = "running"
        try:
            delta = now - rec.start_time
            hours, rem = divmod(int(delta.total_seconds()), 3600)
            minutes, _ = divmod(rem, 60)
            uptime = f"{hours}h{minutes}m"
        except Exception:
            uptime = ""
        rows.append(
            (
                rec.target,
                state,
                rec.command,
                rec.node_hash or "",
                str(rec.pid),
                str(rec.port) if rec.port is not None else "",
                uptime,
            )
        )
    return headers, tuple(rows)


def do_print_table(headers: Tuple[str, ...], rows: Tuple[Tuple[str, ...], ...]) -> None:
    if rows:
        widths = [max(len(cell) for cell in col) for col in zip(headers, *rows)]
    else:
        widths = [len(h) for h in headers]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    for row in rows:
        print(fmt.format(*row))


def make_server_record(
    pid: int,
    command: str,
    target: str,
    port: Optional[int] = None,
    start_time: Optional[datetime] = None,
    node_hash: Optional[str] = None,
) -> ServerRecord:
    """Factory: create a ServerRecord with optional start_time."""
    ts = start_time or datetime.now()
    return ServerRecord(
        pid=pid,
        command=command,
        target=target,
        port=port,
        start_time=ts,
        node_hash=node_hash,
    )


def do_save_server_record(record: ServerRecord, record_dir: Path) -> None:
    """Side effect: save a ServerRecord to JSON file in record_dir."""
    record_dir.mkdir(parents=True, exist_ok=True)
    path = record_dir / f"{record.pid}.json"
    data: dict = {
        "pid": record.pid,
        "command": record.command,
        "target": record.target,
        "port": record.port,
        "start_time": record.start_time.isoformat(),
    }
    if record.node_hash is not None:
        data["to_unbind_hash"] = record.node_hash
    path.write_text(json.dumps(data))


def get_diff_file_list(
    left_dir: Path, right_dir: Path, files: list[str] | None, all_flag: bool
) -> tuple[str, ...]:
    default = ("expr.yaml",)
    if files is not None:
        return tuple(files)
    if all_flag:
        # Exclude node_hashes.yaml as node hashes are not used here
        default_files = (
            "expr.yaml",
            "deferred_reads.yaml",
            "profiles.yaml",
            "sql.yaml",
            "metadata.json",
        )
        sqls = {p.relative_to(left_dir).as_posix() for p in left_dir.rglob("*.sql")} | {
            p.relative_to(right_dir).as_posix() for p in right_dir.rglob("*.sql")
        }
        return default_files + tuple(sorted(sqls))
    return default


def get_keep_files(
    file_list: tuple[str, ...], left_dir: Path, right_dir: Path
) -> tuple[str, ...]:
    return tuple(
        f for f in file_list if (left_dir / f).exists() or (right_dir / f).exists()
    )


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
    con_index=None,
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
    catalog = load_catalog()
    build_dir = resolve_build_dir(expr_path, catalog)
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
            found_con = found_cons[0]
        else:
            subtree_cons = find_all_sources(found)
            if con_index is not None:
                if con_index < 0 or con_index >= len(subtree_cons):
                    raise ValueError(
                        f"Invalid --con-index: {con_index}. Must be between 0 and {len(subtree_cons) - 1}"
                    )
                found_con = subtree_cons[con_index]
            elif len(subtree_cons) == 1:
                found_con = subtree_cons[0]
            else:
                raise ValueError(
                    f"Multiple sources found for expr hash {to_unbind_hash}: "
                    + ", ".join(f"[{i}]: {src}" for i, src in enumerate(subtree_cons))
                    + ". Please specify --con-index to select one."
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

<<<<<<< HEAD
@tracer.start_as_current_span("cli.lineage_command")
>>>>>>> 0ebcec7 (feat: add cache command and ref)
=======

>>>>>>> ec9f7ed (feat: save builds in catalog-builds folder where catalog.yaml lives)
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


def catalog_command(args):
    """
    Manage build catalog subcommands: add, ls, inspect.
    """
    # Determine canonical catalog path in config directory
    config_path = get_catalog_path()
    config_dir = config_path.parent
    if args.subcommand == "add":
        # Use absolute path for build directory
        build_path = Path(args.build_path).resolve()
        alias = args.alias
        # Validate build and extract metadata (expr hash recalculated fresh later)
        build_id, meta_digest, metadata_preview = BuildManager.validate_build(
            build_path
        )
        # Ensure local catalog directory and builds subdirectory exist
        config_dir.mkdir(parents=True, exist_ok=True)
        # Store all builds under a canonical folder
        builds_dir = config_dir / "catalog-builds"
        builds_dir.mkdir(parents=True, exist_ok=True)
        target_dir = builds_dir / build_id
        # Copy into a temporary directory then atomically rename to avoid partial state
        temp_dir = builds_dir / f".{build_id}.tmp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        try:
            shutil.copytree(build_path, temp_dir)
            # Remove old target if exists, then atomically replace
            if target_dir.exists():
                shutil.rmtree(target_dir)
            os.replace(str(temp_dir), str(target_dir))
        except Exception:
            # Clean up temp on failure
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        # Store relative path to build artifacts
        build_path_str = str(Path("catalog-builds") / build_id)
        # Load existing catalog (empty if not exists)
        catalog = load_catalog(path=config_path)
        now = datetime.now(timezone.utc).isoformat()
        # If alias exists, append a new revision to that entry
        if alias and alias in (catalog.get("aliases") or {}):
            mapping = catalog["aliases"][alias]
            entry_id = mapping["entry_id"]
            entry = next(
                e for e in catalog.get("entries", []) if e.get("entry_id") == entry_id
            )
            # Determine next revision number
            existing = [r.get("revision_id", "r0") for r in entry.get("history", [])]
            nums = [int(r[1:]) for r in existing if r.startswith("r")]
            next_num = max(nums, default=0) + 1
            revision_id = f"r{next_num}"
            revision = {
                "revision_id": revision_id,
                "created_at": now,
                "build": {"build_id": build_id, "path": build_path_str},
                # expr_hash recalculated fresh when inspecting
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
                # expr_hash recalculated fresh when inspecting
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
        # Save updated catalog to local catalog file
        save_catalog(catalog, path=config_path)
        print(f"Added build {build_id} as entry {entry_id} revision {revision_id}")

    elif args.subcommand == "ls":
        # Load catalog from local catalog file
        catalog = load_catalog(path=config_path)
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
        # Load catalog from local catalog file
        catalog = load_catalog(path=config_path)
        from xorq.catalog import resolve_target

        target = resolve_target(args.entry, catalog)
        if target is None:
            print(f"Entry {args.entry} not found in catalog")
            return
        entry_id = target.entry_id
        revision_id = args.revision or target.rev
        # Find entry
        entry = next(
            (e for e in catalog.get("entries", []) if e.get("entry_id") == entry_id),
            None,
        )
        if entry is None:
            print(f"Entry {entry_id} not found in catalog")
            return
        # Determine revision
        if not revision_id:
            revision_id = entry.get("current_revision")
        revision = next(
            (
                r
                for r in entry.get("history", [])
                if r.get("revision_id") == revision_id
            ),
            None,
        )
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
        # Only show summary when not focusing on specific sections
        if args.full or not (args.plan or args.profiles or args.hashes):
            print("Summary:")
            print(f"  {'Entry ID':<13}: {entry_id}")
            entry_created = entry.get("created_at")
            if entry_created:
                print(f"  Entry Created: {entry_created}")
            print(f"  {'Revision ID':<13}: {revision_id}")
            revision_created = revision.get("created_at")
            if revision_created:
                print(f"  Revision Created: {revision_created}")
            expr_hash = (revision.get("expr_hashes") or {}).get("expr") or revision.get(
                "build", {}
            ).get("build_id")
            print(f"  {'Expr Hash':<13}: {expr_hash}")
            meta_digest = revision.get("meta_digest")
            if meta_digest:
                print(f"  {'Meta Digest':<13}: {meta_digest}")
        # Resolve build directory path (handle relative paths)
        bp = revision.get("build", {}).get("path")
        build_dir = Path(bp) if bp else None
        if build_dir and not build_dir.is_absolute():
            build_dir = config_dir / build_dir
        expr = None
        schema = None
        if build_dir and (
            args.full or args.plan or args.schema or args.profiles or args.hashes
        ):
            from xorq.ibis_yaml.compiler import load_expr

            try:
                expr = load_expr(build_dir)
                schema = expr.schema()
            except Exception as e:
                print(f"Error loading expression for DAG: {e}")
        if args.full or args.plan:
            print("\nPlan:")
            if expr is not None:
                print(expr)
            else:
                print("  No plan available.")
        if args.full or args.schema:
            print("\nSchema:")
            if schema:
                for name, dtype in schema.items():
                    print(f"  {name}: {dtype}")
            else:
                print("  No schema available.")
        if args.full or args.profiles:
            # Load profiles from build directory
            profiles_file = build_dir / "profiles.yaml" if build_dir else None
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
        if args.full or args.hashes:
            print("\nNode hashes:")
            # Dynamically compute hashes for CachedNode instances
            if expr is not None:
                try:
                    import dask

                    from xorq.common.utils.graph_utils import walk_nodes
                    from xorq.expr.relations import CachedNode

                    nodes = walk_nodes((CachedNode,), expr)
                    if nodes:
                        for node in nodes:
                            h = dask.base.tokenize(node.to_expr())
                            print(f"  {h}")
                    else:
                        print("  No node hashes recorded.")
                except Exception as e:
                    print(f"  Error computing node hashes: {e}")
            else:
                print("  No node hashes recorded.")
        return
    elif args.subcommand == "info":
        # Show top-level catalog info
        # Load catalog from local catalog file
        catalog = load_catalog(path=config_path)
        entries = catalog.get("entries", []) or []
        aliases = catalog.get("aliases", {}) or {}
        print(f"Catalog path: {config_path}")
        print(f"Entries: {len(entries)}")
        print(f"Aliases: {len(aliases)}")
        return
    elif args.subcommand == "rm":
        # Remove an entry or alias from the catalog
        # Load catalog from local catalog file
        catalog = load_catalog(path=config_path)
        token = args.entry
        # Remove alias if present
        aliases = catalog.get("aliases", {}) or {}
        if token in aliases:
            aliases.pop(token, None)
            # If no aliases remain, clean up key
            if not aliases:
                catalog.pop("aliases", None)
            # Save updated catalog
            save_catalog(catalog, path=config_path)
            print(f"Removed alias {token}")
            return
        # Remove entry if present
        entries = catalog.get("entries", [])
        for i, entry in enumerate(entries):
            if entry.get("entry_id") == token:
                # Remove entry and any related aliases
                entries.pop(i)
                # Clean aliases pointing to this entry
                to_remove = [
                    a for a, m in aliases.items() if m.get("entry_id") == token
                ]
                for a in to_remove:
                    aliases.pop(a, None)
                # Clean empty aliases dict
                if not aliases:
                    catalog.pop("aliases", None)
                # Save updated catalog
                save_catalog(catalog, path=config_path)
                print(f"Removed entry {token}")
                return
        # Not found
        print(f"Entry {token} not found in catalog")
        return
    elif args.subcommand == "export":
        # Export catalog.yaml and all builds to a target directory
        export_dir = Path(args.output_path)
        # Ensure export directory exists
        if export_dir.exists() and not export_dir.is_dir():
            print(f"Export path exists and is not a directory: {export_dir}")
            return
        export_dir.mkdir(parents=True, exist_ok=True)
        # Copy catalog file
        if config_path.exists():
            shutil.copy2(config_path, export_dir / config_path.name)
        else:
            print(f"No catalog found at {config_path}")
            return
        # Copy builds directory
        src_builds = config_dir / "catalog-builds"
        if src_builds.exists() and src_builds.is_dir():
            dest_builds = export_dir / src_builds.name
            if dest_builds.exists():
                shutil.rmtree(dest_builds)
            shutil.copytree(src_builds, dest_builds)
        print(f"Exported catalog and builds to {export_dir}")
        return
    elif args.subcommand == "diff-builds":
        # Compare two build artifacts via git diff --no-index
        code = do_diff_builds(args.left, args.right, args.files, args.all)
        sys.exit(code)
    else:
        print(f"Unknown catalog subcommand: {args.subcommand}")


def ps_command(cache_dir: str) -> None:
    """List active xorq servers."""
    record_dir = Path(cache_dir) / "servers"
    headers, rows = format_server_table(filter_running(get_server_records(record_dir)))
    do_print_table(headers, rows)


def hash_command(target: str):
    """
    List replaceable nodes (Read, CachedNode, PhysicalTable) in a build and their dask hashes.
    """
    import dask

    # Resolve build identifier via catalog
    catalog = load_catalog()
    build_dir = resolve_build_dir(target, catalog)
    if build_dir is None or not build_dir.exists() or not build_dir.is_dir():
        print(f"Build target not found: {target}")
        sys.exit(2)
    # Load expression
    expr = load_expr(build_dir)
    # FIXME: there is an issue with dask.base.tokenize and RemoteTable node that takes forever to tokenize
    # so for now we are only lisitng CachedNode and Read nodes
    # from xorq.common.utils.node_utils import replace_typs
    from xorq.common.utils.graph_utils import walk_nodes

    replace_typs = (xo.expr.relations.CachedNode, xo.expr.relations.Read)
    nodes = walk_nodes(replace_typs, expr)
    if not nodes:
        print("No replaceable nodes found.")
        return
    # Print each node's dask hash, class name, and representation
    for node in nodes:
        tok = dask.base.tokenize(node.to_expr())
        print(f"{tok} {type(node).__name__} {node}")


def cache_command(args):
    """
    Cache a built expression output to Parquet using a CachedNode.
    """
    from xorq.caching import ParquetStorage
    from xorq.common.utils.caching_utils import find_backend

    # Resolve build target
    catalog = load_catalog()
    build_dir = resolve_build_dir(args.target, catalog)
    if build_dir is None or not build_dir.exists() or not build_dir.is_dir():
        print(f"Build target not found: {args.target}")
        sys.exit(2)
    # Load expression
    expr = load_expr(build_dir)
    # Determine backend for caching
    con, _ = find_backend(expr.op(), use_default=True)
    # Setup Parquet storage at given cache directory
    base_path = Path(args.cache_dir)
    storage = ParquetStorage(source=con, relative_path=Path("."), base_path=base_path)
    # Attach cache node
    cached_expr = expr.cache(storage=storage)
    # Execute to materialize cache
    try:
        for _ in cached_expr.to_pyarrow_batches():
            pass
    except Exception as e:
        print(f"Error during caching execution: {e}")
        sys.exit(1)
    # Report cache files
    cache_path = storage.cache.storage.path
    print(f"Cache written to: {cache_path}")
    for pq_file in sorted(cache_path.rglob("*.parquet")):
        print(f"  {pq_file.relative_to(cache_path)}")


def profile_command(args):
    """
    Manage connection profiles: add new profiles.
    """
    sub = args.subcommand
    if sub == "add":
        alias = args.alias
        con_name = args.con_name
        params = {}
        for p in args.param:
            if "=" not in p:
                print(f"Invalid parameter '{p}', expected KEY=VALUE")
                sys.exit(1)
            k, v = p.split("=", 1)
            params[k] = v
        # Create and save profile
        from xorq.vendor.ibis.backends.profiles import Profile

        prof = Profile(con_name=con_name, kwargs_tuple=tuple(params.items()))
        try:
            path = prof.save(alias=alias, clobber=False)
            print(f"Profile '{alias}' saved to {path}")
        except ValueError as e:
            print(f"Error saving profile: {e}")
            sys.exit(1)
    else:
        print(f"Unknown profile subcommand: {sub}")
        sys.exit(2)


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
    serve_unbound_parser.add_argument(
        "--con-index",
        type=int,
        default=None,
        help="index of the source connection to use for unbinding when multiple sources exist",
    )
<<<<<<< HEAD
=======
    serve_unbound_parser.add_argument(
        "--prometheus-port",
        type=int,
        default=None,
        help="Port to expose Prometheus metrics (default: disabled)",
    )

    # Serve a built expression via Flight Server (UDXF)
>>>>>>> b11f2ee (add prometheus port to serve-unbound)
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
    catalog_ls = catalog_subparsers.add_parser("ls", help="List catalog entries")

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
    catalog_info = catalog_subparsers.add_parser(
        "info", help="Show catalog information"
    )
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
        "output_path",
        help="Directory path to export catalog.yaml and builds subfolder"
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
                        args.con_index,
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
