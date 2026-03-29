import re
from datetime import datetime
from functools import cache, cached_property
from pathlib import Path

from attr import field, frozen
from attr.validators import instance_of, optional

from xorq.catalog.catalog import CatalogEntry


DEFAULT_REFRESH_INTERVAL = 10

COLUMNS = ("KIND", "ALIAS", "HASH", "BACKEND", "CACHED")

SCHEMA_PREVIEW_COLUMNS = ("NAME", "TYPE")

REVISION_COLUMNS = ("STATUS", "HASH", "COLUMNS", "CACHED", "DATE")

GIT_LOG_COLUMNS = ("HASH", "DATE", "MESSAGE")

RUN_COLUMNS = ("STATUS", "RUN ID", "CACHE", "DURATION", "FORMAT", "DATE")

CACHE_PANEL_COLUMNS = ("KEY", "ENTRY", "SIZE", "ROWS")

SERVICE_COLUMNS = ("STATUS", "NAME", "ENDPOINT", "STARTED")


def _format_cached(value: bool | None) -> str:
    match value:
        case True:
            return "●"
        case False:
            return "○"
        case _:
            return "—"


@frozen
class CatalogRowData:
    entry: CatalogEntry = field(repr=False)
    aliases: tuple[str, ...] = field(factory=tuple, validator=instance_of(tuple))

    @property
    def cached(self) -> bool | None:
        parquet_cache_paths = self.entry.parquet_cache_paths
        if parquet_cache_paths:
            return all(Path(p).exists() for p in parquet_cache_paths)
        return None

    @property
    def kind(self) -> str:
        return str(self.entry.kind)

    @property
    def hash(self) -> str:
        return self.entry.name

    @property
    def backends(self) -> tuple[str, ...]:
        return self.entry.backends

    @property
    def schema_in(self) -> tuple[tuple[str, str], ...] | None:
        si = self.entry.metadata.schema_in
        return tuple(si.items()) if si is not None else None

    @property
    def schema_out(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.entry.metadata.schema_out.items())

    @cached_property
    def aliases_display(self) -> str:
        return ", ".join(self.aliases) if self.aliases else ""

    @cached_property
    def backends_display(self) -> str:
        return ", ".join(sorted(set(self.backends))) if self.backends else ""

    @property
    def cached_display(self) -> str:
        return _format_cached(self.cached)

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.aliases_display, self.hash)

    @cached_property
    def sqls(self) -> tuple[tuple[str, str, str], ...]:
        """((name, engine, sql), ...) for all queries in the expression plan."""
        return self.entry.metadata.sql_queries

    @cached_property
    def lineage_text(self) -> str:
        chain = self.entry.metadata.lineage
        return " → ".join(chain) if chain else "(empty)"

    @cached_property
    def cache_info_text(self) -> str:
        paths = self.entry.parquet_cache_paths
        match paths:
            case () | None:
                return "— unknown"
            case _ if all(Path(p).exists() for p in paths):
                return f"● cached  {paths[0]}"
            case _:
                return "○ uncached"

    @cached_property
    def info_text(self) -> str:
        parts = [
            f"Lineage: {self.lineage_text}",
            f"Cache: {self.cache_info_text}",
        ]
        return "\n".join(parts)

    @property
    def row_key(self) -> str:
        return self.hash

    @property
    def row(self) -> tuple[str, ...]:
        return (
            self.kind,
            self.aliases_display,
            self.hash,
            self.backends_display,
            self.cached_display,
        )


@frozen
class GitLogRowData:
    hash: str = field(default="", validator=instance_of(str))
    date: str = field(default="", validator=instance_of(str))
    message: str = field(default="", validator=instance_of(str))

    @property
    def row(self) -> tuple[str, ...]:
        return (self.hash, self.date, self.message)


@frozen
class RevisionRowData:
    hash: str = field(default="", validator=instance_of(str))
    column_count: int | None = field(default=None, validator=optional(instance_of(int)))
    cached: bool | None = field(default=None, validator=optional(instance_of(bool)))
    commit_date: str = field(default="", validator=instance_of(str))
    is_current: bool = field(default=False, validator=instance_of(bool))

    @cached_property
    def cached_display(self) -> str:
        return _format_cached(self.cached)

    @cached_property
    def status_display(self) -> str:
        return "CURRENT →" if self.is_current else ""

    @cached_property
    def columns_display(self) -> str:
        match self.column_count:
            case None:
                return "?"
            case int(n):
                return f"{n} cols"
            case _:
                return "?"

    @property
    def row(self) -> tuple[str, ...]:
        return (
            self.status_display,
            self.hash,
            self.columns_display,
            self.cached_display,
            self.commit_date,
        )


def _entry_info(entry) -> tuple[int | None, bool | None]:
    parquet_cache_paths = entry.parquet_cache_paths
    cached = (
        all(Path(p).exists() for p in parquet_cache_paths)
        if parquet_cache_paths
        else None
    )
    return len(entry.columns), cached


@frozen
class RunConfig:
    entry_name: str = field(validator=instance_of(str))
    expr_hash: str = field(validator=instance_of(str))
    cache_type: str = field(default="snapshot", validator=instance_of(str))
    ttl: int | None = field(default=None, validator=optional(instance_of(int)))


CACHE_TYPE_LABELS = {
    "snapshot": "snapshot",
    "source": "source",
    "ttl_snapshot": "ttl",
    "none": "—",
}


def _cache_type_display(cache_type: str, ttl: int | None = None) -> str:
    match cache_type:
        case "ttl_snapshot":
            return f"ttl({ttl}s)" if ttl else "ttl"
        case str(ct) if ct in CACHE_TYPE_LABELS:
            return CACHE_TYPE_LABELS[ct]
        case _:
            return "?"


@frozen
class RunRowData:
    run_id: str = field(default="", validator=instance_of(str))
    status: str = field(default="", validator=instance_of(str))
    cache_type: str = field(default="", validator=instance_of(str))
    duration: str = field(default="", validator=instance_of(str))
    output_format: str = field(default="", validator=instance_of(str))
    date: str = field(default="", validator=instance_of(str))
    error: str | None = field(default=None, validator=optional(instance_of(str)))
    ttl: int | None = field(default=None, validator=optional(instance_of(int)))
    output_snapshot_path: str | None = field(
        default=None, validator=optional(instance_of(str))
    )
    meta: tuple[tuple[str, str], ...] = field(
        factory=tuple, validator=instance_of(tuple)
    )

    @cached_property
    def status_display(self) -> str:
        match self.status:
            case "ok":
                return "OK"
            case "error":
                return "ERR"
            case "running":
                return "..."
            case _:
                return self.status.upper() if self.status else "?"

    @cached_property
    def cache_type_display(self) -> str:
        return _cache_type_display(self.cache_type, self.ttl)

    @cached_property
    def run_id_display(self) -> str:
        return self.run_id[:8] if self.run_id else ""

    @property
    def row(self) -> tuple[str, ...]:
        return (
            self.status_display,
            self.run_id_display,
            self.cache_type_display,
            self.duration,
            self.output_format,
            self.date,
        )


def _compute_duration(started: str, completed: str) -> str:
    match (started, completed):
        case ("", _) | (_, ""):
            return ""
        case _:
            try:
                s = datetime.fromisoformat(started)
                c = datetime.fromisoformat(completed)
                delta = (c - s).total_seconds()
                match delta:
                    case d if d < 1:
                        return f"{d * 1000:.0f}ms"
                    case d if d < 60:
                        return f"{d:.1f}s"
                    case d:
                        return f"{d / 60:.1f}m"
            except (ValueError, TypeError):
                return ""


def _format_run_date(started: str) -> str:
    match started:
        case "":
            return ""
        case _:
            try:
                dt = datetime.fromisoformat(started)
                return dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                return started


def _build_run_rows(
    expr_hash: str, max_count: int = 20, runs_dir: Path | None = None
) -> tuple[RunRowData, ...]:
    from xorq.common.utils.logging_utils import Runs, get_xorq_runs_dir  # noqa: PLC0415

    base = runs_dir if runs_dir is not None else get_xorq_runs_dir()
    expr_dir = base / expr_hash
    runs = Runs(expr_dir=expr_dir)
    return tuple(_run_to_row(run) for run in runs.runs[:max_count])


def _run_to_row(run) -> RunRowData:
    meta = run.read_meta()
    match meta:
        case dict():
            started = meta.get("started_at", "")
            completed = meta.get("completed_at", "")
            raw_ttl = meta.get("ttl")
            return RunRowData(
                run_id=meta.get("run_id", run.run_id),
                status=meta.get("status", "?"),
                cache_type=meta.get("cache_type", ""),
                duration=_compute_duration(started, completed),
                output_format=meta.get("output_format", ""),
                date=_format_run_date(started),
                error=meta.get("error"),
                ttl=int(raw_ttl) if raw_ttl is not None else None,
                output_snapshot_path=meta.get("output_snapshot_path"),
                meta=tuple((str(k), str(v)) for k, v in meta.items()),
            )
        case _:
            return RunRowData(run_id=run.run_id, status="running")


def _format_run_detail(run: RunRowData) -> str:
    meta_dict = dict(run.meta)
    parts = [f"Run: {run.run_id}"]
    for key in (
        "status",
        "started_at",
        "completed_at",
        "output_format",
        "output_snapshot_path",
        "expr_hash",
        "error",
    ):
        match meta_dict.get(key):
            case None:
                pass
            case value:
                parts.append(f"{key}: {value}")
    return "\n".join(parts)


@frozen
class CacheRowData:
    key: str = field(default="", validator=instance_of(str))
    entry_label: str = field(default="", validator=instance_of(str))
    size: str = field(default="", validator=instance_of(str))
    rows: str = field(default="", validator=instance_of(str))
    path: str = field(default="", validator=instance_of(str))
    schema: tuple[tuple[str, str], ...] = field(
        factory=tuple, validator=instance_of(tuple)
    )

    @property
    def row(self) -> tuple[str, ...]:
        return (self.key[:16], self.entry_label, self.size, self.rows)

    @cached_property
    def info_text(self) -> str:
        parts = [f"Key: {self.key}"]
        if self.entry_label and self.entry_label != "—":
            parts.append(f"Entry: {self.entry_label}")
        parts.append(f"Size: {self.size}")
        parts.append(f"Rows: {self.rows}")
        if self.path:
            parts.append(f"Path: {self.path}")
        return "\n".join(parts)


def _format_size(size_bytes: int) -> str:
    match size_bytes:
        case b if b < 1024:
            return f"{b} B"
        case b if b < 1024 * 1024:
            return f"{b / 1024:.1f} KB"
        case b if b < 1024 * 1024 * 1024:
            return f"{b / (1024 * 1024):.1f} MB"
        case b:
            return f"{b / (1024 * 1024 * 1024):.1f} GB"


def _build_cache_entry_map(catalog) -> dict[str, str]:
    """Map parquet cache file paths to entry hashes.

    Uses stable entry hashes (not aliases, which can be moved).

    Sources:
    1. Build-time: entry.parquet_cache_paths from catalog metadata
    2. Runtime: output_snapshot_path from run logs (for caches created via TUI run)
    """
    from xorq.common.utils.logging_utils import Runs, get_xorq_runs_dir  # noqa: PLC0415

    result: dict[str, str] = {}

    # 1. Build-time cache paths from catalog metadata
    for entry in catalog.catalog_entries:
        for path in entry.parquet_cache_paths:
            result[path] = entry.name[:12]

    # 2. Runtime cache paths from run logs
    runs_dir = get_xorq_runs_dir()
    if runs_dir.exists():
        for expr_dir in runs_dir.iterdir():
            if not expr_dir.is_dir():
                continue
            expr_hash = expr_dir.name
            for run in Runs(expr_dir=expr_dir).runs:
                meta = run.read_meta()
                match meta:
                    case {"output_snapshot_path": str(snap_path)}:
                        result.setdefault(snap_path, expr_hash[:12])
                    case _:
                        pass

    return result


def _parquet_to_cache_row(
    parquet_path: Path, entry_map: dict[str, str]
) -> CacheRowData:
    key = parquet_path.stem
    entry_label = entry_map.get(str(parquet_path), "—")
    try:
        size = _format_size(parquet_path.stat().st_size)
    except OSError:
        size = "?"
    schema: tuple[tuple[str, str], ...] = ()
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415

        pf_meta = pq.read_metadata(str(parquet_path))
        rows = str(pf_meta.num_rows)
        arrow_schema = pq.read_schema(str(parquet_path))
        schema = tuple((field.name, str(field.type)) for field in arrow_schema)
    except Exception:
        rows = "?"
    return CacheRowData(
        key=key,
        entry_label=entry_label,
        size=size,
        rows=rows,
        path=str(parquet_path),
        schema=schema,
    )


def _build_cache_rows(catalog) -> tuple[CacheRowData, ...]:
    from xorq.common.utils.caching_utils import get_xorq_cache_dir  # noqa: PLC0415

    cache_dir = get_xorq_cache_dir() / "parquet"
    if not cache_dir.exists():
        return ()
    entry_map = _build_cache_entry_map(catalog)
    return tuple(
        _parquet_to_cache_row(p, entry_map) for p in sorted(cache_dir.glob("*.parquet"))
    )


def _load_catalog_row(entry, aliases=()) -> CatalogRowData:
    return CatalogRowData(entry=entry, aliases=aliases)


@cache
def _catalog_list_cached(catalog, yaml_mtime: float) -> tuple:
    """Compute catalog entry list; auto-invalidates when yaml mtime changes."""
    return tuple(catalog.list())


def _get_catalog_list(catalog) -> tuple:
    """Return catalog entry list, recomputing only when the YAML file has changed."""
    yaml_mtime = catalog.catalog_yaml.yaml_path.stat().st_mtime
    return _catalog_list_cached(catalog, yaml_mtime)


@cache
def _catalog_aliases_cached(catalog, yaml_mtime: float) -> tuple:
    """Compute catalog aliases; auto-invalidates when yaml mtime changes."""
    return tuple(catalog.catalog_aliases)


def _get_catalog_aliases(catalog) -> tuple:
    """Return catalog aliases, recomputing only when the YAML file has changed."""
    yaml_mtime = catalog.catalog_yaml.yaml_path.stat().st_mtime
    return _catalog_aliases_cached(catalog, yaml_mtime)


@cache
def _build_alias_multimap(
    catalog_aliases,
) -> dict[str, tuple[str, ...]]:
    from itertools import groupby  # noqa: PLC0415
    from operator import attrgetter  # noqa: PLC0415

    key = attrgetter("catalog_entry.name")
    sorted_aliases = sorted(catalog_aliases, key=key)
    return {
        name: tuple(sorted(ca.alias for ca in group))
        for name, group in groupby(sorted_aliases, key=key)
    }


def _build_git_log_rows(repo, max_count=100) -> tuple[GitLogRowData, ...]:
    return tuple(
        GitLogRowData(
            hash=commit.hexsha[:12],
            date=datetime.fromtimestamp(commit.committed_date).strftime(
                "%Y-%m-%d %H:%M"
            ),
            message=commit.message.strip().split("\n")[0],
        )
        for commit in repo.iter_commits(max_count=max_count)
    )


def _render_sql_dag(sqls: tuple[tuple[str, str, str], ...]) -> str:
    """Render multiple SQL queries as a topologically-sorted DAG."""
    name_to_sql = {name: (engine, sql) for name, engine, sql in sqls}
    # build dependency graph: name -> set of names it depends on
    deps = {
        name: frozenset(
            ref
            for ref in re.findall(r'FROM "([a-f0-9]{20,})"', sql)
            if ref in name_to_sql
        )
        for name, (_, sql) in name_to_sql.items()
    }
    # topological sort (Kahn's algorithm) — leaves first, main last
    in_degree = {n: len(d) for n, d in deps.items()}
    queue = [n for n, d in in_degree.items() if d == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for n, d in deps.items():
            if node in d:
                in_degree[n] -= 1
                if in_degree[n] == 0:
                    queue.append(n)
    # append any remaining (cycle fallback)
    order.extend(n for n in name_to_sql if n not in order)

    parts = []
    for i, name in enumerate(order):
        engine, sql = name_to_sql[name]
        label = "main" if name == "main" else name[:12]
        parts.append(f"-- [{label}] ({engine})\n{sql}")
        if i < len(order) - 1:
            parts.append("  \u2193")
    return "\n\n".join(parts)


def _revision_pair(i, rev_entry, commit):
    exists = rev_entry.exists()
    col_count, cached = _entry_info(rev_entry) if exists else (None, None)
    row = RevisionRowData(
        hash=rev_entry.name,
        column_count=col_count,
        cached=cached,
        commit_date=datetime.fromtimestamp(commit.committed_date).strftime(
            "%Y-%m-%d %H:%M"
        ),
        is_current=(i == 0),
    )
    return row, (rev_entry, commit, exists)


@frozen
class _TogglePanelState:
    visible: bool = field(default=False, validator=instance_of(bool))
    loaded: bool = field(default=False, validator=instance_of(bool))
    entry_hash: str | None = field(default=None, validator=optional(instance_of(str)))


# --- New data models for upcoming workflows ---


@frozen
class SinkConfig:
    """Configuration for a sink-based run (Snowflake, etc.)."""

    entry_name: str = field(validator=instance_of(str))
    expr_hash: str = field(validator=instance_of(str))
    target_backend: str = field(validator=instance_of(str))
    table_name: str = field(validator=instance_of(str))
    unique_key: tuple[str, ...] = field(factory=tuple, validator=instance_of(tuple))
    strategy: str = field(default="append", validator=instance_of(str))
    incremental_column: str | None = field(
        default=None, validator=optional(instance_of(str))
    )


@frozen
class ServiceStatus:
    """Status of a Flight gRPC server instance."""

    name: str = field(validator=instance_of(str))
    endpoint: str = field(validator=instance_of(str))
    status: str = field(default="stopped", validator=instance_of(str))
    started_at: str | None = field(default=None, validator=optional(instance_of(str)))

    @property
    def status_icon(self) -> str:
        match self.status:
            case "running":
                return "●"
            case "stopped":
                return "○"
            case "error":
                return "✕"
            case _:
                return "?"

    @property
    def row(self) -> tuple[str, ...]:
        return (self.status_icon, self.name, self.endpoint, self.started_at or "—")


@frozen
class ComposeConfig:
    """Configuration for a catalog compose operation."""

    entries: tuple[str, ...] = field(validator=instance_of(tuple))
    code: str | None = field(default=None, validator=optional(instance_of(str)))
    alias: str | None = field(default=None, validator=optional(instance_of(str)))
