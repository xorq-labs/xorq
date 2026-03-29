import re
import threading
import zipfile
from datetime import datetime
from functools import cache, cached_property
from pathlib import Path

import yaml12
from attr import field, frozen
from attr.validators import instance_of, optional
from pygments.style import Style as PygmentsStyle
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
)
from rich.syntax import Syntax
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    RadioButton,
    RadioSet,
    Static,
)

from xorq.catalog.catalog import CatalogEntry


DEFAULT_REFRESH_INTERVAL = 10


class XorqSQLStyle(PygmentsStyle):
    """Pygments style using xorq brand colors for SQL syntax highlighting."""

    background_color = "#0a2a2e"
    styles = {
        Token: "#C1F0FF",
        Keyword: "bold #C1F0FF",
        Keyword.DML: "bold #C1F0FF",
        Keyword.DDL: "bold #C1F0FF",
        Name: "#C1F0FF",
        Name.Builtin: "#7ED4C8",
        String: "#2BBE75",
        String.Single: "#2BBE75",
        Number: "#F5CA2C",
        Number.Integer: "#F5CA2C",
        Number.Float: "#F5CA2C",
        Comment: "italic #4AA8EC",
        Comment.Single: "italic #4AA8EC",
        Operator: "#5abfb5",
        Punctuation: "#7ED4C8",
    }


XORQ_DARK = Theme(
    name="xorq-dark",
    primary="#C1F0FF",
    secondary="#4AA8EC",
    warning="#F5CA2C",
    error="#FF4757",
    success="#2BBE75",
    accent="#C1F0FF",
    foreground="#C1F0FF",
    background="#05181A",
    surface="#0a2a2e",
    panel="#0f3338",
    dark=True,
)

COLUMNS = ("KIND", "ALIAS", "HASH", "BACKEND", "CACHED")

SCHEMA_PREVIEW_COLUMNS = ("NAME", "TYPE")

REVISION_COLUMNS = ("STATUS", "HASH", "COLUMNS", "CACHED", "DATE")

GIT_LOG_COLUMNS = ("HASH", "DATE", "MESSAGE")

RUN_COLUMNS = ("STATUS", "RUN ID", "CACHE", "DURATION", "FORMAT", "DATE")

CACHE_PANEL_COLUMNS = ("KEY", "ENTRY", "SIZE", "ROWS")


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
    # build dependency graph: name → set of names it depends on
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
            parts.append("  ↓")
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


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


@frozen
class _TogglePanelState:
    visible: bool = field(default=False, validator=instance_of(bool))
    loaded: bool = field(default=False, validator=instance_of(bool))
    entry_hash: str | None = field(default=None, validator=optional(instance_of(str)))


_CACHE_OPTIONS = (
    ("snapshot", "snapshot"),
    ("source", "source"),
    ("none", "no cache"),
    ("ttl_snapshot", "ttl snapshot"),
)


class RunOptionsScreen(Screen):
    """Lightweight modal for selecting cache strategy. ctrl+r=run, Esc=cancel."""

    BINDINGS = (
        ("escape", "cancel", "Cancel"),
        ("ctrl+r", "confirm", "Run"),
    )

    def __init__(self, entry_name: str, expr_hash: str):
        super().__init__()
        self._entry_name = entry_name
        self._expr_hash = expr_hash

    def compose(self) -> ComposeResult:
        with Vertical(id="run-options-container"):
            yield Static(
                f" run {self._entry_name}  [dim]ctrl+r=run  esc=cancel[/]",
                id="run-options-title",
            )
            with RadioSet(id="cache-strategy"):
                for i, (_, label) in enumerate(_CACHE_OPTIONS):
                    yield RadioButton(label, value=(i == 0))
            with Horizontal(id="ttl-row"):
                yield Static(" ttl:", id="ttl-label")
                yield Input(placeholder="300", id="ttl-input", restrict=r"^\d*$")

    def on_mount(self) -> None:
        self.query_one("#ttl-row").display = False

    def action_confirm(self) -> None:
        self._do_confirm()

    @on(RadioSet.Changed, "#cache-strategy")
    def _on_cache_strategy_changed(self, event: RadioSet.Changed) -> None:
        selected_idx = event.radio_set.pressed_index
        cache_type = _CACHE_OPTIONS[selected_idx][0]
        self.query_one("#ttl-row").display = cache_type == "ttl_snapshot"

    def _do_confirm(self) -> None:
        radio_set = self.query_one("#cache-strategy", RadioSet)
        selected_idx = radio_set.pressed_index
        cache_type = _CACHE_OPTIONS[selected_idx][0]

        ttl = None
        if cache_type == "ttl_snapshot":
            ttl_text = self.query_one("#ttl-input", Input).value.strip()
            ttl = int(ttl_text) if ttl_text else 300

        self.dismiss(
            RunConfig(
                entry_name=self._entry_name,
                expr_hash=self._expr_hash,
                cache_type=cache_type,
                ttl=ttl,
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class RunDataScreen(Screen):
    """Full-screen parquet data viewer pushed on top of the catalog screen."""

    BINDINGS = (
        ("q", "go_back", "Back"),
        ("escape", "go_back", "Back"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("h", "scroll_left", "Left"),
        ("l", "scroll_right", "Right"),
    )

    def __init__(self, parquet_path: str, title: str):
        super().__init__()
        self._parquet_path = parquet_path
        self._title = title

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="run-data-container"):
            yield Static("", id="run-data-status")
            yield DataTable(id="run-data-table")
        yield Footer()

    def on_mount(self) -> None:
        self.title = self._title
        table = self.query_one("#run-data-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.loading = True
        self._load_parquet_data()

    @work(thread=True, exit_on_error=False)
    def _load_parquet_data(self) -> None:
        import pyarrow.parquet as pq  # noqa: PLC0415

        try:
            pf = pq.ParquetFile(self._parquet_path)
            total_rows = pf.metadata.num_rows
            file_size = Path(self._parquet_path).stat().st_size
            # Read first 500 rows for preview
            arrow_table = pf.read().slice(0, 500)
            df = arrow_table.to_pandas()
            columns = tuple(str(c) for c in df.columns)
            rows = tuple(
                tuple(str(round(v, 2)) if isinstance(v, float) else str(v) for v in row)
                for row in df.itertuples(index=False)
            )
            self.app.call_from_thread(
                self._render_data, columns, rows, total_rows, len(columns), file_size
            )
        except Exception as e:
            self.app.call_from_thread(self._render_error, str(e))

    def _render_data(self, columns, rows, total_rows, num_cols, file_size) -> None:
        size_mb = file_size / (1024 * 1024)
        self.query_one("#run-data-status", Static).update(
            f" {total_rows:,} rows \u00b7 {num_cols} columns \u00b7 {size_mb:.1f} MB"
        )
        table = self.query_one("#run-data-table", DataTable)
        table.clear(columns=True)
        table.loading = False
        for col in columns:
            table.add_column(col, key=col)
        for i, row in enumerate(rows):
            table.add_row(*row, key=str(i))

    def _render_error(self, message: str) -> None:
        self.query_one("#run-data-status", Static).update(f" Error: {message}")
        self.query_one("#run-data-table", DataTable).loading = False

    def _focused_widget(self) -> DataTable:
        focused = self.app.focused
        if isinstance(focused, DataTable):
            return focused
        return self.query_one("#run-data-table", DataTable)

    def action_cursor_down(self) -> None:
        self._focused_widget().action_cursor_down()

    def action_cursor_up(self) -> None:
        self._focused_widget().action_cursor_up()

    def action_scroll_left(self) -> None:
        self._focused_widget().action_scroll_left()

    def action_scroll_right(self) -> None:
        self._focused_widget().action_scroll_right()

    def action_go_back(self) -> None:
        self.app.pop_screen()


class CatalogScreen(Screen):
    BINDINGS = (
        ("q", "quit_app", "Quit"),
        ("ctrl+c", "quit_app", "Quit"),
        ("h", "scroll_left", "Left"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("l", "scroll_right", "Right"),
        ("tab", "focus_next_panel", "Next"),
        ("shift+tab", "focus_prev_panel", "Prev"),
        ("r", "run_entry", "Run"),
        ("c", "toggle_caches", "Caches"),
        ("g", "toggle_git_log", "Git Log"),
        ("d", "toggle_data_preview", "Data"),
        ("p", "toggle_profiles", "Profiles"),
    )

    FOCUS_CYCLE = (
        "#catalog-table",
        "#runs-table",
        "#caches-table",
        "#sql-panel",
        "#schema-preview-table",
        "#revisions-preview-table",
    )

    def __init__(self, refresh_interval=DEFAULT_REFRESH_INTERVAL):
        super().__init__()
        self._refresh_interval = refresh_interval
        self._row_cache: dict[str, CatalogRowData] = {}
        self._run_row_cache: dict[str, RunRowData] = {}
        self._current_runs_hash: str | None = None
        self._git_log_visible = False
        self._git_log_loaded = False
        self._refresh_lock = threading.Lock()
        self._data_preview = _TogglePanelState()
        self._profiles = _TogglePanelState()
        self._caches_visible = False
        self._caches_loaded = False
        self._cache_row_paths: dict[str, str] = {}
        self._cache_row_data: dict[str, CacheRowData] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-split"):
            with Vertical(id="left-column"):
                with Vertical(id="catalog-panel"):
                    yield DataTable(id="catalog-table")
                with Vertical(id="runs-panel"):
                    yield DataTable(id="runs-table")
                with Vertical(id="caches-panel"):
                    yield DataTable(id="caches-table")
                with Vertical(id="revisions-panel"):
                    yield DataTable(id="revisions-preview-table")
                with Vertical(id="git-log-panel"):
                    yield DataTable(id="git-log-table")
            with Vertical(id="right-column"):
                with Vertical(id="main-content-panel"):
                    with VerticalScroll(id="sql-panel"):
                        yield Static("", id="sql-preview")
                    with Vertical(id="inline-data-panel"):
                        yield Static("", id="inline-data-status")
                        yield DataTable(id="inline-data-table")
                with Vertical(id="info-panel"):
                    yield Static("", id="info-content")
                with Vertical(id="schema-panel"):
                    with Horizontal(id="schema-split"):
                        with Vertical(id="schema-in-half"):
                            yield DataTable(id="schema-in-table")
                        with Vertical(id="schema-out-half"):
                            yield DataTable(id="schema-preview-table")
                with Vertical(id="data-preview-panel"):
                    yield Static("", id="data-preview-status")
                    yield DataTable(id="data-preview-table")
                with Vertical(id="profiles-panel"):
                    yield DataTable(id="profiles-table")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#catalog-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        for col in COLUMNS:
            table.add_column(col, key=col)

        schema_in_table = self.query_one("#schema-in-table", DataTable)
        schema_in_table.cursor_type = "row"
        schema_in_table.zebra_stripes = True
        for col in SCHEMA_PREVIEW_COLUMNS:
            schema_in_table.add_column(col, key=col)

        schema_table = self.query_one("#schema-preview-table", DataTable)
        schema_table.cursor_type = "row"
        schema_table.zebra_stripes = True
        for col in SCHEMA_PREVIEW_COLUMNS:
            schema_table.add_column(col, key=col)

        rev_table = self.query_one("#revisions-preview-table", DataTable)
        rev_table.cursor_type = "row"
        rev_table.zebra_stripes = True
        for col in REVISION_COLUMNS:
            rev_table.add_column(col, key=col)

        runs_table = self.query_one("#runs-table", DataTable)
        runs_table.cursor_type = "row"
        runs_table.zebra_stripes = True
        for col in RUN_COLUMNS:
            runs_table.add_column(col, key=col)

        git_log_table = self.query_one("#git-log-table", DataTable)
        git_log_table.cursor_type = "row"
        git_log_table.zebra_stripes = True
        for col in GIT_LOG_COLUMNS:
            git_log_table.add_column(col, key=col)

        data_table = self.query_one("#data-preview-table", DataTable)
        data_table.cursor_type = "none"
        data_table.zebra_stripes = True
        data_table.loading = True

        profiles_table = self.query_one("#profiles-table", DataTable)
        profiles_table.cursor_type = "row"
        profiles_table.zebra_stripes = True
        profiles_table.add_column("NAME", key="name")
        profiles_table.add_column("BACKEND", key="backend")
        profiles_table.add_column("PARAMETERS", key="params")
        profiles_table.add_column("ENV VARS", key="env_vars")

        self.query_one("#catalog-panel").border_title = "Expressions"
        self.query_one("#schema-panel").border_title = "Schema"
        self.query_one("#schema-in-half").display = False
        self.query_one("#sql-panel").border_title = "SQL"
        inline_data_panel = self.query_one("#inline-data-panel")
        inline_data_panel.border_title = "Data Preview"
        inline_data_panel.display = False
        inline_dt = self.query_one("#inline-data-table", DataTable)
        inline_dt.cursor_type = "row"
        inline_dt.zebra_stripes = True
        self.query_one("#info-panel").border_title = "Info"
        self.query_one("#revisions-panel").border_title = "Revisions"
        self.query_one("#runs-panel").border_title = "Runs"
        git_log_panel = self.query_one("#git-log-panel")
        git_log_panel.border_title = "Git Log"
        git_log_panel.display = False
        data_panel = self.query_one("#data-preview-panel")
        data_panel.border_title = "Data Preview"
        data_panel.display = False
        profiles_panel = self.query_one("#profiles-panel")
        profiles_panel.border_title = "Profiles"
        profiles_panel.display = False

        caches_table = self.query_one("#caches-table", DataTable)
        caches_table.cursor_type = "row"
        caches_table.zebra_stripes = True
        for col in CACHE_PANEL_COLUMNS:
            caches_table.add_column(col, key=col)
        caches_panel = self.query_one("#caches-panel")
        caches_panel.border_title = "Caches"
        caches_panel.display = False

        self.query_one("#status-bar", Static).update(" Loading catalog...")

        self.set_interval(self._refresh_interval, self._do_refresh)

    @on(DataTable.RowHighlighted, "#catalog-table")
    def _on_catalog_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        schema_in_table = self.query_one("#schema-in-table", DataTable)
        schema_in_table.clear()
        schema_out_table = self.query_one("#schema-preview-table", DataTable)
        schema_out_table.clear()
        sql_preview = self.query_one("#sql-preview", Static)
        info_content = self.query_one("#info-content", Static)
        rev_table = self.query_one("#revisions-preview-table", DataTable)
        rev_table.clear()
        runs_table = self.query_one("#runs-table", DataTable)
        runs_table.clear()

        # Restore SQL when catalog table has focus (click or j/k navigation)
        if self.app.focused is self.query_one("#catalog-table", DataTable):
            self._hide_inline_data()

        if event.row_key is None:
            sql_preview.update("")
            info_content.update("")
            self.query_one("#schema-in-half").display = False
            self.query_one("#revisions-panel").border_title = "Revisions"
            self.query_one("#runs-panel").border_subtitle = ""
            self._current_runs_hash = None
            self._run_row_cache.clear()
            self._reset_toggle_panels()
            return

        row_data = self._row_cache.get(str(event.row_key.value))
        if row_data is None:
            sql_preview.update("")
            info_content.update("")
            self.query_one("#schema-in-half").display = False
            self.query_one("#revisions-panel").border_title = "Revisions"
            self.query_one("#runs-panel").border_subtitle = ""
            self._current_runs_hash = None
            self._run_row_cache.clear()
            self._reset_toggle_panels()
            return

        # Schema
        schema_panel = self.query_one("#schema-panel")
        match row_data.schema_in:
            case None:
                self.query_one("#schema-in-half").display = False
                schema_panel.border_title = "Schema"
                schema_panel.border_subtitle = f"{len(row_data.schema_out)} cols"
            case schema_in:
                self.query_one("#schema-in-half").display = True
                schema_panel.border_title = "Schemas"
                schema_panel.border_subtitle = (
                    f"{len(schema_in)} in · {len(row_data.schema_out)} out"
                )
                for name, dtype in schema_in:
                    schema_in_table.add_row(name, dtype)
        for name, dtype in row_data.schema_out:
            schema_out_table.add_row(name, dtype)

        # SQL preview (sync — in-memory AST compilation)
        sql_panel = self.query_one("#sql-panel")
        match row_data.sqls:
            case ():
                sql_preview.update("(SQL unavailable)")
                sql_panel.border_subtitle = ""
            case ((_, engine, sql),):
                sql_preview.update(
                    Syntax(sql, "sql", theme=XorqSQLStyle, word_wrap=True)
                )
                sql_panel.border_subtitle = engine
            case sqls:
                sql_preview.update(
                    Syntax(
                        _render_sql_dag(sqls),
                        "sql",
                        theme=XorqSQLStyle,
                        word_wrap=True,
                    )
                )
                engines = sorted({engine for _, engine, _ in sqls})
                sql_panel.border_subtitle = (
                    f"{len(sqls)} queries \u00b7 {', '.join(engines)}"
                )

        # Info panel (sync)
        info_content.update(row_data.info_text)

        # Revisions preview (async — requires git I/O)
        match row_data.aliases:
            case (first_alias, *_):
                catalog_alias = next(
                    (ca for ca in self.catalog_aliases if ca.alias == first_alias),
                    None,
                )
                match catalog_alias:
                    case None:
                        self.query_one(
                            "#revisions-panel"
                        ).border_title = "Revisions — (alias not found)"
                    case _:
                        self.query_one(
                            "#revisions-panel"
                        ).border_title = f"Revisions — {first_alias}"
                        self._load_revisions_preview(catalog_alias)
            case _:
                self.query_one(
                    "#revisions-panel"
                ).border_title = "Revisions — (no alias)"

        # Runs (async — reads from disk)
        self._load_runs(row_data.hash)

        # Reset toggle panels on row change
        self._reset_toggle_panels()

    def _reset_toggle_panels(self) -> None:
        self._data_preview = _TogglePanelState()
        self.query_one("#data-preview-panel").display = False
        dt = self.query_one("#data-preview-table", DataTable)
        dt.clear(columns=True)
        dt.loading = True

        self._profiles = _TogglePanelState()
        self.query_one("#profiles-panel").display = False
        self.query_one("#profiles-table", DataTable).clear()

    @work(thread=True, exit_on_error=False)
    def _do_refresh(self) -> None:
        if not self._refresh_lock.acquire(blocking=False):
            return
        try:
            self._do_refresh_locked()
        finally:
            self._refresh_lock.release()

    @property
    def catalog_aliases(self) -> tuple:
        catalog = self.app._catalog
        if catalog is None:
            return ()
        return _get_catalog_aliases(catalog)

    def _do_refresh_locked(self) -> None:
        catalog = self.app._catalog
        if catalog is None:
            return
        repo_path = catalog.repo.working_dir
        expected_keys = frozenset(_get_catalog_list(catalog))
        alias_multimap = _build_alias_multimap(self.catalog_aliases)

        # preserve insertion order from _row_cache (dict is ordered in Python 3.7+)
        cached_rows = tuple(
            self._row_cache[k] for k in self._row_cache if k in expected_keys
        )
        new_keys = expected_keys - self._row_cache.keys()
        removed = self._row_cache.keys() - expected_keys

        # evict removed entries
        self._row_cache = {
            k: v for k, v in self._row_cache.items() if k in expected_keys
        }

        match (bool(removed), bool(cached_rows)):
            case (True, _) | (_, False):
                # re-render when rows were removed or on first refresh
                self.app.call_from_thread(self._render_refresh, repo_path, cached_rows)
            case _:
                pass

        # load new entries incrementally (expensive I/O, off the main thread)
        for entry_hash in new_keys:
            entry = catalog.get_catalog_entry(entry_hash)
            aliases = alias_multimap.get(entry_hash, ())
            row_data = _load_catalog_row(entry, aliases)
            self._row_cache[row_data.row_key] = row_data
            self.app.call_from_thread(self._render_catalog_row, row_data)

        if self._git_log_visible:
            git_rows = _build_git_log_rows(catalog.repo)
            self.app.call_from_thread(self._render_git_log, git_rows)

        if self._current_runs_hash:
            run_rows = _build_run_rows(self._current_runs_hash)
            self.app.call_from_thread(self._render_runs, run_rows)

        stamp = datetime.now().strftime("%H:%M:%S")
        self.app.call_from_thread(self._render_status, stamp, repo_path)

    def _render_refresh(self, repo_path, cached_rows) -> None:
        with self.app.batch_update():
            catalog_name = Path(repo_path).name
            self.query_one(
                "#catalog-panel"
            ).border_title = f"Expressions — {catalog_name}"

            table = self.query_one("#catalog-table", DataTable)
            saved_cursor = table.cursor_row
            table.clear()
            for row_data in cached_rows:
                table.add_row(*row_data.row, key=row_data.row_key)
            count = table.row_count
            if saved_cursor is not None and count > 0:
                table.move_cursor(row=min(saved_cursor, count - 1))

    def _render_catalog_row(self, row_data) -> None:
        with self.app.batch_update():
            table = self.query_one("#catalog-table", DataTable)
            if len(table.columns) < len(COLUMNS):
                return
            table.add_row(*row_data.row, key=row_data.row_key)

    def _render_status(self, stamp, repo_path) -> None:
        count = self.query_one("#catalog-table", DataTable).row_count
        self.query_one("#status-bar", Static).update(
            f" {count} entries \u00b7 {repo_path} \u00b7 {stamp}"
        )

    # --- Toggle: Git Log ---

    def action_toggle_git_log(self) -> None:
        self._git_log_visible = not self._git_log_visible
        self.query_one("#git-log-panel").display = self._git_log_visible
        if self._git_log_visible and not self._git_log_loaded:
            self._load_git_log()

    @work(thread=True)
    def _load_git_log(self) -> None:
        catalog = self.app._catalog
        if catalog is None:
            return
        rows = _build_git_log_rows(catalog.repo)
        self._git_log_loaded = True
        self.app.call_from_thread(self._render_git_log, rows)

    def _render_git_log(self, rows) -> None:
        with self.app.batch_update():
            table = self.query_one("#git-log-table", DataTable)
            table.clear()
            for i, row_data in enumerate(rows):
                table.add_row(*row_data.row, key=str(i))

    # --- Toggle: Data Preview ---

    def action_toggle_data_preview(self) -> None:
        table = self.query_one("#catalog-table", DataTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        entry_hash = str(row_key.value)
        row_data = self._row_cache.get(entry_hash)
        if row_data is None:
            return

        toggled = not self._data_preview.visible
        self._data_preview = _TogglePanelState(
            visible=toggled,
            loaded=self._data_preview.loaded,
            entry_hash=self._data_preview.entry_hash,
        )
        self.query_one("#data-preview-panel").display = toggled

        if not toggled:
            return

        match row_data.cached:
            case True:
                if (
                    not self._data_preview.loaded
                    or self._data_preview.entry_hash != entry_hash
                ):
                    self._data_preview = _TogglePanelState(
                        visible=True,
                        loaded=True,
                        entry_hash=entry_hash,
                    )
                    self.query_one("#data-preview-status", Static).update(
                        " Loading data preview..."
                    )
                    self._load_data_preview(row_data.entry)
            case _:
                self.query_one("#data-preview-status", Static).update(
                    " uncached — run to materialize"
                )
                self.query_one("#data-preview-table", DataTable).loading = False

    @work(thread=True, exit_on_error=False)
    def _load_data_preview(self, entry) -> None:
        try:
            df = entry.expr.head(50).execute()
            columns = tuple(str(c) for c in df.columns)
            rows = tuple(
                tuple(str(round(v, 2)) if isinstance(v, float) else str(v) for v in row)
                for row in df.itertuples(index=False)
            )
            total_rows = len(df)
            self.app.call_from_thread(
                self._render_data_preview, columns, rows, total_rows
            )
        except Exception as e:
            self.app.call_from_thread(self._render_data_preview_error, str(e))

    def _render_data_preview(self, columns, rows, total_rows) -> None:
        with self.app.batch_update():
            self.query_one("#data-preview-status", Static).update(
                f" Data Preview — {total_rows} rows (max 50)"
            )
            data_table = self.query_one("#data-preview-table", DataTable)
            data_table.clear(columns=True)
            data_table.loading = False
            for col in columns:
                data_table.add_column(col, key=col)
            for i, row in enumerate(rows):
                data_table.add_row(*row, key=str(i))
            data_table.cursor_type = "row"

    def _render_data_preview_error(self, message) -> None:
        self.query_one("#data-preview-status", Static).update(f" Error: {message}")
        self.query_one("#data-preview-table", DataTable).loading = False

    # --- Toggle: Profiles ---

    def action_toggle_profiles(self) -> None:
        table = self.query_one("#catalog-table", DataTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        entry_hash = str(row_key.value)
        row_data = self._row_cache.get(entry_hash)
        if row_data is None:
            return

        toggled = not self._profiles.visible
        self._profiles = _TogglePanelState(
            visible=toggled,
            loaded=self._profiles.loaded,
            entry_hash=self._profiles.entry_hash,
        )
        self.query_one("#profiles-panel").display = toggled

        if toggled and (
            not self._profiles.loaded or self._profiles.entry_hash != entry_hash
        ):
            self._profiles = _TogglePanelState(
                visible=True,
                loaded=True,
                entry_hash=entry_hash,
            )
            self._load_profiles(row_data.entry)

    @work(thread=True, exit_on_error=False)
    def _load_profiles(self, entry) -> None:
        if not entry.catalog_path.exists() or not zipfile.is_zipfile(
            entry.catalog_path
        ):
            return

        env_re = re.compile(r"^\$\{(.+)\}$|^\$(.+)$")

        def _extract_env_vars(kwargs):
            return tuple(
                m.group(1) or m.group(2)
                for v in kwargs.values()
                if isinstance(v, str) and (m := env_re.match(v))
            )

        with zipfile.ZipFile(entry.catalog_path, "r") as zf:
            member_path = f"{entry.name}/profiles.yaml"
            if member_path not in zf.namelist():
                return
            data = yaml.safe_load(zf.read(member_path))
        match data:
            case dict():
                pass
            case _:
                return
        rows = tuple(
            (
                pname,
                pdata.get("con_name", "?"),
                ", ".join(
                    f"{k}={v}"
                    for k, v in (pdata.get("kwargs_tuple") or {}).items()
                    if v is not None
                ),
                ", ".join(_extract_env_vars(pdata.get("kwargs_tuple") or {})),
            )
            for pname, pdata in data.items()
        )
        self.app.call_from_thread(self._render_profiles, rows)

    def _render_profiles(self, rows) -> None:
        with self.app.batch_update():
            table = self.query_one("#profiles-table", DataTable)
            table.clear()
            for i, (name, backend, params, env_vars) in enumerate(rows):
                table.add_row(name, backend, params, env_vars, key=str(i))

    # --- Toggle: Caches ---

    def action_toggle_caches(self) -> None:
        self._caches_visible = not self._caches_visible
        self.query_one("#caches-panel").display = self._caches_visible
        if self._caches_visible and not self._caches_loaded:
            self._load_caches()

    @work(thread=True, exit_on_error=False)
    def _load_caches(self) -> None:
        catalog = self.app._catalog
        if catalog is None:
            return
        rows = _build_cache_rows(catalog)
        self._caches_loaded = True
        self.app.call_from_thread(self._render_caches, rows)

    def _render_caches(self, rows: tuple[CacheRowData, ...]) -> None:
        with self.app.batch_update():
            table = self.query_one("#caches-table", DataTable)
            table.clear()
            self._cache_row_paths.clear()
            self._cache_row_data.clear()
            for i, row_data in enumerate(rows):
                key = str(i)
                table.add_row(*row_data.row, key=key)
                self._cache_row_paths[key] = row_data.path
                self._cache_row_data[key] = row_data
            panel = self.query_one("#caches-panel")
            match rows:
                case ():
                    panel.border_subtitle = "empty"
                case _:
                    total_size = (
                        sum(
                            Path(p).stat().st_size
                            for p in (self._get_cache_dir() / "parquet").glob(
                                "*.parquet"
                            )
                            if p.exists()
                        )
                        if self._get_cache_dir()
                        else 0
                    )
                    panel.border_subtitle = (
                        f"{len(rows)} files · {_format_size(total_size)}"
                    )

    @staticmethod
    def _get_cache_dir() -> Path | None:
        try:
            from xorq.common.utils.caching_utils import (  # noqa: PLC0415
                get_xorq_cache_dir,
            )

            return get_xorq_cache_dir()
        except Exception:
            return None

    # --- Revisions Preview ---

    @work(thread=True, exit_on_error=False)
    def _load_revisions_preview(self, catalog_alias) -> None:
        try:
            raw_revisions = catalog_alias.list_revisions()
        except (KeyError, ValueError, OSError):
            return
        revision_rows = tuple(
            row
            for i, (rev_entry, commit) in enumerate(raw_revisions)
            for row, _ in (_revision_pair(i, rev_entry, commit),)
        )
        self.app.call_from_thread(self._render_revisions_preview, revision_rows)

    def _render_revisions_preview(self, revision_rows) -> None:
        with self.app.batch_update():
            rev_table = self.query_one("#revisions-preview-table", DataTable)
            rev_table.clear()
            for i, row_data in enumerate(revision_rows):
                rev_table.add_row(*row_data.row, key=str(i))
            rev_panel = self.query_one("#revisions-panel")
            rev_panel.border_subtitle = f"{len(revision_rows)} revisions"

    # --- Runs ---

    @work(thread=True, exit_on_error=False)
    def _load_runs(self, expr_hash: str) -> None:
        rows = _build_run_rows(expr_hash)
        self._current_runs_hash = expr_hash
        self.app.call_from_thread(self._render_runs, rows)

    def _render_runs(self, rows: tuple[RunRowData, ...]) -> None:
        with self.app.batch_update():
            runs_table = self.query_one("#runs-table", DataTable)
            runs_table.clear()
            self._run_row_cache.clear()
            for i, row_data in enumerate(rows):
                key = str(i)
                runs_table.add_row(*row_data.row, key=key)
                self._run_row_cache[key] = row_data
            runs_panel = self.query_one("#runs-panel")
            match rows:
                case ():
                    runs_panel.border_subtitle = "no runs"
                case _:
                    runs_panel.border_subtitle = f"{len(rows)} runs"

    @on(DataTable.RowHighlighted, "#runs-table")
    def _on_run_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None:
            return
        run_data = self._run_row_cache.get(str(event.row_key.value))
        if run_data is None:
            return
        info_content = self.query_one("#info-content", Static)
        info_content.update(_format_run_detail(run_data))

        # Only show inline data when runs table is focused
        if self.app.focused is not self.query_one("#runs-table", DataTable):
            return
        match run_data.output_snapshot_path:
            case str(path) if Path(path).exists():
                self._show_inline_data(path, f"run {run_data.run_id_display}")
            case _:
                self._hide_inline_data()

    @on(DataTable.RowHighlighted, "#caches-table")
    def _on_cache_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None:
            return
        # Only update panels when caches table is focused
        if self.app.focused is not self.query_one("#caches-table", DataTable):
            return

        row_key = str(event.row_key.value)
        cache_data = self._cache_row_data.get(row_key)
        if cache_data is None:
            return

        # Info panel
        self.query_one("#info-content", Static).update(cache_data.info_text)

        # Schema panel — show output schema from parquet columns
        schema_in_half = self.query_one("#schema-in-half")
        schema_in_half.display = False
        schema_panel = self.query_one("#schema-panel")
        schema_panel.border_title = "Schema"

        schema_out_table = self.query_one("#schema-preview-table", DataTable)
        schema_out_table.clear()
        match cache_data.schema:
            case ():
                schema_panel.border_subtitle = ""
            case cols:
                schema_panel.border_subtitle = f"{len(cols)} cols"
                for name, dtype in cols:
                    schema_out_table.add_row(name, dtype)

        # Inline data preview
        path = self._cache_row_paths.get(row_key)
        if path and Path(path).exists():
            self._show_inline_data(path, Path(path).stem[:16])
        else:
            self._hide_inline_data()

    def _show_inline_data(self, parquet_path: str, label: str) -> None:
        self.query_one("#sql-panel").display = False
        panel = self.query_one("#inline-data-panel")
        panel.display = True
        panel.border_title = f"Data Preview — {label}"
        self.query_one("#inline-data-status", Static).update(" Loading...")
        self.query_one("#inline-data-table", DataTable).loading = True
        self._load_inline_data(parquet_path)

    def _hide_inline_data(self) -> None:
        self.query_one("#inline-data-panel").display = False
        self.query_one("#sql-panel").display = True

    @work(thread=True, exit_on_error=False)
    def _load_inline_data(self, parquet_path: str) -> None:
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415

            pf = pq.ParquetFile(parquet_path)
            total_rows = pf.metadata.num_rows
            df = pf.read().slice(0, 100).to_pandas()
            columns = tuple(str(c) for c in df.columns)
            rows = tuple(
                tuple(str(round(v, 2)) if isinstance(v, float) else str(v) for v in row)
                for row in df.itertuples(index=False)
            )
            self.app.call_from_thread(
                self._render_inline_data, columns, rows, total_rows
            )
        except Exception as e:
            self.app.call_from_thread(self._render_inline_data_error, str(e))

    def _render_inline_data(self, columns, rows, total_rows) -> None:
        self.query_one("#inline-data-status", Static).update(
            f" {total_rows:,} rows · {len(columns)} cols (showing {len(rows)})"
        )
        table = self.query_one("#inline-data-table", DataTable)
        table.clear(columns=True)
        table.loading = False
        for col in columns:
            table.add_column(col, key=col)
        for i, row in enumerate(rows):
            table.add_row(*row, key=str(i))

    def _render_inline_data_error(self, message: str) -> None:
        self.query_one("#inline-data-status", Static).update(f" Error: {message}")
        self.query_one("#inline-data-table", DataTable).loading = False

    # --- Run Execution ---

    def _get_current_alias(self) -> str | None:
        table = self.query_one("#catalog-table", DataTable)
        if table.row_count == 0:
            return None
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        row_data = self._row_cache.get(str(row_key.value))
        match row_data:
            case CatalogRowData(aliases=(first_alias, *_)):
                return first_alias
            case _:
                return None

    def action_run_entry(self) -> None:
        table = self.query_one("#catalog-table", DataTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        entry_hash = str(row_key.value)
        row_data = self._row_cache.get(entry_hash)
        if row_data is None:
            return

        if row_data.kind == "unbound_expr":
            self.query_one("#status-bar", Static).update(
                " Cannot run unbound expression — compose with a source first"
            )
            return

        match row_data.aliases:
            case (first_alias, *_):
                entry_name = first_alias
            case _:
                entry_name = entry_hash

        self.app.push_screen(
            RunOptionsScreen(entry_name, entry_hash),
            callback=self._on_run_options_dismissed,
        )

    def _on_run_options_dismissed(self, config: RunConfig | None) -> None:
        if config is None:
            return
        self.query_one("#runs-panel").border_subtitle = "running..."
        self._execute_run(config)

    @work(thread=True, exit_on_error=False)
    def _execute_run(self, config: RunConfig) -> None:
        import datetime as dt  # noqa: PLC0415
        import os  # noqa: PLC0415

        from xorq.common.utils.logging_utils import RunLogger  # noqa: PLC0415
        from xorq.common.utils.profile_utils import timed  # noqa: PLC0415

        catalog = self.app._catalog
        if catalog is None:
            return

        try:
            catalog_entry = catalog.get_catalog_entry(
                config.entry_name, maybe_alias=True
            )
            expr = catalog_entry.expr
            snapshot_path = None

            match config.cache_type:
                case "snapshot":
                    from xorq.caching import ParquetSnapshotCache  # noqa: PLC0415

                    cache = ParquetSnapshotCache.from_kwargs()
                    run_expr = expr.cache(cache=cache)
                    snapshot_path = str(cache.storage.get_path(cache.calc_key(expr)))
                case "source":
                    from xorq.caching import SourceCache  # noqa: PLC0415

                    cache = SourceCache.from_kwargs()
                    run_expr = expr.cache(cache=cache)
                case "ttl_snapshot":
                    from xorq.caching import (  # noqa: PLC0415
                        ParquetTTLSnapshotCache,
                    )

                    ttl_delta = dt.timedelta(seconds=config.ttl or 300)
                    cache = ParquetTTLSnapshotCache.from_kwargs(ttl=ttl_delta)
                    run_expr = expr.cache(cache=cache)
                    snapshot_path = str(cache.storage.get_path(cache.calc_key(expr)))
                case _:
                    run_expr = expr

            run_params = (
                ("expr_hash", config.expr_hash),
                ("cache_type", config.cache_type),
                ("output_format", "parquet"),
                *((("ttl", config.ttl),) if config.ttl is not None else ()),
                *(
                    (("output_snapshot_path", snapshot_path),)
                    if snapshot_path is not None
                    else ()
                ),
            )

            with RunLogger.from_expr_hash(
                config.expr_hash, params_tuple=run_params
            ) as rl:
                rl.log_event("run.start", dict(run_params))
                with timed() as get_elapsed:
                    run_expr.to_parquet(os.devnull)
                rl.log_event(
                    "run.done",
                    {"elapsed_s": round(get_elapsed(), 3)},
                )
                rl.finalize(status="ok")

            status = "ok"
            detail = f"Cached at {snapshot_path}" if snapshot_path else "Run completed"
        except Exception as e:
            status = "error"
            detail = str(e)

        run_rows = _build_run_rows(config.expr_hash)
        match status:
            case "ok":
                message = f"Run completed · {detail}" if detail else "Run completed"
            case _:
                message = f"Run {status} · {detail}" if detail else f"Run {status}"
        self.app.call_from_thread(self._render_run_result, run_rows, message)

    def _render_run_result(
        self, run_rows: tuple[RunRowData, ...], message: str
    ) -> None:
        self._render_runs(run_rows)
        self.query_one("#status-bar", Static).update(f" {message}")

    # --- Navigation ---

    def _focused_widget(self) -> DataTable | VerticalScroll:
        focused = self.app.focused
        if isinstance(focused, (DataTable, VerticalScroll)):
            return focused
        return self.query_one("#catalog-table", DataTable)

    def action_scroll_left(self) -> None:
        w = self._focused_widget()
        match w:
            case DataTable():
                w.action_scroll_left()
            case VerticalScroll():
                pass

    def action_cursor_down(self) -> None:
        w = self._focused_widget()
        match w:
            case DataTable():
                w.action_cursor_down()
            case VerticalScroll():
                w.scroll_down()

    def action_cursor_up(self) -> None:
        w = self._focused_widget()
        match w:
            case DataTable():
                w.action_cursor_up()
            case VerticalScroll():
                w.scroll_up()

    def action_scroll_right(self) -> None:
        w = self._focused_widget()
        match w:
            case DataTable():
                w.action_scroll_right()
            case VerticalScroll():
                pass

    def action_focus_next_panel(self) -> None:
        self._cycle_focus(1)

    def action_focus_prev_panel(self) -> None:
        self._cycle_focus(-1)

    def _cycle_focus(self, direction: int) -> None:
        visible = tuple(
            sel for sel in self.FOCUS_CYCLE if self.query_one(sel).display is not False
        )
        if not visible:
            return
        current = self.app.focused
        current_idx = next(
            (
                i
                for i, sel in enumerate(visible)
                if current is self.query_one(sel)
                or (
                    isinstance(current, DataTable)
                    and current.parent is not None
                    and current.parent is self.query_one(sel).parent
                )
            ),
            0,
        )
        next_idx = (current_idx + direction) % len(visible)
        target = self.query_one(visible[next_idx])
        target.focus()
        self._on_panel_focus_changed(target)

    def _on_panel_focus_changed(self, widget) -> None:
        """Swap main content pane between SQL and data preview based on focus."""
        runs_table = self.query_one("#runs-table", DataTable)
        caches_table = self.query_one("#caches-table", DataTable)

        match widget:
            case _ if widget is caches_table:
                # Trigger cache row highlight to update all right panels
                if caches_table.row_count > 0:
                    row_key, _ = caches_table.coordinate_to_cell_key(
                        caches_table.cursor_coordinate
                    )
                    cache_data = self._cache_row_data.get(str(row_key.value))
                    if cache_data is not None:
                        # Update info + schema
                        self.query_one("#info-content", Static).update(
                            cache_data.info_text
                        )
                        self.query_one("#schema-in-half").display = False
                        schema_panel = self.query_one("#schema-panel")
                        schema_panel.border_title = "Schema"
                        schema_out = self.query_one("#schema-preview-table", DataTable)
                        schema_out.clear()
                        match cache_data.schema:
                            case ():
                                schema_panel.border_subtitle = ""
                            case cols:
                                schema_panel.border_subtitle = f"{len(cols)} cols"
                                for name, dtype in cols:
                                    schema_out.add_row(name, dtype)
                    # Update data preview
                    path = self._cache_row_paths.get(str(row_key.value))
                    if path and Path(path).exists():
                        self._show_inline_data(path, Path(path).stem[:16])
                        return
                self._hide_inline_data()
            case _ if widget is runs_table:
                # Show data for selected run
                if runs_table.row_count > 0:
                    row_key, _ = runs_table.coordinate_to_cell_key(
                        runs_table.cursor_coordinate
                    )
                    run_data = self._run_row_cache.get(str(row_key.value))
                    match run_data:
                        case RunRowData(output_snapshot_path=str(p)) if Path(
                            p
                        ).exists():
                            self._show_inline_data(p, f"run {run_data.run_id_display}")
                            return
                        case _:
                            pass
                self._hide_inline_data()
            case _ if widget is self.query_one("#catalog-table", DataTable):
                self._hide_inline_data()
                self._refresh_sql_for_current_entry()
            case _:
                self._hide_inline_data()

    def _refresh_sql_for_current_entry(self) -> None:
        """Re-render SQL and info for the currently selected catalog entry."""
        catalog_table = self.query_one("#catalog-table", DataTable)
        if catalog_table.row_count == 0:
            return
        row_key, _ = catalog_table.coordinate_to_cell_key(
            catalog_table.cursor_coordinate
        )
        row_data = self._row_cache.get(str(row_key.value))
        if row_data is None:
            return

        sql_preview = self.query_one("#sql-preview", Static)
        sql_panel = self.query_one("#sql-panel")
        match row_data.sqls:
            case ():
                sql_preview.update("(SQL unavailable)")
                sql_panel.border_subtitle = ""
            case ((_, engine, sql),):
                sql_preview.update(
                    Syntax(sql, "sql", theme=XorqSQLStyle, word_wrap=True)
                )
                sql_panel.border_subtitle = engine
            case sqls:
                sql_preview.update(
                    Syntax(
                        _render_sql_dag(sqls),
                        "sql",
                        theme=XorqSQLStyle,
                        word_wrap=True,
                    )
                )
                engines = sorted({engine for _, engine, _ in sqls})
                sql_panel.border_subtitle = (
                    f"{len(sqls)} queries \u00b7 {', '.join(engines)}"
                )

        self.query_one("#info-content", Static).update(row_data.info_text)

    def action_quit_app(self) -> None:
        self.app.exit()


class CatalogTUI(App):
    TITLE = "xorq catalog"
    CSS = """
    #main-split {
        height: 1fr;
    }
    #left-column {
        width: 2fr;
    }
    #right-column {
        width: 3fr;
    }
    #catalog-panel {
        height: 2fr;
        border: solid #C1F0FF;
        border-title-color: #C1F0FF;
        background: $surface;
    }
    #catalog-table {
        height: 1fr;
    }
    #runs-panel {
        height: 1fr;
        border: solid #F5CA2C;
        border-title-color: #F5CA2C;
        border-subtitle-color: #F5CA2C;
    }
    #runs-table {
        height: 1fr;
    }
    #revisions-panel {
        height: 1fr;
        border: solid #5abfb5;
        border-title-color: #5abfb5;
        border-subtitle-color: #5abfb5;
    }
    #revisions-preview-table {
        height: 1fr;
    }
    #git-log-panel {
        height: 1fr;
        border: solid #4AA8EC;
        border-title-color: #4AA8EC;
    }
    #git-log-table {
        height: 1fr;
    }
    #main-content-panel {
        height: 2fr;
    }
    #sql-panel {
        height: 1fr;
        border: solid #2BBE75;
        border-title-color: #2BBE75;
        border-subtitle-color: #2BBE75;
    }
    #sql-panel:focus-within {
        border: double #2BBE75;
    }
    #sql-preview {
        height: auto;
        padding: 1 2;
    }
    #inline-data-panel {
        height: 1fr;
        border: solid #C1F0FF;
        border-title-color: #C1F0FF;
        border-subtitle-color: #C1F0FF;
    }
    #inline-data-status {
        height: 1;
        padding: 0 1;
    }
    #inline-data-table {
        height: 1fr;
    }
    DataTable:focus {
        border: none;
    }
    #info-panel {
        height: auto;
        max-height: 6;
        border: solid #5abfb5;
        border-title-color: #5abfb5;
        padding: 0 1;
    }
    #info-content {
        height: auto;
    }
    #schema-panel {
        height: 1fr;
        border: solid #4AA8EC;
        border-title-color: #4AA8EC;
        border-subtitle-color: #4AA8EC;
    }
    #schema-split {
        height: 1fr;
    }
    #schema-in-half {
        width: 1fr;
    }
    #schema-out-half {
        width: 1fr;
    }
    #schema-in-table {
        height: 1fr;
    }
    #schema-preview-table {
        height: 1fr;
    }
    #caches-panel {
        height: 1fr;
        border: solid #F5CA2C;
        border-title-color: #F5CA2C;
        border-subtitle-color: #F5CA2C;
    }
    #caches-table {
        height: 1fr;
    }
    #data-preview-panel {
        height: 2fr;
        border: solid #C1F0FF;
        border-title-color: #C1F0FF;
        border-subtitle-color: #C1F0FF;
    }
    #data-preview-status {
        height: 1;
        padding: 0 2;
    }
    #data-preview-table {
        height: 1fr;
    }
    #profiles-panel {
        height: 1fr;
        border: solid #2BBE75;
        border-title-color: #2BBE75;
    }
    #profiles-table {
        height: 1fr;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $surface;
    }
    RunDataScreen {
        background: $surface;
    }
    RunDataScreen #run-data-container {
        height: 1fr;
    }
    RunDataScreen #run-data-status {
        height: 1;
        padding: 0 2;
        background: $panel;
    }
    RunDataScreen #run-data-table {
        height: 1fr;
    }
    RunOptionsScreen {
        align: center middle;
        background: rgba(5, 24, 26, 0.85);
    }
    RunOptionsScreen #run-options-container {
        width: 40;
        height: auto;
        max-height: 14;
        border: solid #5abfb5;
        background: $surface;
        padding: 0 1;
    }
    RunOptionsScreen #run-options-title {
        height: 1;
        color: #5abfb5;
    }
    RunOptionsScreen #cache-strategy {
        height: auto;
        margin: 0;
    }
    RunOptionsScreen #ttl-row {
        height: 3;
        padding: 0 1;
    }
    RunOptionsScreen #ttl-label {
        width: auto;
    }
    RunOptionsScreen #ttl-input {
        width: 10;
    }
    """

    def __init__(self, make_catalog, refresh_interval=DEFAULT_REFRESH_INTERVAL):
        super().__init__()
        self._catalog = None
        self._make_catalog = make_catalog
        self._refresh_interval = refresh_interval
        self.register_theme(XORQ_DARK)
        self.theme = "xorq-dark"

    def on_mount(self) -> None:
        self.push_screen(CatalogScreen(refresh_interval=self._refresh_interval))
        self._load_catalog()

    @work(thread=True)
    def _load_catalog(self) -> None:
        catalog = self._make_catalog()
        self.app.call_from_thread(self._set_catalog, catalog)

    def _set_catalog(self, catalog) -> None:
        self._catalog = catalog
        self.screen._do_refresh()
