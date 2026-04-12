import math
import re
import subprocess
import threading
from datetime import datetime
from functools import cache, cached_property
from pathlib import Path
from typing import Literal

from attr import evolve, field, frozen
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
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.suggester import SuggestFromList
from textual.theme import Theme
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Static,
    Tree,
)

from xorq.caching.storage import resolve_parquet_cache_path
from xorq.catalog.catalog import CatalogEntry
from xorq.common.utils.caching_utils import CacheKey
from xorq.common.utils.logging_utils import Runs


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

KIND_ORDER = ("source", "expr", "unbound_expr", "composed")

KIND_ICONS = {
    "source": "⊞",
    "expr": "⊕",
    "unbound_expr": "⊘",
    "composed": "⊛",
}

KIND_COLORS = {
    "source": "#C1F0FF",
    "expr": "#2BBE75",
    "unbound_expr": "#F5CA2C",
    "composed": "#4AA8EC",
}

SCHEMA_PREVIEW_COLUMNS = ("NAME", "TYPE")

REVISION_COLUMNS = ("STATUS", "HASH", "COLUMNS", "CACHED", "DATE")

GIT_LOG_COLUMNS = ("HASH", "DATE", "MESSAGE")

RUN_COLUMNS = ("STATUS", "RUN ID", "CACHE", "DURATION", "FORMAT", "DATE")


def _styled_branch_label(kind: str, count: int) -> Text:
    icon = KIND_ICONS.get(kind, "·")
    color = KIND_COLORS.get(kind, "#C1F0FF")
    label = Text()
    label.append(f"{icon} ", style=f"bold {color}")
    label.append(f"{kind} ", style=f"bold {color}")
    label.append(f"({count})", style=f"dim {color}")
    return label


def _format_cached(value: bool | None) -> str:
    match value:
        case True:
            return "●"
        case False:
            return "○"
        case _:
            return "—"


def get_cache_keys_paths(cache_keys: tuple[CacheKey, ...]) -> tuple[str, ...]:
    return tuple(
        str(resolve_parquet_cache_path(ck.relative_path, ck.key)) for ck in cache_keys
    )


@frozen
class CatalogRowData:
    entry: CatalogEntry = field(repr=False)
    aliases: tuple[str, ...] = field(factory=tuple, validator=instance_of(tuple))

    @property
    def cached(self) -> bool | None:
        if cache_keys_paths := get_cache_keys_paths(
            self.entry.parquet_snapshot_cache_keys
        ):
            return all(Path(p).exists() for p in cache_keys_paths)
        return None

    @property
    def kind(self) -> str:
        return str(self.entry.kind)

    @property
    def hash(self) -> str:
        return self.entry.name

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

    @property
    def cached_display(self) -> str:
        return _format_cached(self.cached)

    @cached_property
    def sqls(self) -> tuple[tuple[str, str, str], ...]:
        """((name, engine, sql), ...) for all queries in the expression plan."""
        return self.entry.metadata.sql_queries

    @cached_property
    def lineage_text(self) -> str:
        lineage = self.entry.metadata.lineage
        if not lineage:
            return "(empty)"
        labels = [n["label"] for n in lineage.nodes]
        return " → ".join(labels) if labels else "(empty)"

    @cached_property
    def cache_info_text(self) -> str:
        paths = get_cache_keys_paths(self.entry.parquet_snapshot_cache_keys)
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

    @cached_property
    def tree_label(self) -> str:
        """Label for tree leaf node: 'alias — hash[:12]' or 'hash[:12]'."""
        if self.aliases_display:
            return f"{self.aliases_display} — {self.hash[:12]}"
        return self.hash[:12]

    @property
    def row_key(self) -> str:
        return self.hash


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


@frozen
class RunRowData:
    run_id: str = field(validator=instance_of(str))
    status: str = field(validator=instance_of(str))
    cache_type: str = field(default="", validator=instance_of(str))
    duration_ms: int | None = field(default=None, validator=optional(instance_of(int)))
    output_format: str = field(default="", validator=instance_of(str))
    date: str = field(default="", validator=instance_of(str))
    output_path: str | None = field(default=None, validator=optional(instance_of(str)))

    @property
    def status_display(self) -> str:
        match self.status:
            case "ok":
                return "✓"
            case "error":
                return "✗"
            case "running":
                return "…"
            case _:
                return self.status

    @property
    def duration_display(self) -> str:
        if self.duration_ms is None:
            return "—"
        if self.duration_ms < 1000:
            return f"{self.duration_ms}ms"
        secs = self.duration_ms / 1000
        if secs < 60:
            return f"{secs:.1f}s"
        return f"{secs / 60:.1f}m"

    @property
    def row(self) -> tuple[str, ...]:
        return (
            self.status_display,
            self.run_id[:8],
            self.cache_type or "none",
            self.duration_display,
            self.output_format or "—",
            self.date,
        )


def _compute_duration(meta: dict) -> int | None:
    started = meta.get("started_at")
    completed = meta.get("completed_at")
    if not started or not completed:
        return None
    try:
        s = datetime.fromisoformat(started)
        c = datetime.fromisoformat(completed)
        return int((c - s).total_seconds() * 1000)
    except (ValueError, TypeError):
        return None


def _format_run_date(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return iso_str


def _build_run_rows(
    expr_hash: str, aliases: tuple[str, ...] = ()
) -> tuple[RunRowData, ...]:
    from xorq.common.utils.logging_utils import get_xorq_runs_dir  # noqa: PLC0415

    runs_dir = get_xorq_runs_dir()
    # catalog run logs under alias names, xorq run logs under expr hash
    lookup_keys = (expr_hash, *aliases)
    seen_run_ids = set()
    rows = []
    for key in lookup_keys:
        expr_dir = runs_dir / key
        for run in Runs(expr_dir=expr_dir).runs:
            if run.run_id in seen_run_ids:
                continue
            seen_run_ids.add(run.run_id)
            meta = run.read_meta()
            if meta is None:
                continue
            rows.append(
                RunRowData(
                    run_id=meta.get("run_id", run.run_id),
                    status=meta.get("status", "unknown"),
                    cache_type=meta.get("cache_strategy", ""),
                    duration_ms=_compute_duration(meta),
                    output_format=meta.get("output_format", ""),
                    date=_format_run_date(meta.get("completed_at", "")),
                    output_path=meta.get("output_path")
                    or meta.get("output_snapshot_path"),
                )
            )
    rows.sort(key=lambda r: r.date, reverse=True)
    return tuple(rows)


VIEW_LIMIT = 50_000

SNAPSHOT_CACHE_EXPR = "xo.ParquetSnapshotCache.from_kwargs()"


def _wrap_with_cache(code: str | None) -> str:
    """Wrap a code expression so the result is snapshot-cached."""
    cache = SNAPSHOT_CACHE_EXPR
    if not code:
        return f"source.cache({cache})"
    return f"({code}).cache({cache})"


VERB_TEMPLATES = {
    "filter": "source.filter({input})",
    "mutate": "source.mutate({input})",
    "select": "source.select({input})",
    "order_by": "source.order_by({input})",
    "drop": "source.drop({input})",
    "agg": "source.group_by({group}).agg({input})",
}


def build_code(verb: str, user_input: str, group: str = "") -> str:
    if verb == "freeform":
        return user_input
    if verb == "agg":
        return VERB_TEMPLATES[verb].format(group=group, input=user_input)
    return VERB_TEMPLATES[verb].format(input=user_input)


@frozen
class ExprStep:
    """A single user-applied Ibis operation."""

    verb: str = field(validator=instance_of(str))
    user_input: str = field(validator=instance_of(str))
    code: str = field(validator=instance_of(str))


@frozen
class ExprStack:
    """Immutable operation stack with undo/redo cursor."""

    base_expr: object = field(repr=False)
    steps: tuple[ExprStep, ...] = field(factory=tuple)
    cursor: int = field(default=0, validator=instance_of(int))

    def push(self, step: ExprStep) -> "ExprStack":
        """Apply new step, discard any steps after cursor (fork)."""
        return evolve(
            self,
            steps=self.steps[: self.cursor] + (step,),
            cursor=self.cursor + 1,
        )

    def undo(self) -> "ExprStack":
        return evolve(self, cursor=max(0, self.cursor - 1))

    def redo(self) -> "ExprStack":
        return evolve(self, cursor=min(len(self.steps), self.cursor + 1))

    @property
    def can_undo(self) -> bool:
        return self.cursor > 0

    @property
    def can_redo(self) -> bool:
        return self.cursor < len(self.steps)

    def current_expr(self):
        """Replay active steps onto base via _eval_code."""
        from xorq.catalog.bind import _eval_code  # noqa: PLC0415

        expr = self.base_expr
        for step in self.steps[: self.cursor]:
            expr = _eval_code(step.code, expr)
        return expr

    @property
    def current_code(self) -> str:
        """Single evaluable expression chaining all active steps.

        Each step's code starts with ``source.verb(...)``.  For steps after
        the first, the leading ``source`` is replaced with the accumulated
        expression so the result is one chained call that ``_eval_code`` can
        evaluate in a single ``eval()``.
        """
        if self.cursor == 0:
            return ""
        steps = self.steps[: self.cursor]
        result = steps[0].code
        for step in steps[1:]:
            result = step.code.replace("source", f"({result})", 1)
        return result


def _entry_info(entry: CatalogEntry) -> tuple[int | None, bool | None]:
    cache_keys_paths = get_cache_keys_paths(entry.parquet_snapshot_cache_keys)
    cached = (
        all(Path(p).exists() for p in cache_keys_paths) if cache_keys_paths else None
    )
    return len(entry.columns), cached


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

    parts = tuple(
        segment
        for i, name in enumerate(order)
        for engine, sql in (name_to_sql[name],)
        for label in (("main" if name == "main" else name[:12]),)
        for segment in (
            (f"-- [{label}] ({engine})\n{sql}", "  ↓")
            if i < len(order) - 1
            else (f"-- [{label}] ({engine})\n{sql}",)
        )
    )
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


class CatalogScreen(Screen):
    BINDINGS = (
        ("q", "quit_app", "Quit"),
        ("ctrl+c", "quit_app", "Quit"),
        ("h", "tree_collapse", "Collapse"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("l", "tree_expand", "Expand"),
        ("tab", "focus_next_panel", "Next"),
        ("shift+tab", "focus_prev_panel", "Prev"),
        ("1", "view_sql", "SQL"),
        ("2", "view_data", "Data"),
        ("e", "open_data_view", "Explore"),
        ("r", "toggle_runs", "Runs"),
        ("v", "toggle_revisions", "Revisions"),
        ("g", "toggle_git_log", "Git Log"),
    )

    FOCUS_CYCLE = (
        "#catalog-tree",
        "#runs-table",
        "#sql-panel",
        "#data-preview-panel",
        "#schema-preview-table",
    )

    def __init__(self, refresh_interval=DEFAULT_REFRESH_INTERVAL):
        super().__init__()
        self._refresh_interval = refresh_interval
        self._row_cache: dict[str, CatalogRowData] = {}
        self._new_keys: set[str] = set()
        self._runs_visible = False
        self._git_log_visible = False
        self._git_log_loaded = False
        self._refresh_lock = threading.Lock()
        self._active_view: Literal["sql", "data"] = "sql"
        self._data_preview_hash: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-split"):
            with Vertical(id="left-column"):
                with Vertical(id="catalog-panel"):
                    yield Tree("Catalog", id="catalog-tree")
                with Vertical(id="runs-panel"):
                    yield DataTable(id="runs-table")
                with Vertical(id="revisions-panel"):
                    yield DataTable(id="revisions-preview-table")
                with Vertical(id="git-log-panel"):
                    yield DataTable(id="git-log-table")
            with Vertical(id="right-column"):
                with VerticalScroll(id="sql-panel"):
                    yield Static("", id="sql-preview")
                with Vertical(id="data-preview-panel"):
                    yield Static("", id="data-preview-status")
                    yield DataTable(id="data-preview-table")
                with Vertical(id="info-panel"):
                    yield Static("", id="info-content")
                with Vertical(id="schema-panel"):
                    with Horizontal(id="schema-split"):
                        with Vertical(id="schema-in-half"):
                            yield DataTable(id="schema-in-table")
                        with Vertical(id="schema-out-half"):
                            yield DataTable(id="schema-preview-table")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        tree = self.query_one("#catalog-tree", Tree)
        tree.show_root = False
        tree.guide_depth = 3

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

        self.query_one("#catalog-panel").border_title = "Expressions"
        self.query_one("#schema-panel").border_title = "Schema"
        self.query_one("#schema-in-half").display = False
        self.query_one("#sql-panel").border_title = "SQL"
        self.query_one("#sql-preview", Static).update(
            Text("← Select an expression to view its SQL", style="dim")
        )
        self.query_one("#info-panel").border_title = "Info"
        self.query_one("#info-content", Static).update(
            Text("← Select an expression", style="dim")
        )
        runs_panel = self.query_one("#runs-panel")
        runs_panel.border_title = "Runs"
        runs_panel.display = False

        self.query_one("#revisions-panel").border_title = "Revisions"
        self.query_one("#revisions-panel").display = False

        git_log_panel = self.query_one("#git-log-panel")
        git_log_panel.border_title = "Git Log"
        git_log_panel.display = False

        data_panel = self.query_one("#data-preview-panel")
        data_panel.border_title = "Data Preview"
        data_panel.display = False

        self.query_one("#status-bar", Static).update(" Loading catalog...")

        self.set_interval(self._refresh_interval, self._do_refresh)

    # --- Tree node selection ---

    @on(Tree.NodeHighlighted, "#catalog-tree")
    def _on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
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

        # Branch nodes (kind groupings) have children; only leaf nodes are entries
        entry_hash = event.node.data
        row_data = (
            self._row_cache.get(entry_hash)
            if not event.node.children and entry_hash is not None
            else None
        )
        if row_data is None:
            sql_preview.update("")
            info_content.update("")
            self.query_one("#schema-in-half").display = False
            self.query_one("#revisions-panel").border_title = "Revisions"
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

        # SQL preview
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
                    f"{len(sqls)} queries · {', '.join(engines)}"
                )

        # Info panel
        info_content.update(row_data.info_text)

        # Revisions preview
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

        # Runs preview
        if self._runs_visible:
            self._load_runs_preview(row_data.hash, row_data.aliases)

        # Update data preview if data view is active
        if self._active_view == "data":
            self._refresh_data_preview(row_data)

    def _tree_entry_hashes(self) -> set[str]:
        """Return set of entry hashes currently in the tree."""
        tree = self.query_one("#catalog-tree", Tree)
        return {
            node.data
            for branch in tree.root.children
            for node in branch.children
            if node.data is not None
        }

    # --- Refresh ---

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

        # Age out: keys that were pink last cycle turn green now
        prev_new = self._new_keys
        self._new_keys = set()

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

        # Track new keys for pink highlighting (skip first load — everything is new)
        if cached_rows:
            self._new_keys = set(new_keys)

        if prev_new:
            self.app.call_from_thread(self._settle_new_labels, prev_new)

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

        stamp = datetime.now().strftime("%H:%M:%S")
        self.app.call_from_thread(self._render_status, stamp, repo_path)

    def _render_refresh(self, repo_path, cached_rows) -> None:
        with self.app.batch_update():
            catalog_name = Path(repo_path).name
            self.query_one(
                "#catalog-panel"
            ).border_title = f"Expressions — {catalog_name}"

            tree = self.query_one("#catalog-tree", Tree)
            saved_line = tree.cursor_line
            tree.clear()

            # Group rows by kind
            groups: dict[str, list[CatalogRowData]] = {}
            for row_data in cached_rows:
                groups.setdefault(row_data.kind, []).append(row_data)

            # Add branches in KIND_ORDER, then any remaining kinds
            for kind in (*KIND_ORDER, *(k for k in groups if k not in KIND_ORDER)):
                if kind not in groups:
                    continue
                entries = groups[kind]
                branch = tree.root.add(
                    _styled_branch_label(kind, len(entries)), data=kind
                )
                branch.expand()
                for row_data in entries:
                    branch.add_leaf(
                        self._styled_leaf_label(row_data), data=row_data.row_key
                    )

            # Restore approximate cursor position
            total = sum(1 + len(b.children) for b in tree.root.children)
            if total > 0:
                tree.cursor_line = min(saved_line, total - 1)

    def _render_catalog_row(self, row_data) -> None:
        with self.app.batch_update():
            tree = self.query_one("#catalog-tree", Tree)
            kind = row_data.kind

            # Find or create the kind branch
            branch = None
            for child in tree.root.children:
                if child.data == kind:
                    branch = child
                    break

            if branch is None:
                branch = tree.root.add(_styled_branch_label(kind, 1), data=kind)
                branch.expand()
            else:
                count = len(branch.children) + 1
                branch.set_label(_styled_branch_label(kind, count))

            branch.add_leaf(self._styled_leaf_label(row_data), data=row_data.row_key)

    def _styled_leaf_label(self, row_data) -> Text:
        is_new = row_data.row_key in self._new_keys
        cache_icon = _format_cached(row_data.cached)
        ncols = len(row_data.schema_out)

        label = Text()
        if is_new:
            label.append(f"{cache_icon} ", style="bold #FF69B4")
            if row_data.aliases_display:
                label.append(row_data.aliases_display, style="bold #FF69B4")
                label.append(f" — {row_data.hash[:12]}", style="#FF69B4")
            else:
                label.append(row_data.hash[:12], style="bold #FF69B4")
            label.append(f"  {ncols} cols", style="dim #FF69B4")
        else:
            cache_color = {"●": "#2BBE75", "○": "#F5CA2C"}.get(cache_icon, "dim")
            label.append(f"{cache_icon} ", style=cache_color)
            if row_data.aliases_display:
                label.append(row_data.aliases_display, style="bold #C1F0FF")
                label.append(f" — {row_data.hash[:12]}", style="dim #5abfb5")
            else:
                label.append(row_data.hash[:12], style="#C1F0FF")
            label.append(f"  {ncols} cols", style="dim")
        return label

    def _settle_new_labels(self, keys: set[str]) -> None:
        tree = self.query_one("#catalog-tree", Tree)
        for branch in tree.root.children:
            for leaf in branch.children:
                if leaf.data in keys:
                    row_data = self._row_cache.get(leaf.data)
                    if row_data:
                        leaf.set_label(self._styled_leaf_label(row_data))

    def _render_status(self, stamp, repo_path) -> None:
        rows = self._row_cache.values()
        count = len(self._row_cache)
        kind_counts = {}
        cached_count = 0
        for r in rows:
            kind_counts[r.kind] = kind_counts.get(r.kind, 0) + 1
            if r.cached:
                cached_count += 1
        kinds_str = ", ".join(
            f"{c} {k}" for k, c in sorted(kind_counts.items(), key=lambda x: -x[1])
        )
        parts = [f" {count} entries"]
        if kinds_str:
            parts[0] += f" ({kinds_str})"
        if cached_count:
            parts.append(f"{cached_count} cached")
        parts.append(str(repo_path))
        parts.append(stamp)
        self.query_one("#status-bar", Static).update(" · ".join(parts))

    # --- Toggle: Runs ---

    def action_toggle_runs(self) -> None:
        self._runs_visible = not self._runs_visible
        self.query_one("#runs-panel").display = self._runs_visible
        if self._runs_visible:
            tree = self.query_one("#catalog-tree", Tree)
            node = tree.cursor_node
            if node and not node.children and node.data:
                row_data = self._row_cache.get(node.data)
                aliases = row_data.aliases if row_data else ()
                self._load_runs_preview(node.data, aliases)

    @work(thread=True, exit_on_error=False)
    def _load_runs_preview(self, expr_hash, aliases=()) -> None:
        run_rows = _build_run_rows(expr_hash, aliases)
        self.app.call_from_thread(self._render_runs_preview, run_rows, expr_hash)

    def _render_runs_preview(self, run_rows, expr_hash) -> None:
        with self.app.batch_update():
            runs_table = self.query_one("#runs-table", DataTable)
            runs_table.clear()
            runs_panel = self.query_one("#runs-panel")
            if run_rows:
                for i, row_data in enumerate(run_rows):
                    runs_table.add_row(*row_data.row, key=str(i))
                runs_panel.border_title = f"Runs — {len(run_rows)} runs"
                runs_panel.border_subtitle = expr_hash[:12]
            else:
                runs_panel.border_title = "Runs — no runs"
                runs_panel.border_subtitle = expr_hash[:12]

    def _get_selected_run(self) -> RunRowData | None:
        runs_table = self.query_one("#runs-table", DataTable)
        if runs_table.row_count == 0:
            return None
        row_key, _ = runs_table.coordinate_to_cell_key(runs_table.cursor_coordinate)
        try:
            idx = int(row_key.value)
        except (ValueError, TypeError):
            return None
        tree = self.query_one("#catalog-tree", Tree)
        node = tree.cursor_node
        if not node or node.children or not node.data:
            return None
        row_data = self._row_cache.get(node.data)
        aliases = row_data.aliases if row_data else ()
        run_rows = _build_run_rows(node.data, aliases)
        if 0 <= idx < len(run_rows):
            return run_rows[idx]
        return None

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

    # --- View switching (1/2) ---

    def _set_active_view(self, view: Literal["sql", "data"]) -> None:
        self._active_view = view
        self.query_one("#sql-panel").display = view == "sql"
        self.query_one("#data-preview-panel").display = view == "data"

        if view == "data":
            tree = self.query_one("#catalog-tree", Tree)
            node = tree.cursor_node
            if node is not None and node.data is not None:
                row_data = self._row_cache.get(node.data)
                if row_data is not None:
                    self._refresh_data_preview(row_data)
        else:
            self._data_preview_hash = None

    def action_view_sql(self) -> None:
        self._set_active_view("sql")

    def action_view_data(self) -> None:
        self._set_active_view("data")

    def _refresh_data_preview(self, row_data) -> None:
        entry_hash = row_data.row_key
        if self._data_preview_hash == entry_hash:
            return
        self._data_preview_hash = entry_hash
        if row_data.cached is True:
            self.query_one("#data-preview-status", Static).update(
                " Loading data preview..."
            )
            self.query_one("#data-preview-table", DataTable).loading = True
            self._load_data_preview(row_data.entry)
        else:
            self.query_one("#data-preview-status", Static).update(
                " uncached — run to materialize"
            )
            dt = self.query_one("#data-preview-table", DataTable)
            dt.clear(columns=True)
            dt.loading = False

    # --- Toggle: Revisions (v) ---

    def action_toggle_revisions(self) -> None:
        panel = self.query_one("#revisions-panel")
        panel.display = not panel.display

    # --- Data Preview (worker) ---

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

    # --- Navigation ---

    def action_tree_collapse(self) -> None:
        """h: collapse branch in tree, scroll left in DataTable."""
        focused = self.app.focused
        tree = self.query_one("#catalog-tree", Tree)
        if focused is tree:
            node = tree.cursor_node
            if node is not None:
                if node.children and node.is_expanded:
                    node.collapse()
                elif node.parent is not None and node.parent is not tree.root:
                    tree.select_node(node.parent)
                    node.parent.collapse()
        elif isinstance(focused, DataTable):
            focused.action_scroll_left()

    def action_cursor_down(self) -> None:
        focused = self.app.focused
        tree = self.query_one("#catalog-tree", Tree)
        if focused is tree:
            tree.action_cursor_down()
        elif isinstance(focused, DataTable):
            focused.action_cursor_down()
        elif isinstance(focused, VerticalScroll):
            focused.scroll_down()

    def action_cursor_up(self) -> None:
        focused = self.app.focused
        tree = self.query_one("#catalog-tree", Tree)
        if focused is tree:
            tree.action_cursor_up()
        elif isinstance(focused, DataTable):
            focused.action_cursor_up()
        elif isinstance(focused, VerticalScroll):
            focused.scroll_up()

    def action_tree_expand(self) -> None:
        """l: expand branch in tree, scroll right in DataTable."""
        focused = self.app.focused
        tree = self.query_one("#catalog-tree", Tree)
        if focused is tree:
            node = tree.cursor_node
            if node is not None:
                if node.children and not node.is_expanded:
                    node.expand()
                elif node.children and node.is_expanded:
                    first_child = node.children[0]
                    tree.select_node(first_child)
        elif isinstance(focused, DataTable):
            focused.action_scroll_right()

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
                    isinstance(current, (DataTable, Tree))
                    and current.parent is not None
                    and current.parent is self.query_one(sel).parent
                )
            ),
            0,
        )
        next_idx = (current_idx + direction) % len(visible)
        self.query_one(visible[next_idx]).focus()

    def action_open_data_view(self) -> None:
        focused = self.app.focused
        runs_table = self.query_one("#runs-table", DataTable)
        if focused is runs_table:
            self._open_run_data_view()
            return
        tree = self.query_one("#catalog-tree", Tree)
        node = tree.cursor_node
        if node is None or node.children:
            return
        row_data = self._row_cache.get(node.data)
        if row_data is None or row_data.kind == "unbound_expr":
            return
        self.app.push_screen(DataViewScreen(entry=row_data.entry, row_data=row_data))

    def _open_run_data_view(self) -> None:
        run_row = self._get_selected_run()
        if run_row is None:
            return
        parquet_path = run_row.output_path
        if not parquet_path or parquet_path == "-":
            # Fallback: try cache paths from the selected entry
            tree = self.query_one("#catalog-tree", Tree)
            node = tree.cursor_node
            if node and not node.children and node.data:
                row_data = self._row_cache.get(node.data)
                if row_data:
                    paths = get_cache_keys_paths(
                        row_data.entry.parquet_snapshot_cache_keys
                    )
                    parquet_path = next((p for p in paths if Path(p).exists()), None)
        if not parquet_path or not Path(parquet_path).exists():
            self.notify("No parquet file available for this run", severity="warning")
            return
        self.app.push_screen(RunDataScreen(parquet_path=parquet_path, run_row=run_row))

    def action_quit_app(self) -> None:
        self.app.exit()


class DataViewScreen(Screen):
    """Full-screen data viewer with interactive expression composition.

    Every user action (filter, mutate, select, sort, aggregate) is a raw Ibis
    API call pushed onto an undo/redo ExprStack.
    """

    BINDINGS = (
        Binding("escape", "cancel_or_back", "Back"),
        Binding("q", "cancel_or_back", "Back", show=False),
        Binding("h", "cursor_left", "Col ←", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("l", "cursor_right", "Col →", show=False),
        Binding("g", "scroll_top", "Top", show=False),
        Binding("shift+g", "scroll_bottom", "Bottom", show=False),
        Binding("[", "sort_desc", "Sort ↓", show=False),
        Binding("]", "sort_asc", "Sort ↑", show=False),
        Binding("d", "drop_column", "Drop"),
        Binding("u", "undo", "Undo"),
        Binding("ctrl+r", "redo", "Redo"),
        Binding("e", "toggle_stack_browser", "Stack"),
        Binding("w", "persist", "Save"),
        Binding("f", "verb_filter", "Filter"),
        Binding("=", "verb_mutate", "Mutate"),
        Binding("-", "verb_select", "Select"),
        Binding("#", "verb_agg", "Agg"),
        Binding(":", "verb_freeform", "Free"),
    )

    def __init__(self, entry, row_data):
        super().__init__()
        self._entry = entry
        self._row_data = row_data
        self._stack = None
        self._df = None
        self._cursor_column_index = 0
        self._stack_browser_visible = False
        self._command_verb = None
        self._agg_group = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="data-view-status")
        with Horizontal(id="data-view-split"):
            yield DataTable(id="data-view-table")
            with Vertical(id="stack-browser-panel"):
                yield Static("", id="stack-browser-content")
        yield Input(id="command-input", placeholder="")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#data-view-table", DataTable)
        table.cursor_type = "cell"
        table.zebra_stripes = True
        table.loading = True

        stack_panel = self.query_one("#stack-browser-panel")
        stack_panel.border_title = "Expression Stack"
        stack_panel.display = False

        cmd_input = self.query_one("#command-input", Input)
        cmd_input.display = False

        label = self._row_data.aliases_display or self._row_data.hash[:12]
        self.query_one("#data-view-status", Static).update(f" Loading {label}...")
        self._load_data()

    def _catalog_run_cmd(self, code=None) -> list[str]:
        """Build the xorq catalog run subprocess command."""
        catalog = self.app._catalog
        entry_name = (
            self._row_data.aliases[0] if self._row_data.aliases else self._entry.name
        )
        cmd = [
            "xorq",
            "catalog",
            "--path",
            str(catalog.repo_path),
            "run",
            entry_name,
            "--limit",
            str(VIEW_LIMIT),
            "-o",
            "-",
            "-f",
            "arrow",
        ]
        if code:
            cmd.extend(["-c", code])
        return cmd

    def _run_catalog_subprocess(self, code=None):
        """Run xorq catalog run and return a pandas DataFrame."""
        import pyarrow as pa  # noqa: PLC0415

        cmd = self._catalog_run_cmd(code)
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode().strip())
        reader = pa.ipc.open_stream(proc.stdout)
        return reader.read_pandas()

    @work(thread=True, exit_on_error=False)
    def _load_data(self) -> None:
        try:
            self._stack = ExprStack(base_expr=self._entry)
            df = self._run_catalog_subprocess(_wrap_with_cache(None))
            self.app.call_from_thread(self._on_data_loaded, df)
        except Exception as e:
            self.app.call_from_thread(self._render_error, str(e))

    def _on_data_loaded(self, df) -> None:
        self._df = df
        self._cursor_column_index = 0
        self._render_table()
        self._update_command_suggester()

    def _render_table(self) -> None:
        df = self._df
        if df is None:
            return
        with self.app.batch_update():
            table = self.query_one("#data-view-table", DataTable)
            table.clear(columns=True)
            table.loading = False
            for col in df.columns:
                table.add_column(str(col), key=str(col))
            for i, row in enumerate(df.itertuples(index=False)):
                table.add_row(
                    *(
                        "—"
                        if isinstance(v, float) and math.isnan(v)
                        else str(round(v, 2))
                        if isinstance(v, float)
                        else str(v)
                        for v in row
                    ),
                    key=str(i),
                )
            table.cursor_type = "cell"
            self._update_status_bar()

    def _update_status_bar(self) -> None:
        df = self._df
        if df is None:
            return
        label = self._row_data.aliases_display or self._row_data.hash[:12]
        stack = self._stack
        step_info = ""
        if stack and stack.cursor > 0:
            step = stack.steps[stack.cursor - 1]
            step_info = f" | step {stack.cursor}/{len(stack.steps)} {step.verb}"
        col_info = ""
        cols = df.columns
        idx = self._cursor_column_index
        if 0 <= idx < len(cols):
            col_info = f" | [{cols[idx]}]"
        self.query_one("#data-view-status", Static).update(
            f" {label} \u2014 {len(df)} rows \u00d7 {len(cols)} cols{col_info}{step_info}"
        )

    def _render_error(self, message) -> None:
        self.query_one("#data-view-status", Static).update(f" Error: {message}")
        self.query_one("#data-view-table", DataTable).loading = False

    _IBIS_METHOD_SUGGESTIONS = (
        "filter",
        "mutate",
        "select",
        "order_by",
        "group_by",
        "agg",
        "distinct",
        "head",
        "limit",
        "join",
        "drop",
    )

    def _update_command_suggester(self) -> None:
        """Update tab-completion suggestions from current expression columns."""
        cols = tuple(self._df.columns) if self._df is not None else ()
        self.query_one("#command-input", Input).suggester = SuggestFromList(
            cols + self._IBIS_METHOD_SUGGESTIONS, case_sensitive=False
        )

    # --- Stack operations ---

    def _push_step(self, verb: str, user_input: str, code: str) -> None:
        """Push a step and re-execute in background."""
        step = ExprStep(verb=verb, user_input=user_input, code=code)
        self._stack = self._stack.push(step)
        self._execute_current()

    @work(thread=True, exit_on_error=False)
    def _execute_current(self) -> None:
        """Evaluate current stack expression via subprocess."""
        try:
            stack = self._stack
            code = stack.current_code or None
            df = self._run_catalog_subprocess(_wrap_with_cache(code))
            self.app.call_from_thread(self._on_stack_executed, df)
        except Exception as e:
            self._stack = stack.undo()
            self.app.call_from_thread(self._show_command_error, str(e))

    def _on_stack_executed(self, df) -> None:
        self._df = df
        self._cursor_column_index = 0
        self._render_table()
        self._update_command_suggester()
        self._render_stack_browser()

    def _show_command_error(self, message) -> None:
        cmd = self.query_one("#command-input", Input)
        cmd.display = True
        cmd.value = f"Error: {message}"
        cmd.add_class("error")

    # --- Command input ---

    def _open_command_input(self, verb: str) -> None:
        """Show the command input docked at bottom with verb prompt."""
        self._command_verb = verb
        cmd = self.query_one("#command-input", Input)
        cmd.remove_class("error")
        cmd.value = ""
        prompt = verb if verb != "freeform" else ":"
        cmd.placeholder = (
            f"{prompt}\u25b8 type expression, Tab to complete, Enter to apply"
        )
        cmd.border_title = f"{prompt}\u25b8"
        cmd.display = True
        cmd.focus()

    @on(Input.Submitted, "#command-input")
    def _on_command_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        cmd = self.query_one("#command-input", Input)

        verb = self._command_verb

        # Save accepts empty input (no alias)
        if verb == "save":
            alias = user_input or None
            cmd.display = False
            self._command_verb = None
            self.query_one("#data-view-table", DataTable).focus()
            self._do_persist(alias)
            return

        if not user_input:
            cmd.display = False
            self._command_verb = None
            self._agg_group = None
            self.query_one("#data-view-table", DataTable).focus()
            return

        if verb is None:
            cmd.display = False
            self.query_one("#data-view-table", DataTable).focus()
            return

        # Two-phase aggregate: first group_by, then agg
        if verb == "agg_group":
            self._agg_group = user_input
            self._command_verb = "agg"
            cmd.value = ""
            cmd.placeholder = (
                "agg\u25b8 aggregation expressions (e.g. avg=source.amount.mean())"
            )
            cmd.border_title = "agg\u25b8"
            return

        try:
            if verb == "agg":
                group = self._agg_group or ""
                code = build_code(verb, user_input, group=group)
                display_input = f"group_by({group}).agg({user_input})"
            else:
                code = build_code(verb, user_input)
                display_input = user_input
        except Exception as e:
            cmd.value = f"Error building code: {e}"
            cmd.add_class("error")
            return

        cmd.display = False
        self._command_verb = None
        self._agg_group = None
        self.query_one("#data-view-table", DataTable).focus()
        self._push_step(verb, display_input, code)

    # --- Verb actions ---

    def action_verb_filter(self) -> None:
        if self._stack is None:
            return
        self._open_command_input("filter")

    def action_verb_mutate(self) -> None:
        if self._stack is None:
            return
        self._open_command_input("mutate")

    def action_verb_select(self) -> None:
        if self._stack is None:
            return
        self._open_command_input("select")

    def action_verb_agg(self) -> None:
        if self._stack is None:
            return
        self._agg_group = None
        self._command_verb = "agg_group"
        cmd = self.query_one("#command-input", Input)
        cmd.remove_class("error")
        cmd.value = ""
        cmd.placeholder = (
            'group_by\u25b8 column names to group by (e.g. "category", "region")'
        )
        cmd.border_title = "group_by\u25b8"
        cmd.display = True
        cmd.focus()

    def action_verb_freeform(self) -> None:
        if self._stack is None:
            return
        self._open_command_input("freeform")

    # --- Instant actions (no input required) ---

    def action_sort_asc(self) -> None:
        if self._stack is None or self._df is None:
            return
        col = self._df.columns[self._cursor_column_index]
        code = f'source.order_by("{col}")'
        self._push_step("order_by", f'"{col}"', code)

    def action_sort_desc(self) -> None:
        if self._stack is None or self._df is None:
            return
        col = self._df.columns[self._cursor_column_index]
        code = f'source.order_by(ibis.desc("{col}"))'
        self._push_step("order_by", f'ibis.desc("{col}")', code)

    def action_drop_column(self) -> None:
        if self._stack is None or self._df is None:
            return
        col = self._df.columns[self._cursor_column_index]
        code = f'source.drop("{col}")'
        self._push_step("drop", f'"{col}"', code)

    # --- Undo / Redo ---

    def action_undo(self) -> None:
        if self._stack is None or not self._stack.can_undo:
            return
        self._stack = self._stack.undo()
        self._execute_current()

    def action_redo(self) -> None:
        if self._stack is None or not self._stack.can_redo:
            return
        self._stack = self._stack.redo()
        self._execute_current()

    # --- Navigation ---

    def action_cancel_or_back(self) -> None:
        cmd = self.query_one("#command-input", Input)
        if cmd.display:
            cmd.display = False
            self._command_verb = None
            self._agg_group = None
            self.query_one("#data-view-table", DataTable).focus()
        else:
            self._df = None
            self._stack = None
            self.app.pop_screen()

    def action_cursor_down(self) -> None:
        self.query_one("#data-view-table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#data-view-table", DataTable).action_cursor_up()

    def action_cursor_left(self) -> None:
        self.query_one("#data-view-table", DataTable).action_cursor_left()

    def action_cursor_right(self) -> None:
        self.query_one("#data-view-table", DataTable).action_cursor_right()

    def action_scroll_top(self) -> None:
        table = self.query_one("#data-view-table", DataTable)
        table.move_cursor(row=0)

    def action_scroll_bottom(self) -> None:
        table = self.query_one("#data-view-table", DataTable)
        if table.row_count > 0:
            table.move_cursor(row=table.row_count - 1)

    @on(DataTable.CellHighlighted, "#data-view-table")
    def _on_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        col = event.coordinate.column
        if self._df is not None and 0 <= col < len(self._df.columns):
            self._cursor_column_index = col
            self._update_status_bar()

    # --- Stack browser ---

    def action_toggle_stack_browser(self) -> None:
        self._stack_browser_visible = not self._stack_browser_visible
        panel = self.query_one("#stack-browser-panel")
        panel.display = self._stack_browser_visible
        if self._stack_browser_visible:
            self._render_stack_browser()

    def _render_stack_browser(self) -> None:
        if not self._stack_browser_visible or self._stack is None:
            return
        stack = self._stack
        label = self._row_data.aliases_display or self._row_data.hash[:12]
        base_marker = "\u2192 " if stack.cursor == 0 else "  "
        step_lines = tuple(
            "{}{:<3} {:<9} {}{}".format(
                "\u2192 " if (i + 1) == stack.cursor else "  ",
                i + 1,
                step.verb,
                step.user_input,
                "  (undone)" if (i + 1) > stack.cursor else "",
            )
            for i, step in enumerate(stack.steps)
        )
        code = stack.current_code
        code_lines = (
            ("--code equivalent:", code) if code else ("(no transforms applied)",)
        )
        lines = (f"{base_marker}0  base: {label}", *step_lines, "", *code_lines)
        self.query_one("#stack-browser-content", Static).update("\n".join(lines))

    # --- Persist to catalog ---

    def action_persist(self) -> None:
        if self._stack is None or self._stack.cursor == 0:
            return
        self._command_verb = "save"
        cmd = self.query_one("#command-input", Input)
        cmd.remove_class("error")
        cmd.value = ""
        cmd.placeholder = "alias name (leave empty to save without alias)"
        cmd.border_title = "save\u25b8"
        cmd.display = True
        cmd.focus()

    def _do_persist(self, alias=None) -> None:
        """Persist current stack expression to catalog via ExprComposer."""
        if self._stack is None or self._stack.cursor == 0:
            return
        self._persist_to_catalog(alias)

    @work(thread=True, exit_on_error=False)
    def _persist_to_catalog(self, alias) -> None:
        try:
            from xorq.catalog.composer import ExprComposer  # noqa: PLC0415

            code = self._stack.current_code
            composer = ExprComposer(source=self._entry, code=code, alias=alias)
            expr = composer.expr
            catalog = self.app._catalog
            if catalog is None:
                self.app.call_from_thread(
                    self._show_command_error, "No catalog available"
                )
                return
            entry = catalog.add(expr)
            if alias:
                catalog.add_alias(entry.name, alias)
            msg = f"Saved as '{alias or entry.name[:12]}'"
            self.app.call_from_thread(self._show_persist_success, msg)
        except Exception as e:
            self.app.call_from_thread(self._show_command_error, f"Save failed: {e}")

    def _show_persist_success(self, message) -> None:
        self.query_one("#data-view-status", Static).update(f" \u2713 {message}")


class RunDataScreen(Screen):
    """Full-screen parquet viewer for inspecting run output."""

    BINDINGS = (
        Binding("escape", "go_back", "Back"),
        Binding("q", "go_back", "Back", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("h", "scroll_left", "Left", show=False),
        Binding("l", "scroll_right", "Right", show=False),
    )

    def __init__(self, parquet_path: str, run_row: RunRowData | None = None):
        super().__init__()
        self._parquet_path = parquet_path
        self._run_row = run_row

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="run-data-status")
        yield DataTable(id="run-data-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#run-data-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.loading = True

        run_label = ""
        if self._run_row:
            run_label = f" · run {self._run_row.run_id[:8]}"
        self.query_one("#run-data-status", Static).update(
            f" Loading {self._parquet_path}{run_label}..."
        )
        self._load_data()

    @work(thread=True, exit_on_error=False)
    def _load_data(self) -> None:
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415

            pf = pq.ParquetFile(self._parquet_path)
            metadata = pf.metadata
            total_rows = metadata.num_rows
            num_cols = metadata.num_columns
            file_size = Path(self._parquet_path).stat().st_size

            table = pf.read_row_group(0) if metadata.num_row_groups > 0 else pf.read()
            df = table.to_pandas().head(500)
            columns = tuple(str(c) for c in df.columns)
            rows = tuple(
                tuple(
                    "—"
                    if isinstance(v, float) and math.isnan(v)
                    else str(round(v, 2))
                    if isinstance(v, float)
                    else str(v)
                    for v in row
                )
                for row in df.itertuples(index=False)
            )
            self.app.call_from_thread(
                self._render_data,
                columns,
                rows,
                total_rows,
                num_cols,
                file_size,
                len(df),
            )
        except Exception as e:
            self.app.call_from_thread(self._render_error, str(e))

    def _render_data(
        self, columns, rows, total_rows, num_cols, file_size, preview_rows
    ) -> None:
        with self.app.batch_update():
            size_str = (
                f"{file_size / 1024 / 1024:.1f} MB"
                if file_size > 1024 * 1024
                else f"{file_size / 1024:.1f} KB"
            )
            run_info = ""
            if self._run_row:
                run_info = f" · {self._run_row.status_display} {self._run_row.duration_display}"
            self.query_one("#run-data-status", Static).update(
                f" {total_rows} rows × {num_cols} cols · {size_str}"
                f" · showing {preview_rows}{run_info}"
            )
            table = self.query_one("#run-data-table", DataTable)
            table.clear(columns=True)
            table.loading = False
            for col in columns:
                table.add_column(col, key=col)
            for i, row in enumerate(rows):
                table.add_row(*row, key=str(i))

    def _render_error(self, message) -> None:
        self.query_one("#run-data-status", Static).update(f" Error: {message}")
        self.query_one("#run-data-table", DataTable).loading = False

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_cursor_down(self) -> None:
        self.query_one("#run-data-table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#run-data-table", DataTable).action_cursor_up()

    def action_scroll_left(self) -> None:
        self.query_one("#run-data-table", DataTable).action_scroll_left()

    def action_scroll_right(self) -> None:
        self.query_one("#run-data-table", DataTable).action_scroll_right()


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
    #catalog-panel:focus-within {
        border: double #C1F0FF;
    }
    #catalog-tree {
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
    #sql-panel {
        height: 2fr;
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
    DataTable:focus {
        border: none;
    }
    Tree:focus {
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
    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $surface;
    }
    RunDataScreen #run-data-status {
        height: 1;
        padding: 0 2;
        background: $surface;
    }
    RunDataScreen #run-data-table {
        height: 1fr;
    }
    DataViewScreen #data-view-status {
        height: 1;
        padding: 0 2;
        background: $surface;
    }
    DataViewScreen #data-view-split {
        height: 1fr;
    }
    DataViewScreen #data-view-table {
        height: 1fr;
    }
    DataViewScreen #stack-browser-panel {
        width: 40;
        border: solid #5abfb5;
        border-title-color: #5abfb5;
        padding: 0 1;
    }
    DataViewScreen #stack-browser-content {
        height: auto;
    }
    DataViewScreen #command-input {
        dock: bottom;
        height: 3;
        border: solid #2BBE75;
        border-title-color: #2BBE75;
        padding: 0 1;
    }
    DataViewScreen #command-input.error {
        border: solid #FF4757;
        border-title-color: #FF4757;
        color: #FF4757;
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
