import math
import re
import subprocess
import threading
from collections import Counter
from datetime import datetime
from functools import cache, cached_property, lru_cache
from pathlib import Path
from typing import Literal

from attr import evolve, field, frozen
from attr.validators import instance_of, optional
from pygments import lex as pygments_lex
from pygments.lexers import get_lexer_by_name as pygments_get_lexer
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
from xorq.common.utils.logging_utils import get_logger
from xorq.config import options
from xorq.ibis_yaml.enums import ExprKind


logger = get_logger(__name__)

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
    variables={
        "flash-new": "#FF69B4",
        "subdued": "#5abfb5",
        "panel-dim": "#3d6670",
        "panel-dim-fg": "#7aa8b2",
    },
)

KIND_ORDER: tuple[ExprKind, ...] = (
    ExprKind.Source,
    ExprKind.Expr,
    ExprKind.UnboundExpr,
    ExprKind.Composed,
    ExprKind.ExprBuilder,
)


_SQL_LEXER = pygments_get_lexer("sql", stripnl=False)


@lru_cache(maxsize=64)
def _pygments_tokens(sql: str) -> tuple[tuple[str, str], ...]:
    tokens = []
    for ttype, value in pygments_lex(sql, _SQL_LEXER):
        info = XorqSQLStyle.style_for_token(ttype)
        parts = []
        if info.get("bold"):
            parts.append("bold")
        if info.get("italic"):
            parts.append("italic")
        if info.get("color"):
            parts.append(f"#{info['color']}")
        tokens.append((value, " ".join(parts)))
    return tuple(tokens)


def _pygments_to_text(sql: str) -> Text:
    text = Text(no_wrap=False, overflow="fold")
    for value, style in _pygments_tokens(sql):
        text.append(value, style=style)
    return text


SQL_HIGHLIGHT_MAX_LINES = 500


def _render_sql_text(raw: str) -> Text:
    # Line-length is intentionally unchecked: extremely wide lines render slowly
    # in Textual, but that's an acceptable tradeoff vs. adding another heuristic.
    max_lines = options.tui.sql_highlight_max_lines
    if max_lines == 0 or raw.count("\n") > max_lines:
        note = (
            "-- syntax highlighting disabled\n"
            if max_lines == 0
            else f"-- syntax highlighting disabled (query exceeds {max_lines} lines)\n"
        )
        rich_text = Text(no_wrap=False, overflow="fold")
        rich_text.append(note, style="italic #4AA8EC")
        rich_text.append(raw)
        return rich_text
    return _pygments_to_text(raw)


@frozen
class KindStyle:
    icon: str = field(validator=instance_of(str))
    color: str = field(validator=instance_of(str))


KIND_STYLES: dict[ExprKind, KindStyle] = {
    ExprKind.Source: KindStyle(icon="⊞", color=XORQ_DARK.primary),
    ExprKind.Expr: KindStyle(icon="⊕", color=XORQ_DARK.success),
    ExprKind.UnboundExpr: KindStyle(icon="⊘", color=XORQ_DARK.warning),
    ExprKind.Composed: KindStyle(icon="⊛", color=XORQ_DARK.secondary),
    ExprKind.ExprBuilder: KindStyle(icon="⊡", color=XORQ_DARK.secondary),
}

CACHE_STYLE: dict[bool | None, tuple[str, str]] = {
    True: ("●", XORQ_DARK.success),
    False: ("○", XORQ_DARK.warning),
    None: ("—", "dim"),
}

FLASH_NEW = XORQ_DARK.variables["flash-new"]
SUBDUED = XORQ_DARK.variables["subdued"]

SCHEMA_PREVIEW_COLUMNS = ("NAME", "TYPE")

REVISION_COLUMNS = ("STATUS", "HASH", "COLUMNS", "CACHED", "DATE")

GIT_LOG_COLUMNS = ("HASH", "DATE", "MESSAGE")


def _styled_branch_label(kind: str, count: int) -> Text:
    style = KIND_STYLES[kind]
    label = Text()
    label.append(f"{style.icon} ", style=f"bold {style.color}")
    label.append(f"{kind} ", style=f"bold {style.color}")
    label.append(f"({count})", style=f"dim {style.color}")
    return label


def _format_cached(value: bool | None) -> str:
    return CACHE_STYLE[value][0]


def get_cache_key_path(cache_key: CacheKey | None) -> str | None:
    return (
        str(resolve_parquet_cache_path(cache_key.relative_path, cache_key.key))
        if cache_key is not None
        else None
    )


@frozen
class CatalogRowData:
    entry: CatalogEntry = field(repr=False)
    aliases: tuple[str, ...] = field(factory=tuple, validator=instance_of(tuple))

    @property
    def cached(self) -> bool | None:
        if path := get_cache_key_path(self.entry.projected_cache_key):
            return Path(path).exists()
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
        path = get_cache_key_path(self.entry.projected_cache_key)
        match path:
            case None:
                return "— unknown"
            case _ if Path(path).exists():
                return f"● cached  {path}"
            case _:
                return "○ uncached"

    @cached_property
    def info_text(self) -> str:
        parts = [
            f"Lineage: {self.lineage_text}",
            f"Cache: {self.cache_info_text}",
            f"Hash: {self.hash}",
        ]
        return "\n".join(parts)

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


VIEW_LIMIT = 50_000


@cache
def _ibis_table_method_names() -> tuple[str, ...]:
    """Public method names on the ibis Table class, for tab-completion."""
    from xorq.vendor.ibis.expr.types.relations import Table  # noqa: PLC0415

    return tuple(name for name in dir(Table) if not name.startswith("_"))


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

    @property
    def current_code(self) -> str:
        """Single evaluable expression chaining all active steps.

        Each step is wrapped in a ``(lambda source: step_code)(prior)`` call,
        so every ``source`` identifier in the step binds to the prior step's
        result via the same namespace mechanism ``_eval_code`` uses — no
        string substitution of ``source`` inside user code.
        """
        if self.cursor == 0:
            return ""
        result = "source"
        for step in self.steps[: self.cursor]:
            result = f"(lambda source: {step.code})({result})"
        return result


def _entry_info(entry: CatalogEntry) -> tuple[int | None, bool | None]:
    path = get_cache_key_path(entry.projected_cache_key)
    cached = Path(path).exists() if path is not None else None
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
        Binding("q", "quit_app", "Quit"),
        Binding("ctrl+c", "quit_app", "Quit", show=False),
        Binding("h", "tree_collapse", "Collapse", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("l", "tree_expand", "Expand", show=False),
        Binding("tab", "focus_next_panel", "Next", show=False),
        Binding("shift+tab", "focus_prev_panel", "Prev", show=False),
        Binding("e", "open_data_view", "Explore"),
        Binding("v", "toggle_revisions", "Revisions"),
        Binding("g", "toggle_git_log", "Git Log"),
        Binding("1", "view_sql", "SQL", priority=True),
        Binding("2", "view_data", "Data", priority=True),
    )

    FOCUS_CYCLE = (
        "#catalog-tree",
        "#sql-panel",
        "#data-preview-panel",
        "#schema-preview-table",
    )

    def __init__(self, refresh_interval=DEFAULT_REFRESH_INTERVAL):
        super().__init__()
        self._refresh_interval = refresh_interval
        self._row_cache: dict[str, CatalogRowData] = {}
        # Written under _refresh_lock on worker thread; read on main thread
        # only via call_from_thread callbacks or during locked render calls.
        self._new_keys: set[str] = set()
        self._git_log_visible = options.tui.git_log_open
        self._git_log_loaded = False
        self._refresh_lock = threading.Lock()
        self._active_view: Literal["sql", "data"] = "sql"
        self._data_preview_hash: str | None = None
        self._current_sql_hash: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-split"):
            with Vertical(id="left-column"):
                with Vertical(id="catalog-panel"):
                    yield Tree("Catalog", id="catalog-tree")
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
        self.query_one("#revisions-panel").border_title = "Revisions"
        self.query_one("#revisions-panel").display = options.tui.revisions_open

        git_log_panel = self.query_one("#git-log-panel")
        git_log_panel.border_title = "Git Log"
        git_log_panel.display = options.tui.git_log_open

        self.query_one("#left-column").styles.width = f"{options.tui.left_ratio}fr"
        self.query_one("#right-column").styles.width = f"{options.tui.right_ratio}fr"

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

        # Branch nodes (kind groupings) have children; only leaf nodes are entries
        entry_hash = event.node.data
        row_data = (
            self._row_cache.get(entry_hash)
            if not event.node.children and entry_hash is not None
            else None
        )
        if row_data is None:
            self._current_sql_hash = None
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
                self._current_sql_hash = None
                sql_preview.update("(SQL unavailable)")
                sql_panel.border_subtitle = ""
            case ((_, engine, sql),):
                sql_panel.border_subtitle = engine
                self._current_sql_hash = row_data.row_key
                sql_preview.update(Text("Rendering SQL Query…", style="dim"))
                self._load_sql_preview(row_data.row_key, sql)
            case sqls:
                engines = sorted({engine for _, engine, _ in sqls})
                sql_panel.border_subtitle = (
                    f"{len(sqls)} queries · {', '.join(engines)}"
                )
                self._current_sql_hash = row_data.row_key
                sql_preview.update(Text("Rendering SQL Query…", style="dim"))
                self._load_sql_preview(row_data.row_key, sqls)

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

    def _styled_leaf_label(self, row_data: CatalogRowData) -> Text:
        is_new = row_data.row_key in self._new_keys
        cache_icon, cache_color = CACHE_STYLE[row_data.cached]
        ncols = len(row_data.schema_out)
        short_hash = row_data.hash[:12]

        if is_new:
            icon_style = f"bold {FLASH_NEW}"
            name_style = f"bold {FLASH_NEW}"
            hash_style = f"dim {FLASH_NEW}"
            badge_style = f"dim {FLASH_NEW}"
        else:
            icon_style = cache_color
            name_style = f"bold {XORQ_DARK.primary}"
            hash_style = f"dim {SUBDUED}"
            badge_style = "dim"

        label = Text()
        label.append(f"{cache_icon} ", style=icon_style)
        if row_data.aliases_display:
            label.append(row_data.aliases_display, style=name_style)
            label.append(f" {short_hash}", style=hash_style)
        else:
            label.append(short_hash, style=name_style if is_new else XORQ_DARK.primary)
        label.append(f" ·{ncols}", style=badge_style)
        return label

    def _settle_new_labels(self, keys: set[str]) -> None:
        tree = self.query_one("#catalog-tree", Tree)
        for branch in tree.root.children:
            for leaf in branch.children:
                if leaf.data in keys and leaf.data in self._row_cache:
                    leaf.set_label(self._styled_leaf_label(self._row_cache[leaf.data]))

    def _render_status(self, stamp, repo_path) -> None:
        rows = self._row_cache.values()
        count = len(self._row_cache)
        kind_counts = Counter(r.kind for r in rows)
        cached_count = sum(1 for r in rows if r.cached)
        kinds_str = ", ".join(
            f"{kind_counts[k]} {k}" for k in KIND_ORDER if k in kind_counts
        )
        header = f" {count} entries" + (f" ({kinds_str})" if kinds_str else "")
        parts = [header]
        if cached_count:
            parts.append(f"{cached_count} cached")
        parts.append(str(repo_path))
        parts.append(stamp)
        self.query_one("#status-bar", Static).update(" · ".join(parts))

    # --- SQL preview worker ---

    @work(thread=True, exit_on_error=False, exclusive=True, group="sql_render")
    def _load_sql_preview(
        self,
        entry_hash: str,
        raw: str | tuple[tuple[str, str, str], ...],
    ) -> None:
        try:
            if not isinstance(raw, str):
                raw = _render_sql_dag(raw)
            rich_text = _render_sql_text(raw)
        except Exception:
            logger.exception("sql_preview_render_failed", entry_hash=entry_hash)
            rich_text = Text("(render error)", style="dim")

        def _apply():
            if not self.is_attached or self._current_sql_hash != entry_hash:
                return
            self.query_one("#sql-preview", Static).update(rich_text)

        self.app.call_from_thread(_apply)

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
        tree = self.query_one("#catalog-tree", Tree)
        node = tree.cursor_node
        if node is None or node.children:
            return
        row_data = self._row_cache.get(node.data)
        if row_data is None or row_data.kind == "unbound_expr":
            return
        self.app.push_screen(DataViewScreen(entry=row_data.entry, row_data=row_data))

    def action_quit_app(self) -> None:
        self.app.exit()


class DataViewScreen(Screen):
    """Full-screen data viewer with interactive expression composition.

    Column-level verbs (sort, drop) and a freeform `:` prompt each push an
    Ibis call onto an undo/redo ExprStack; the full chain is re-evaluated via
    ``xorq catalog run -c`` on every change.
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
        Binding("s", "toggle_stack_browser", "Stack"),
        Binding("w", "persist", "Save"),
        Binding(":", "open_freeform", "Expr"),
    )

    def __init__(self, entry, row_data):
        super().__init__()
        self._entry = entry
        self._row_data = row_data
        self._stack = None
        self._df = None
        self._cursor_column_index = 0
        self._stack_browser_visible = False
        self._command_mode = None
        self._active_proc = None
        self._proc_lock = threading.Lock()

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

    def on_unmount(self) -> None:
        self._kill_active_proc()

    def _catalog_base_cmd(self, subcommand: str) -> list[str]:
        """Shared ``xorq catalog --path <repo> <subcommand> <entry>`` prefix."""
        catalog = self.app._catalog
        entry_name = (
            self._row_data.aliases[0] if self._row_data.aliases else self._entry.name
        )
        return [
            "xorq",
            "catalog",
            "--path",
            str(catalog.repo_path),
            subcommand,
            entry_name,
        ]

    def _catalog_run_cmd(self, code=None) -> list[str]:
        """Build the xorq catalog run subprocess command."""
        cmd = self._catalog_base_cmd("run") + [
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

    def _spawn_run(self, cmd):
        with self._proc_lock:
            prior = self._active_proc
            if prior is not None and prior.poll() is None:
                prior.kill()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._active_proc = proc
        try:
            stdout, stderr = proc.communicate()
        finally:
            with self._proc_lock:
                if self._active_proc is proc:
                    self._active_proc = None
        return proc.returncode, stdout, stderr

    def _run_catalog_subprocess(self, code=None):
        """Run xorq catalog run and return a pandas DataFrame.

        Try `--use-this-venv` first; fall back to the uv-isolated path
        on any failure.
        """
        import pyarrow as pa  # noqa: PLC0415

        cmd = self._catalog_run_cmd(code)
        fast_stderr = ""
        try:
            returncode, stdout, stderr = self._spawn_run([*cmd, "--use-this-venv"])
            fast_stderr = stderr.decode(errors="replace").strip()
            if returncode == 0:
                return pa.ipc.open_stream(stdout).read_pandas()
            logger.debug(
                "catalog_run_fast_path_nonzero",
                returncode=returncode,
                stderr=fast_stderr[-500:],
            )
        except (OSError, pa.lib.ArrowException):
            logger.exception(
                "catalog_run_fast_path_failed",
                stderr=fast_stderr,
            )
        returncode, stdout, stderr = self._spawn_run(cmd)
        if returncode != 0:
            raise RuntimeError(stderr.decode(errors="replace").strip())
        return pa.ipc.open_stream(stdout).read_pandas()

    def _kill_active_proc(self) -> None:
        with self._proc_lock:
            proc = self._active_proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    @work(thread=True, exit_on_error=False)
    def _load_data(self) -> None:
        entry_hash = self._row_data.hash[:12]
        try:
            self._stack = ExprStack(base_expr=self._entry)
            df = self._run_catalog_subprocess()
            self.app.call_from_thread(self._on_data_loaded, df)
        except Exception as e:
            logger.exception("data_view_load_failed", entry_hash=entry_hash)
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

    def _update_command_suggester(self) -> None:
        """Update tab-completion suggestions from current expression columns."""
        cols = tuple(self._df.columns) if self._df is not None else ()
        self.query_one("#command-input", Input).suggester = SuggestFromList(
            cols + _ibis_table_method_names(), case_sensitive=False
        )

    # --- Stack operations ---

    def _push_step(self, verb: str, user_input: str, code: str) -> None:
        """Push a step and re-execute in background."""
        step = ExprStep(verb=verb, user_input=user_input, code=code)
        self._stack = self._stack.push(step)
        self._execute_current()

    @work(thread=True, exit_on_error=False, exclusive=True, group="execute_current")
    def _execute_current(self) -> None:
        """Evaluate current stack expression via subprocess."""
        stack = self._stack
        code = stack.current_code or None
        try:
            df = self._run_catalog_subprocess(code)
        except Exception as e:
            logger.exception(
                "stack_execute_failed",
                cursor=stack.cursor,
                steps=len(stack.steps),
                code=code[:500] + "..." if code and len(code) > 500 else code,
            )
            self.app.call_from_thread(self._on_stack_execute_failed, stack, str(e))
            return
        self.app.call_from_thread(self._on_stack_executed, stack, df)

    def _on_stack_executed(self, stack, df) -> None:
        if self._stack is not stack:
            return
        self._df = df
        self._cursor_column_index = 0
        self._render_table()
        self._update_command_suggester()
        self._render_stack_browser()

    def _on_stack_execute_failed(self, stack, message) -> None:
        if self._stack is not stack:
            return
        self._stack = stack.undo()
        self._show_command_error(message)
        self._render_stack_browser()

    def _show_command_error(self, message) -> None:
        self.app.notify(message, title="Error", severity="error", timeout=6)

    # --- Command input ---

    def _close_command_input(self) -> None:
        self.query_one("#command-input", Input).display = False
        self._command_mode = None
        self.query_one("#data-view-table", DataTable).focus()

    @on(Input.Submitted, "#command-input")
    def _on_command_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        mode = self._command_mode
        self._close_command_input()

        if mode == "save":
            self._do_persist(user_input or None)
        elif user_input:
            self._push_step("freeform", user_input, user_input)

    # --- Freeform action ---

    def action_open_freeform(self) -> None:
        if self._stack is None:
            return
        self._command_mode = "freeform"
        cmd = self.query_one("#command-input", Input)
        cmd.value = ""
        cmd.placeholder = ":\u25b8 type expression, Tab to complete, Enter to apply"
        cmd.border_title = ":\u25b8"
        cmd.display = True
        cmd.focus()

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
            self._close_command_input()
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
            ("\u2014 code equivalent:", code) if code else ("(no transforms applied)",)
        )
        lines = (f"{base_marker}0  base: {label}", *step_lines, "", *code_lines)
        self.query_one("#stack-browser-content", Static).update("\n".join(lines))

    # --- Persist to catalog ---

    def action_persist(self) -> None:
        if self._stack is None or self._stack.cursor == 0:
            return
        self._command_mode = "save"
        cmd = self.query_one("#command-input", Input)
        cmd.value = ""
        cmd.placeholder = "alias name (leave empty to save without alias)"
        cmd.border_title = "save\u25b8"
        cmd.display = True
        cmd.focus()

    def _do_persist(self, alias=None) -> None:
        if self._stack is None or self._stack.cursor == 0:
            return
        self._persist_to_catalog(alias)

    def _catalog_compose_cmd(self, code: str, alias: str | None) -> list[str]:
        cmd = self._catalog_base_cmd("compose") + ["-c", code]
        if alias:
            cmd.extend(["-a", alias])
        return cmd

    @work(thread=True, exit_on_error=False)
    def _persist_to_catalog(self, alias) -> None:
        code = self._stack.current_code
        try:
            cmd = self._catalog_compose_cmd(code, alias)
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode(errors="replace").strip())
            msg = f"Saved as '{alias}'" if alias else "Saved"
            self.app.call_from_thread(self._show_persist_success, msg)
        except Exception as e:
            logger.exception(
                "catalog_compose_failed",
                alias=alias,
                code=code[:500] + "..." if code and len(code) > 500 else code,
            )
            self.app.call_from_thread(self._show_command_error, f"Save failed: {e}")

    def _show_persist_success(self, message) -> None:
        self.query_one("#data-view-status", Static).update(f" \u2713 {message}")


class CatalogTUI(App):
    TITLE = "xorq catalog"
    CSS = """
    #main-split { height: 1fr; }
    #left-column { width: 2fr; }
    #right-column { width: 3fr; }

    #catalog-panel,
    #revisions-panel,
    #git-log-panel,
    #sql-panel,
    #info-panel,
    #schema-panel,
    #data-preview-panel,
    DataViewScreen #stack-browser-panel {
        border: solid $panel-dim;
        border-title-color: $panel-dim-fg;
        border-subtitle-color: $panel-dim-fg;
    }
    #catalog-panel:focus-within,
    #revisions-panel:focus-within,
    #git-log-panel:focus-within,
    #sql-panel:focus-within,
    #info-panel:focus-within,
    #schema-panel:focus-within,
    #data-preview-panel:focus-within,
    DataViewScreen #stack-browser-panel:focus-within {
        border: double $accent;
        border-title-color: $accent;
        border-subtitle-color: $accent;
    }

    #catalog-panel { height: 2fr; background: $surface; }
    #catalog-tree { height: 1fr; }
    #catalog-tree > .tree--guides { color: $panel-dim; }
    #catalog-tree > .tree--guides-hover { color: $subdued; }
    #catalog-tree > .tree--guides-selected { color: $accent; }
    #revisions-panel { height: 1fr; }
    #revisions-preview-table { height: 1fr; }
    #git-log-panel { height: 1fr; }
    #git-log-table { height: 1fr; }
    #sql-panel { height: 2fr; }
    #sql-preview { height: auto; padding: 1 2; }

    DataTable:focus { border: none; }
    Tree:focus { border: none; }

    #info-panel { height: auto; max-height: 6; padding: 0 1; }
    #info-content { height: auto; }

    #schema-panel { height: 1fr; }
    #schema-split { height: 1fr; }
    #schema-in-half { width: 1fr; }
    #schema-out-half { width: 1fr; }
    #schema-in-table { height: 1fr; }
    #schema-preview-table { height: 1fr; }

    #data-preview-panel { height: 2fr; }
    #data-preview-status { height: 1; padding: 0 2; }
    #data-preview-table { height: 1fr; }

    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $surface;
    }

    DataViewScreen #data-view-status {
        height: 1;
        padding: 0 2;
        background: $surface;
    }
    DataViewScreen #data-view-split { height: 1fr; }
    DataViewScreen #data-view-table { height: 1fr; }
    DataViewScreen #stack-browser-panel { width: 40; padding: 0 1; }
    DataViewScreen #stack-browser-content { height: auto; }

    DataViewScreen #command-input {
        dock: bottom;
        height: 3;
        border: solid #2BBE75;
        border-title-color: #2BBE75;
        padding: 0 1;
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
