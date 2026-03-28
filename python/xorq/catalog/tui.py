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
    Static,
)
from toolz.curried import excepts as cexcepts

from xorq.catalog.catalog import CatalogEntry
from xorq.common.utils.func_utils import return_constant


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


def maybe(default, exc=Exception):
    return cexcepts(exc, handler=return_constant(default))


def _format_cached(value: bool | None) -> str:
    match value:
        case True:
            return "●"
        case False:
            return "○"
        case _:
            return "—"


def _format_column_count(n: int | None) -> str:
    match n:
        case None:
            return "?"
        case int(n):
            return f"{n} cols"
        case _:
            return "?"


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
        expr = maybe_expr(self.entry)
        match expr:
            case None:
                return ()
            case _:
                return maybe_sqls(expr)

    @cached_property
    def lineage_text(self) -> str:
        expr = maybe_expr(self.entry)
        match expr:
            case None:
                return "(unavailable)"
            case _:
                return maybe_lineage(expr)

    @cached_property
    def cache_info_text(self) -> str:
        expr = maybe_expr(self.entry)
        is_cached, cache_path = maybe_cache_info(expr)
        match (is_cached, cache_path):
            case (True, str() as path):
                return f"● cached  {path}"
            case (True, _):
                return "● cached"
            case (False, _):
                return "○ uncached"
            case _:
                return "— unknown"

    @cached_property
    def metadata_text(self) -> str:
        items = maybe_metadata(self.entry)
        match items:
            case ():
                return ""
            case _:
                return "  ".join(f"{k}={v}" for k, v in items)

    @cached_property
    def info_text(self) -> str:
        parts = [
            f"Lineage: {self.lineage_text}",
            f"Cache: {self.cache_info_text}",
        ]
        if self.metadata_text:
            parts.append(f"Metadata: {self.metadata_text}")
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
        return _format_column_count(self.column_count)

    @property
    def row(self) -> tuple[str, ...]:
        return (
            self.status_display,
            self.hash,
            self.columns_display,
            self.cached_display,
            self.commit_date,
        )


def _check_cached(expr) -> bool:
    if not expr.ls.has_cached:
        return False
    return any(cn.to_expr().ls.exists() for cn in expr.ls.cached_nodes)


def _entry_info(entry) -> tuple[int | None, bool | None, str, None]:
    parquet_cache_paths = entry.parquet_cache_paths
    cached = (
        all(Path(p).exists() for p in parquet_cache_paths)
        if parquet_cache_paths
        else None
    )
    return len(entry.columns), cached, entry.root_tag, None


def _load_catalog_row(entry, aliases=()) -> CatalogRowData:
    return CatalogRowData(entry=entry, aliases=aliases)


def _build_lineage_chain(expr) -> tuple[str, ...]:
    from xorq.common.utils.graph_utils import gen_children_of, to_node  # noqa: PLC0415
    from xorq.common.utils.lineage_utils import format_node  # noqa: PLC0415

    def _walk(node):
        yield format_node(node)
        match tuple(gen_children_of(node)):
            case (first, *_):
                yield from _walk(first)
            case _:
                pass

    return tuple(reversed(tuple(_walk(to_node(expr)))))


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


@maybe(default=None)
def maybe_expr(entry):
    return entry.lazy_expr


@maybe(default="(unavailable)")
def maybe_lineage(expr) -> str:
    chain = _build_lineage_chain(expr)
    return " → ".join(chain) if chain else "(empty)"


@maybe(default=None)
def maybe_cache_path(expr) -> str | None:
    paths = expr.ls.get_cache_paths()
    return str(paths[0]) if paths else None


def maybe_cache_info(expr) -> tuple[bool | None, str | None]:
    match expr:
        case None:
            return None, None
        case _ if not _check_cached(expr):
            return False, None
        case _:
            return True, maybe_cache_path(expr)


@maybe(default=())
def maybe_sqls(expr) -> tuple[tuple[str, str, str], ...]:
    """Return ((name, engine, sql), ...) for all queries in the expression plan.

    Uses generate_sql_plans to decompose expressions into per-backend queries,
    showing every sub-query that gets executed.  Falls back to xorq's to_sql
    for unbound expressions that have no backend.
    """
    from xorq.expr.api import _remove_tag_nodes  # noqa: PLC0415
    from xorq.expr.api import to_sql as xorq_to_sql  # noqa: PLC0415
    from xorq.ibis_yaml.sql import generate_sql_plans  # noqa: PLC0415

    clean = _remove_tag_nodes(expr)
    try:
        sql_plans, deferred_reads = generate_sql_plans(clean)
    except Exception:
        # unbound expressions have no backend; use xorq default compiler
        sql = str(xorq_to_sql(clean)).strip()
        return (("main", "xorq", sql),) if sql else ()
    return tuple(
        (name, info.get("engine", "?"), info.get("sql", "").strip())
        for mapping in (sql_plans.get("queries", {}), deferred_reads.get("reads", {}))
        for name, info in mapping.items()
        if info.get("sql", "").strip()
    )


@maybe(default=())
def maybe_metadata(entry) -> tuple[tuple[str, str], ...]:
    if not entry.metadata_path.exists():
        return ()
    meta = yaml12.read_yaml(entry.metadata_path)
    match meta:
        case dict():
            return tuple((str(k), str(v)) for k, v in meta.items())
        case _:
            return ()


def _render_sql_dag(sqls: tuple[tuple[str, str, str], ...]) -> str:
    """Render multiple SQL queries as a topologically-sorted DAG."""
    import re as _re  # noqa: PLC0415

    name_to_sql = {name: (engine, sql) for name, engine, sql in sqls}
    # build dependency graph: name → set of names it depends on
    deps = {
        name: frozenset(
            ref
            for ref in _re.findall(r'FROM "([a-f0-9]{20,})"', sql)
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
    try:
        exists = rev_entry.exists()
    except Exception:
        exists = False
    col_count, cached, _, _ = (
        _entry_info(rev_entry) if exists else (None, None, "", None)
    )
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
        ("h", "scroll_left", "Left"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("l", "scroll_right", "Right"),
        ("tab", "focus_next_panel", "Next"),
        ("shift+tab", "focus_prev_panel", "Prev"),
        ("g", "toggle_git_log", "Git Log"),
        ("d", "toggle_data_preview", "Data"),
        ("p", "toggle_profiles", "Profiles"),
    )

    FOCUS_CYCLE = (
        "#catalog-table",
        "#sql-panel",
        "#schema-preview-table",
        "#revisions-preview-table",
    )

    def __init__(self, refresh_interval=DEFAULT_REFRESH_INTERVAL):
        super().__init__()
        self._refresh_interval = refresh_interval
        self._row_cache: dict[str, CatalogRowData] = {}
        self._git_log_visible = False
        self._git_log_loaded = False
        self._refresh_lock = threading.Lock()
        self._data_preview_visible = False
        self._data_preview_loaded = False
        self._data_preview_entry_hash: str | None = None
        self._profiles_visible = False
        self._profiles_loaded = False
        self._profiles_entry_hash: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-split"):
            with Vertical(id="left-column"):
                with Vertical(id="catalog-panel"):
                    yield DataTable(id="catalog-table")
                with Vertical(id="revisions-panel"):
                    yield DataTable(id="revisions-preview-table")
                with Vertical(id="git-log-panel"):
                    yield DataTable(id="git-log-table")
            with Vertical(id="right-column"):
                with VerticalScroll(id="sql-panel"):
                    yield Static("", id="sql-preview")
                with Vertical(id="info-panel"):
                    yield Static("", id="info-content")
                with Vertical(id="schema-in-panel"):
                    yield DataTable(id="schema-in-table")
                with Vertical(id="schema-out-panel"):
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
        self.query_one("#schema-in-panel").border_title = "Schema In"
        self.query_one("#schema-in-panel").display = False
        self.query_one("#schema-out-panel").border_title = "Schema"
        self.query_one("#sql-panel").border_title = "SQL"
        self.query_one("#info-panel").border_title = "Info"
        self.query_one("#revisions-panel").border_title = "Revisions"
        git_log_panel = self.query_one("#git-log-panel")
        git_log_panel.border_title = "Git Log"
        git_log_panel.display = False
        data_panel = self.query_one("#data-preview-panel")
        data_panel.border_title = "Data Preview"
        data_panel.display = False
        profiles_panel = self.query_one("#profiles-panel")
        profiles_panel.border_title = "Profiles"
        profiles_panel.display = False
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

        if event.row_key is None:
            sql_preview.update("")
            info_content.update("")
            self.query_one("#schema-in-panel").display = False
            self.query_one("#revisions-panel").border_title = "Revisions"
            self._reset_toggle_panels()
            return

        row_data = self._row_cache.get(str(event.row_key.value))
        if row_data is None:
            sql_preview.update("")
            info_content.update("")
            self.query_one("#schema-in-panel").display = False
            self.query_one("#revisions-panel").border_title = "Revisions"
            self._reset_toggle_panels()
            return

        # Schema
        match row_data.schema_in:
            case None:
                self.query_one("#schema-in-panel").display = False
                out_panel = self.query_one("#schema-out-panel")
                out_panel.border_title = "Schema"
                out_panel.border_subtitle = f"{len(row_data.schema_out)} cols"
            case schema_in:
                self.query_one("#schema-in-panel").display = True
                in_panel = self.query_one("#schema-in-panel")
                in_panel.border_subtitle = f"{len(schema_in)} cols"
                out_panel = self.query_one("#schema-out-panel")
                out_panel.border_title = "Schema Out"
                out_panel.border_subtitle = f"{len(row_data.schema_out)} cols"
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

        # Reset toggle panels on row change
        self._reset_toggle_panels()

    def _reset_toggle_panels(self) -> None:
        self._data_preview_loaded = False
        self._data_preview_entry_hash = None
        self._data_preview_visible = False
        self.query_one("#data-preview-panel").display = False
        dt = self.query_one("#data-preview-table", DataTable)
        dt.clear(columns=True)
        dt.loading = True

        self._profiles_loaded = False
        self._profiles_entry_hash = None
        self._profiles_visible = False
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

        self._data_preview_visible = not self._data_preview_visible
        self.query_one("#data-preview-panel").display = self._data_preview_visible

        if not self._data_preview_visible:
            return

        match row_data.cached:
            case True:
                if (
                    not self._data_preview_loaded
                    or self._data_preview_entry_hash != entry_hash
                ):
                    self._data_preview_loaded = True
                    self._data_preview_entry_hash = entry_hash
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
            expr = entry.expr
            try:
                df = expr.head(50).execute()
            except Exception:
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

        self._profiles_visible = not self._profiles_visible
        self.query_one("#profiles-panel").display = self._profiles_visible

        if self._profiles_visible and (
            not self._profiles_loaded or self._profiles_entry_hash != entry_hash
        ):
            self._profiles_loaded = True
            self._profiles_entry_hash = entry_hash
            self._load_profiles(row_data.entry)

    @work(thread=True, exit_on_error=False)
    def _load_profiles(self, entry) -> None:
        env_re = re.compile(r"^\$\{(.+)\}$|^\$(.+)$")

        def _extract_env_vars(kwargs):
            return tuple(
                m.group(1) or m.group(2)
                for v in kwargs.values()
                if isinstance(v, str) and (m := env_re.match(v))
            )

        try:
            with zipfile.ZipFile(entry.catalog_path, "r") as zf:
                member_path = f"{entry.name}/profiles.yaml"
                if member_path not in zf.namelist():
                    return
                data = yaml12.parse_yaml(zf.read(member_path))
            if not isinstance(data, dict):
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
        except Exception:
            pass

    def _render_profiles(self, rows) -> None:
        with self.app.batch_update():
            table = self.query_one("#profiles-table", DataTable)
            table.clear()
            for i, (name, backend, params, env_vars) in enumerate(rows):
                table.add_row(name, backend, params, env_vars, key=str(i))

    # --- Revisions Preview ---

    @work(thread=True, exit_on_error=False)
    def _load_revisions_preview(self, catalog_alias) -> None:
        try:
            raw_revisions = catalog_alias.list_revisions()
        except Exception:
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
        self.query_one(visible[next_idx]).focus()

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
    #schema-in-panel {
        height: 1fr;
        border: solid #7ED4C8;
        border-title-color: #7ED4C8;
        border-subtitle-color: #7ED4C8;
    }
    #schema-in-table {
        height: 1fr;
    }
    #schema-out-panel {
        height: 1fr;
        border: solid #4AA8EC;
        border-title-color: #4AA8EC;
        border-subtitle-color: #4AA8EC;
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
