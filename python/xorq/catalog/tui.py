from datetime import datetime
from functools import cache
from pathlib import Path

import yaml
from attr import field, frozen
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Static,
    TabbedContent,
    TabPane,
)
from toolz.curried import excepts as cexcepts

from xorq.common.utils.func_utils import return_constant


DEFAULT_REFRESH_INTERVAL = 10

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

COLUMNS = ("KIND", "ALIAS", "HASH", "BACKENDS", "OUTPUT", "CACHED", "TAGS")

SCHEMA_PREVIEW_COLUMNS = ("NAME", "TYPE")

REVISION_COLUMNS = ("STATUS", "HASH", "COLUMNS", "CACHED", "DATE")

ALIAS_COLUMNS = ("ALIAS",)


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
    kind: str = "expr"
    aliases: tuple[str, ...] = ()
    hash: str = ""
    backends: tuple[str, ...] = ()
    column_count: int | None = None
    cached: bool | None = None
    tags: tuple[str, ...] = ()
    cached_expr: object = field(default=None, eq=False, hash=False, repr=False)

    @property
    @cache
    def aliases_display(self) -> str:
        return ", ".join(self.aliases) if self.aliases else ""

    @property
    @cache
    def backends_display(self) -> str:
        return ", ".join(sorted(set(self.backends))) if self.backends else ""

    @property
    @cache
    def output_display(self) -> str:
        return _format_column_count(self.column_count)

    @property
    @cache
    def cached_display(self) -> str:
        return _format_cached(self.cached)

    @property
    @cache
    def tags_display(self) -> str:
        return ", ".join(self.tags) if self.tags else ""

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.aliases_display, self.hash)

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
            self.output_display,
            self.cached_display,
            self.tags_display,
        )


@frozen
class ExploreData:
    hash: str = ""
    alias: str = ""
    schema_items: tuple[tuple[str, str], ...] = ()
    lineage_text: str = ""
    is_cached: bool | None = None
    cache_path: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()
    has_alias: bool = False


@frozen
class RevisionRowData:
    hash: str = ""
    column_count: int | None = None
    cached: bool | None = None
    commit_date: str = ""
    is_current: bool = False

    @property
    @cache
    def cached_display(self) -> str:
        return _format_cached(self.cached)

    @property
    @cache
    def status_display(self) -> str:
        return "CURRENT →" if self.is_current else ""

    @property
    @cache
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


@maybe(default=False)
def maybe_check_cached(expr) -> bool:
    """Walk the expression tree for any materialized CachedNode."""
    if not expr.ls.has_cached:
        return False
    return any(cn.to_expr().ls.exists() for cn in expr.ls.cached_nodes)


@maybe(default=(None, None, (), None))
def maybe_entry_info(
    entry,
) -> tuple[int | None, bool | None, tuple[str, ...], object]:
    expr = entry.expr
    column_count = len(expr.columns)
    cached = maybe_check_cached(expr)
    tags = tuple(t.tag for t in expr.ls.tags if t.tag is not None)
    return column_count, cached, tags, expr


@maybe(default=())
def maybe_extract_backends(entry) -> tuple[str, ...]:
    import tarfile

    with tarfile.open(entry.catalog_path, "r:gz") as tf:
        f = tf.extractfile(f"{entry.name}/profiles.yaml")
        if f is None:
            return ()
        data = yaml.safe_load(f.read())
    if not isinstance(data, dict):
        return ()
    return tuple(
        pdata.get("con_name", "?") for pdata in data.values() if isinstance(pdata, dict)
    )


def _load_catalog_row(entry, aliases=()) -> CatalogRowData:
    column_count, cached, tags, expr = maybe_entry_info(entry)
    return CatalogRowData(
        kind="expr",
        aliases=aliases,
        hash=entry.name,
        backends=maybe_extract_backends(entry),
        column_count=column_count,
        cached=cached,
        tags=tags,
        cached_expr=expr,
    )


def _build_lineage_chain(expr) -> tuple[str, ...]:
    from xorq.common.utils.graph_utils import gen_children_of, to_node
    from xorq.common.utils.lineage_utils import format_node

    def _walk(node):
        yield format_node(node)
        match tuple(gen_children_of(node)):
            case (first, *_):
                yield from _walk(first)
            case _:
                pass

    return tuple(reversed(tuple(_walk(to_node(expr)))))


@maybe(default=None)
def maybe_expr(entry):
    return entry.expr


@maybe(default=())
def maybe_schema(expr) -> tuple[tuple[str, str], ...]:
    return tuple((name, str(dtype)) for name, dtype in expr.schema().items())


@maybe(default="(unavailable)")
def maybe_lineage(expr) -> str:
    chain = _build_lineage_chain(expr)
    return " → ".join(chain) if chain else "(empty)"


@maybe(default=None)
def maybe_cache_path(expr) -> str | None:
    paths = expr.ls.get_cache_paths()
    return str(paths[0]) if paths else None


def maybe_cache_info(expr) -> tuple[bool | None, str | None]:
    if expr is None:
        return None, None
    is_cached = maybe_check_cached(expr)
    if not is_cached:
        return is_cached, None
    return True, maybe_cache_path(expr)


@maybe(default=())
def maybe_metadata(entry) -> tuple[tuple[str, str], ...]:
    if not entry.metadata_path.exists():
        return ()
    meta = yaml.safe_load(entry.metadata_path.read_text())
    match meta:
        case dict():
            return tuple((str(k), str(v)) for k, v in meta.items())
        case _:
            return ()


def _build_explore_data(
    entry, alias, known_cached=None, cached_expr=None
) -> ExploreData:
    expr = cached_expr if cached_expr is not None else maybe_expr(entry)
    computed_cached, cache_path = maybe_cache_info(expr)
    return ExploreData(
        hash=entry.name,
        alias=alias,
        schema_items=maybe_schema(expr),
        lineage_text=maybe_lineage(expr),
        is_cached=known_cached if known_cached is not None else computed_cached,
        cache_path=cache_path,
        metadata=maybe_metadata(entry),
        has_alias=bool(alias),
    )


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


class CatalogScreen(Screen):
    BINDINGS = (
        ("q", "quit_app", "Quit"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("enter", "explore", "Explore"),
        ("e", "explore", "Explore"),
    )

    def __init__(self, refresh_interval=DEFAULT_REFRESH_INTERVAL):
        super().__init__()
        self._refresh_interval = refresh_interval
        self._row_cache: dict[str, CatalogRowData] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="catalog-panel"):
            yield DataTable(id="catalog-table")
        with Vertical(id="schema-panel"):
            yield DataTable(id="schema-preview-table")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#catalog-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        for col in COLUMNS:
            table.add_column(col, key=col)

        schema_table = self.query_one("#schema-preview-table", DataTable)
        schema_table.cursor_type = "row"
        schema_table.zebra_stripes = True
        for col in SCHEMA_PREVIEW_COLUMNS:
            schema_table.add_column(col, key=col)

        self.query_one("#catalog-panel").border_title = "Expressions"
        self.query_one("#schema-panel").border_title = "Schema"
        self.query_one("#status-bar", Static).update(" Loading catalog...")

        self.set_interval(self._refresh_interval, self._do_refresh)

    @on(DataTable.RowHighlighted, "#catalog-table")
    def _on_catalog_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        schema_table = self.query_one("#schema-preview-table", DataTable)
        schema_table.clear()
        if event.row_key is None:
            return
        row_data = self._row_cache.get(str(event.row_key.value))
        if row_data is None:
            return
        for name, dtype in maybe_schema(row_data.cached_expr):
            schema_table.add_row(name, dtype)

    @work(thread=True, exclusive=True)
    def _do_refresh(self) -> None:
        catalog = self.app._catalog
        if catalog is None:
            return
        repo_path = catalog.repo.working_dir
        current_hashes = frozenset(catalog.list())

        # build a multimap: hash -> sorted tuple of aliases
        alias_multimap: dict[str, list[str]] = {}
        for ca in catalog.catalog_aliases:
            alias_multimap.setdefault(ca.catalog_entry.name, []).append(ca.alias)
        alias_multimap_frozen: dict[str, tuple[str, ...]] = {
            k: tuple(sorted(v)) for k, v in alias_multimap.items()
        }

        # one row per hash; row_key is the hash itself
        expected_keys: frozenset[str] = frozenset(current_hashes)

        # render cached rows immediately
        cached_rows = tuple(
            sorted(
                (self._row_cache[k] for k in expected_keys if k in self._row_cache),
                key=lambda r: r.sort_key,
            )
        )
        new_keys = expected_keys - self._row_cache.keys()
        self.app.call_from_thread(self._render_refresh_start, repo_path, cached_rows)

        # load new entries incrementally (one row per hash)
        for entry_hash in new_keys:
            entry = catalog.get_catalog_entry(entry_hash)
            aliases = alias_multimap_frozen.get(entry_hash, ())
            row_data = _load_catalog_row(entry, aliases)
            self._row_cache[row_data.row_key] = row_data
            self.app.call_from_thread(self._render_catalog_row, row_data)

        # evict removed entries
        removed = self._row_cache.keys() - expected_keys
        for k in removed:
            del self._row_cache[k]

        stamp = datetime.now().strftime("%H:%M:%S")
        self.app.call_from_thread(self._render_refresh_done, stamp, repo_path)

    def _render_refresh_start(self, repo_path, cached_rows) -> None:
        catalog_name = Path(repo_path).name
        self.query_one("#catalog-panel").border_title = f"Expressions — {catalog_name}"

        table = self.query_one("#catalog-table", DataTable)
        self._saved_cursor = table.cursor_row
        table.clear()
        for row_data in cached_rows:
            table.add_row(*row_data.row, key=row_data.row_key)

    def _render_catalog_row(self, row_data) -> None:
        self.query_one("#catalog-table", DataTable).add_row(
            *row_data.row, key=row_data.row_key
        )

    def _render_refresh_done(self, stamp, repo_path) -> None:
        table = self.query_one("#catalog-table", DataTable)
        table.sort("ALIAS", "HASH")
        count = table.row_count
        if self._saved_cursor is not None and count > 0:
            table.move_cursor(row=min(self._saved_cursor, count - 1))
        self.query_one("#status-bar", Static).update(
            f" {count} entries | {repo_path} | refreshed {stamp}"
        )

    def action_cursor_down(self) -> None:
        self.query_one("#catalog-table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#catalog-table", DataTable).action_cursor_up()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_explore(self) -> None:
        table = self.query_one("#catalog-table", DataTable)
        if table.row_count == 0:
            return
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        entry_hash = str(row_key.value)
        catalog = self.app._catalog
        try:
            entry = catalog.get_catalog_entry(entry_hash)
        except (AssertionError, KeyError):
            self.notify("Entry not found", severity="error")
            return
        row_data = self._row_cache.get(entry_hash)
        aliases = row_data.aliases if row_data is not None else ()
        first_alias = aliases[0] if aliases else ""
        catalog_alias = (
            next(
                (ca for ca in catalog.catalog_aliases if ca.alias == first_alias),
                None,
            )
            if first_alias
            else None
        )
        self.app.push_screen(
            ExploreScreen(
                entry, first_alias, catalog_alias=catalog_alias, row_data=row_data
            )
        )


class ExploreScreen(Screen):
    BINDINGS = (
        ("q", "go_back", "Back"),
        ("escape", "go_back", "Back"),
        ("1", "tab_schema", "Schema"),
        ("2", "tab_data", "Data"),
        ("3", "tab_revisions", "Revisions"),
        ("4", "tab_info", "Info"),
        ("5", "tab_profiles", "Profiles"),
        ("6", "tab_aliases", "Aliases"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("enter", "select_row", "Select"),
    )

    def __init__(self, entry, alias, catalog_alias=None, row_data=None):
        super().__init__()
        self._entry = entry
        self._alias = alias
        self._catalog_alias = catalog_alias
        self._row_data = row_data
        self._explore_data = None
        self._data_loaded = False
        self._revisions_loaded = False
        self._revisions = ()

    def compose(self) -> ComposeResult:
        alias_part = f" ({self._alias})" if self._alias else ""
        yield Header(show_clock=True)
        yield Static(
            f" {self._entry.name[:12]}{alias_part}",
            id="breadcrumb",
        )
        with TabbedContent(id="explore-tabs"):
            with TabPane("Schema", id="pane-schema"):
                yield DataTable(id="schema-table")
            with TabPane("Data", id="pane-data", disabled=True):
                yield Static("Loading...", id="data-status")
                yield DataTable(id="data-table")
            with TabPane("Revisions", id="pane-revisions", disabled=True):
                yield Static("Loading...", id="revisions-status")
                yield DataTable(id="revisions-table")
            with TabPane("Info", id="pane-info"):
                with VerticalScroll(id="info-scroll"):
                    with Vertical(id="lineage-section", classes="info-section"):
                        yield Static("", id="lineage-content")
                    with Vertical(id="cache-section", classes="info-section"):
                        yield Static("", id="cache-content")
                    with Vertical(id="metadata-section", classes="info-section"):
                        yield Static("", id="metadata-content")
            with TabPane("Profiles", id="pane-profiles"):
                yield DataTable(id="profiles-table")
            with TabPane("Aliases", id="pane-aliases"):
                yield DataTable(id="aliases-table")
        yield Footer()

    def on_mount(self) -> None:
        schema_table = self.query_one("#schema-table", DataTable)
        schema_table.cursor_type = "row"
        schema_table.zebra_stripes = True
        schema_table.add_column("NAME", key="name")
        schema_table.add_column("TYPE", key="type")

        aliases_table = self.query_one("#aliases-table", DataTable)
        aliases_table.cursor_type = "row"
        aliases_table.zebra_stripes = True
        for col in ALIAS_COLUMNS:
            aliases_table.add_column(col, key=col)

        data_table = self.query_one("#data-table", DataTable)
        data_table.cursor_type = "row"
        data_table.zebra_stripes = True
        data_table.loading = True

        rev_table = self.query_one("#revisions-table", DataTable)
        rev_table.cursor_type = "row"
        rev_table.zebra_stripes = True
        for col in REVISION_COLUMNS:
            rev_table.add_column(col, key=col)

        profiles_table = self.query_one("#profiles-table", DataTable)
        profiles_table.cursor_type = "row"
        profiles_table.zebra_stripes = True
        profiles_table.add_column("NAME", key="name")
        profiles_table.add_column("BACKEND", key="backend")
        profiles_table.add_column("PARAMETERS", key="params")
        profiles_table.add_column("ENV VARS", key="env_vars")
        self._load_profiles()

        self.query_one("#lineage-section").border_title = "Lineage"
        self.query_one("#cache-section").border_title = "Cache"
        self.query_one("#metadata-section").border_title = "Metadata"
        self.query_one("#metadata-section").display = False

        self._load_explore_data()

    @work(thread=True)
    def _load_profiles(self) -> None:
        import re
        import tarfile

        env_re = re.compile(r"^\$\{(.+)\}$|^\$(.+)$")

        def _extract_env_vars(kwargs):
            return tuple(
                m.group(1) or m.group(2)
                for v in kwargs.values()
                if isinstance(v, str) and (m := env_re.match(v))
            )

        try:
            with tarfile.open(self._entry.catalog_path, "r:gz") as tf:
                f = tf.extractfile(f"{self._entry.name}/profiles.yaml")
                if f is None:
                    return
                data = yaml.safe_load(f.read())
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
        table = self.query_one("#profiles-table", DataTable)
        for i, (name, backend, params, env_vars) in enumerate(rows):
            table.add_row(name, backend, params, env_vars, key=str(i))

    @work(thread=True)
    def _load_explore_data(self) -> None:
        known_cached = self._row_data.cached if self._row_data is not None else None
        cached_expr = self._row_data.cached_expr if self._row_data is not None else None
        data = _build_explore_data(
            self._entry,
            self._alias,
            known_cached=known_cached,
            cached_expr=cached_expr,
        )
        self.app.call_from_thread(self._render_explore, data)

    def _render_explore(self, data) -> None:
        self._explore_data = data

        schema_table = self.query_one("#schema-table", DataTable)
        if data.schema_items:
            for name, dtype in data.schema_items:
                schema_table.add_row(name, dtype)
        else:
            schema_table.add_row("(unavailable)", "")

        self.query_one("#lineage-content", Static).update(data.lineage_text)

        match data.is_cached:
            case True:
                cache_text = "● cached"
                if data.cache_path:
                    cache_text += f"\n  Path: {data.cache_path}"
            case False:
                cache_text = "○ uncached"
            case _:
                cache_text = "— unknown"
        self.query_one("#cache-content", Static).update(cache_text)

        if data.metadata:
            self.query_one("#metadata-section").display = True
            metadata_text = "\n".join(f"{k}: {v}" for k, v in data.metadata)
            self.query_one("#metadata-content", Static).update(metadata_text)

        aliases_table = self.query_one("#aliases-table", DataTable)
        aliases_table.clear()
        if self._row_data is not None:
            for i, alias_name in enumerate(self._row_data.aliases):
                aliases_table.add_row(alias_name, key=str(i))

        if data.is_cached is True:
            self.query_one("#pane-data", TabPane).disabled = False
            self.query_one("#data-status", Static).update(
                " Data preview (select tab to load)"
            )
        else:
            self.query_one("#data-status", Static).update(
                " uncached — run to materialize"
            )

        if data.has_alias:
            self.query_one("#pane-revisions", TabPane).disabled = False
            self.query_one("#revisions-status", Static).update(
                " Revisions (select tab to load)"
            )
        else:
            self.query_one("#revisions-status", Static).update(
                " No alias — revisions unavailable"
            )

    @on(TabbedContent.TabActivated)
    def _on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        match event.pane.id:
            case "pane-data":
                if (
                    not self._data_loaded
                    and self._explore_data
                    and self._explore_data.is_cached is True
                ):
                    self._data_loaded = True
                    self._load_data_preview()
            case "pane-revisions":
                if (
                    not self._revisions_loaded
                    and self._explore_data
                    and self._explore_data.has_alias
                ):
                    self._revisions_loaded = True
                    self._load_revisions()
            case _:
                pass

    @work(thread=True)
    def _load_data_preview(self) -> None:
        try:
            expr = (
                self._row_data.cached_expr
                if self._row_data is not None and self._row_data.cached_expr is not None
                else self._entry.expr
            )
            df = expr.head(50).execute()
            columns = tuple(str(c) for c in df.columns)
            rows = tuple(
                tuple(str(v) for v in row) for row in df.itertuples(index=False)
            )
            total_rows = len(df)
            self.app.call_from_thread(self._render_data, columns, rows, total_rows)
        except Exception as e:
            self.app.call_from_thread(self._render_data_error, str(e))

    def _render_data(self, columns, rows, total_rows) -> None:
        self.query_one("#data-status", Static).update(
            f" Data Preview — {total_rows} rows shown (max 50)"
        )
        data_table = self.query_one("#data-table", DataTable)
        data_table.loading = False
        for col in columns:
            data_table.add_column(col, key=col)
        for i, row in enumerate(rows):
            data_table.add_row(*row, key=str(i))

    def _render_data_error(self, message) -> None:
        self.query_one("#data-status", Static).update(f" Error loading data: {message}")
        self.query_one("#data-table", DataTable).loading = False

    @work(thread=True)
    def _load_revisions(self) -> None:
        if self._catalog_alias is None:
            self.app.call_from_thread(
                self._render_revisions_error, "Alias object not available"
            )
            return
        try:
            raw_revisions = self._catalog_alias.list_revisions()
        except Exception as e:
            self.app.call_from_thread(
                self._render_revisions_error,
                f"Failed to load revisions: {e}",
            )
            return

        def _revision_pair(i, rev_entry, commit):
            try:
                exists = rev_entry.exists()
            except Exception:
                exists = False
            col_count, cached, _, _ = (
                maybe_entry_info(rev_entry) if exists else (None, None, (), None)
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

        pairs = tuple(
            _revision_pair(i, rev_entry, commit)
            for i, (rev_entry, commit) in enumerate(raw_revisions)
        )
        revision_rows = tuple(row for row, _ in pairs)
        valid_revisions = tuple(info for _, info in pairs)
        self.app.call_from_thread(
            self._render_revisions,
            revision_rows,
            valid_revisions,
        )

    def _render_revisions(self, revision_rows, valid_revisions) -> None:
        self._revisions = valid_revisions
        alias_name = (
            (self._alias or self._explore_data.alias) if self._explore_data else "?"
        )
        self.query_one("#revisions-status", Static).update(
            f" Revisions for '{alias_name}' — {len(revision_rows)} revisions"
        )
        rev_table = self.query_one("#revisions-table", DataTable)
        for i, row_data in enumerate(revision_rows):
            rev_table.add_row(*row_data.row, key=str(i))

    def _render_revisions_error(self, message) -> None:
        self.query_one("#revisions-status", Static).update(f" Error: {message}")

    def _active_table(self) -> DataTable | None:
        tabs = self.query_one("#explore-tabs", TabbedContent)
        match tabs.active:
            case "pane-schema":
                return self.query_one("#schema-table", DataTable)
            case "pane-aliases":
                return self.query_one("#aliases-table", DataTable)
            case "pane-data":
                return self.query_one("#data-table", DataTable)
            case "pane-revisions":
                return self.query_one("#revisions-table", DataTable)
            case "pane-profiles":
                return self.query_one("#profiles-table", DataTable)
            case _:
                return None

    def action_go_back(self) -> None:
        self.dismiss()

    def action_cursor_down(self) -> None:
        table = self._active_table()
        if table is not None:
            table.action_cursor_down()
        else:
            self.query_one("#info-scroll", VerticalScroll).scroll_down()

    def action_cursor_up(self) -> None:
        table = self._active_table()
        if table is not None:
            table.action_cursor_up()
        else:
            self.query_one("#info-scroll", VerticalScroll).scroll_up()

    def action_tab_schema(self) -> None:
        self.query_one("#explore-tabs", TabbedContent).active = "pane-schema"

    def action_tab_aliases(self) -> None:
        self.query_one("#explore-tabs", TabbedContent).active = "pane-aliases"

    def action_tab_data(self) -> None:
        pane = self.query_one("#pane-data", TabPane)
        if pane.disabled:
            self.notify("Data tab unavailable (not cached)", severity="warning")
            return
        self.query_one("#explore-tabs", TabbedContent).active = "pane-data"

    def action_tab_revisions(self) -> None:
        pane = self.query_one("#pane-revisions", TabPane)
        if pane.disabled:
            self.notify("Revisions tab unavailable (no alias)", severity="warning")
            return
        self.query_one("#explore-tabs", TabbedContent).active = "pane-revisions"

    def action_tab_info(self) -> None:
        self.query_one("#explore-tabs", TabbedContent).active = "pane-info"

    def action_tab_profiles(self) -> None:
        self.query_one("#explore-tabs", TabbedContent).active = "pane-profiles"

    def action_select_row(self) -> None:
        tabs = self.query_one("#explore-tabs", TabbedContent)
        if tabs.active != "pane-revisions":
            return
        rev_table = self.query_one("#revisions-table", DataTable)
        if rev_table.row_count == 0:
            return
        row_key, _ = rev_table.coordinate_to_cell_key(rev_table.cursor_coordinate)
        idx = int(row_key.value)
        if idx >= len(self._revisions):
            return
        rev_entry, _, exists = self._revisions[idx]
        if not exists:
            self.notify("Entry no longer exists on disk", severity="warning")
            return
        explore_screens = tuple(
            s for s in self.app.screen_stack if isinstance(s, ExploreScreen)
        )
        if len(explore_screens) >= 2:
            self.notify("Maximum drill-down depth reached", severity="warning")
            return
        self.app.push_screen(ExploreScreen(rev_entry, ""))


class CatalogTUI(App):
    TITLE = "xorq catalog"
    CSS = """
    #catalog-panel {
        height: 2fr;
        border: solid $primary;
    }
    #catalog-table {
        height: 1fr;
    }
    #schema-panel {
        height: 1fr;
        border: solid $primary;
    }
    #schema-preview-table {
        height: 1fr;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
    }
    #breadcrumb {
        height: 1;
        text-style: bold;
        padding: 0 2;
    }
    #explore-tabs {
        height: 1fr;
    }
    #schema-table {
        height: 1fr;
    }
    #data-status {
        height: 1;
        padding: 0 2;
    }
    #data-table {
        height: 1fr;
    }
    #revisions-status {
        height: 1;
        padding: 0 2;
    }
    #revisions-table {
        height: 1fr;
    }
    #info-scroll {
        height: 1fr;
    }
    .info-section {
        height: auto;
        margin: 0 1 1 1;
        padding: 1 2;
    }
    .info-section Static {
        height: auto;
    }
    #aliases-table {
        height: 1fr;
    }
    #profiles-table {
        height: 1fr;
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
