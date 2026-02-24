from datetime import datetime
from functools import cache

import yaml
from attr import frozen
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Static,
    TabbedContent,
    TabPane,
)


REFRESH_INTERVAL = 2

COLUMNS = ("KIND", "ALIAS", "HASH", "BACKENDS", "OUTPUT", "CACHED", "TAGS")

REFLOG_COLUMNS = ("HASH", "DATE", "MESSAGE")

REVISION_COLUMNS = ("STATUS", "HASH", "COLUMNS", "CACHED", "DATE")


@frozen
class CatalogRowData:
    kind: str = "expr"
    alias: str = ""
    hash: str = ""
    backends: tuple[str, ...] = ()
    column_count: int | None = None
    cached: bool | None = None
    tags: tuple[str, ...] = ()

    @property
    @cache
    def backends_display(self) -> str:
        return ", ".join(sorted(set(self.backends))) if self.backends else ""

    @property
    @cache
    def output_display(self) -> str:
        match self.column_count:
            case None:
                return "?"
            case int(n):
                return f"{n} cols"
            case _:
                return "?"

    @property
    @cache
    def cached_display(self) -> str:
        match self.cached:
            case True:
                return "●"
            case False:
                return "○"
            case _:
                return "—"

    @property
    @cache
    def tags_display(self) -> str:
        return ", ".join(self.tags) if self.tags else ""

    @property
    def row(self) -> tuple[str, ...]:
        return (
            self.kind,
            self.alias,
            self.hash,
            self.backends_display,
            self.output_display,
            self.cached_display,
            self.tags_display,
        )


@frozen
class GitLogRowData:
    hash: str = ""
    date: str = ""
    message: str = ""

    @property
    def row(self) -> tuple[str, ...]:
        return (self.hash, self.date, self.message)


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
        match self.cached:
            case True:
                return "●"
            case False:
                return "○"
            case _:
                return "—"

    @property
    @cache
    def status_display(self) -> str:
        return "CURRENT →" if self.is_current else ""

    @property
    @cache
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


def _check_cached(expr):
    """Walk the expression tree for any materialized CachedNode."""
    try:
        if not expr.ls.has_cached:
            return False
        return any(cn.to_expr().ls.exists() for cn in expr.ls.cached_nodes)
    except Exception:
        return False


def _safe_entry_info(entry):
    try:
        expr = entry.expr
        column_count = len(expr.columns)
        cached = _check_cached(expr)
        tags = tuple(t.tag for t in expr.ls.tags if t.tag is not None)
    except Exception:
        column_count, cached, tags = None, None, ()
    return column_count, cached, tags


def _extract_backends(entry):
    import tarfile

    try:
        with tarfile.open(entry.catalog_path, "r:gz") as tf:
            f = tf.extractfile(f"{entry.name}/profiles.yaml")
            if f is None:
                return ()
            data = yaml.safe_load(f.read())
        if not isinstance(data, dict):
            return ()
        return tuple(
            pdata.get("con_name", "?")
            for pdata in data.values()
            if isinstance(pdata, dict)
        )
    except Exception:
        return ()


def snapshot_catalog(catalog):
    alias_lookup = {ca.catalog_entry.name: ca.alias for ca in catalog.catalog_aliases}
    return tuple(
        CatalogRowData(
            kind="expr",
            alias=alias_lookup.get(entry.name, ""),
            hash=entry.name,
            backends=_extract_backends(entry),
            column_count=column_count,
            cached=cached,
            tags=tags,
        )
        for entry in catalog.catalog_entries
        for column_count, cached, tags in (_safe_entry_info(entry),)
    )


def snapshot_git_reflog(catalog, max_count=50):
    try:
        entries = list(catalog.repo.head.log())
        return tuple(
            GitLogRowData(
                hash=entry.newhexsha[:12],
                date=datetime.fromtimestamp(entry.time[0]).strftime("%Y-%m-%d %H:%M"),
                message=entry.message.strip()[:80],
            )
            for entry in reversed(entries[-max_count:])
        )
    except Exception:
        return ()


def _build_lineage_chain(expr):
    from xorq.common.utils.graph_utils import gen_children_of, to_node
    from xorq.common.utils.lineage_utils import format_node

    try:
        node = to_node(expr)
        chain = []
        while True:
            chain.append(format_node(node))
            children = tuple(gen_children_of(node))
            if not children:
                break
            node = children[0]
        return tuple(reversed(chain))
    except Exception:
        return ("(lineage unavailable)",)


def _build_explore_data(entry, alias):
    expr = None
    try:
        expr = entry.expr
    except Exception:
        pass

    schema_items = ()
    if expr is not None:
        try:
            schema_items = tuple(
                (name, str(dtype)) for name, dtype in expr.schema().items()
            )
        except Exception:
            pass

    lineage_text = "(unavailable)"
    if expr is not None:
        try:
            lineage_chain = _build_lineage_chain(expr)
            lineage_text = " → ".join(lineage_chain) if lineage_chain else "(empty)"
        except Exception:
            pass

    is_cached = None
    cache_path = None
    if expr is not None:
        try:
            is_cached = _check_cached(expr)
        except Exception:
            pass
        if is_cached is True:
            try:
                paths = expr.ls.get_cache_paths()
                cache_path = paths[0] if paths else None
            except Exception:
                pass

    metadata_items = ()
    try:
        if entry.metadata_path.exists():
            meta = yaml.safe_load(entry.metadata_path.read_text())
            if isinstance(meta, dict):
                metadata_items = tuple((str(k), str(v)) for k, v in meta.items())
    except Exception:
        pass

    return ExploreData(
        hash=entry.name,
        alias=alias,
        schema_items=schema_items,
        lineage_text=lineage_text,
        is_cached=is_cached,
        cache_path=str(cache_path) if cache_path else None,
        metadata=metadata_items,
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="catalog-panel"):
            yield DataTable(id="catalog-table")
        with Vertical(id="log-panel"):
            yield DataTable(id="log-table")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#catalog-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        for col in COLUMNS:
            table.add_column(col, key=col)

        log_table = self.query_one("#log-table", DataTable)
        log_table.cursor_type = "row"
        log_table.zebra_stripes = True
        for col in REFLOG_COLUMNS:
            log_table.add_column(col, key=col)

        catalog_name = self.app._catalog.repo_path.name
        self.query_one("#catalog-panel").border_title = f"Expressions — {catalog_name}"
        self.query_one("#log-panel").border_title = "Git Reflog"

        self._do_refresh()
        self.set_interval(REFRESH_INTERVAL, self._do_refresh)

    def _do_refresh(self) -> None:
        catalog = self.app._catalog

        table = self.query_one("#catalog-table", DataTable)
        cursor_row = table.cursor_row
        rows = snapshot_catalog(catalog)
        table.clear()
        for row_data in rows:
            table.add_row(*row_data.row, key=row_data.hash)
        if cursor_row is not None and len(rows) > 0:
            table.move_cursor(row=min(cursor_row, len(rows) - 1))

        log_table = self.query_one("#log-table", DataTable)
        reflog_rows = snapshot_git_reflog(catalog)
        log_table.clear()
        for i, log_row in enumerate(reflog_rows):
            log_table.add_row(*log_row.row, key=str(i))

        stamp = datetime.now().strftime("%H:%M:%S")
        repo_path = catalog.repo.working_dir
        self.query_one("#status-bar", Static).update(
            f" {len(rows)} entries | {repo_path} | refreshed {stamp}"
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
        alias_lookup = {
            ca.catalog_entry.name: ca.alias for ca in catalog.catalog_aliases
        }
        try:
            entry = catalog.get_catalog_entry(entry_hash)
        except (AssertionError, KeyError):
            self.notify("Entry not found", severity="error")
            return
        alias = alias_lookup.get(entry_hash, "")
        catalog_alias = None
        if alias:
            for ca in catalog.catalog_aliases:
                if ca.alias == alias:
                    catalog_alias = ca
                    break
        self.app.push_screen(ExploreScreen(entry, alias, catalog_alias=catalog_alias))


class ExploreScreen(Screen):
    BINDINGS = (
        ("q", "go_back", "Back"),
        ("escape", "go_back", "Back"),
        ("1", "tab_schema", "Schema"),
        ("2", "tab_data", "Data"),
        ("3", "tab_revisions", "Revisions"),
        ("4", "tab_info", "Info"),
        ("5", "tab_profiles", "Profiles"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
        ("enter", "select_row", "Select"),
    )

    def __init__(self, entry, alias, catalog_alias=None):
        super().__init__()
        self._entry = entry
        self._alias = alias
        self._catalog_alias = catalog_alias
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
        yield Footer()

    def on_mount(self) -> None:
        schema_table = self.query_one("#schema-table", DataTable)
        schema_table.cursor_type = "row"
        schema_table.zebra_stripes = True
        schema_table.add_column("NAME", key="name")
        schema_table.add_column("TYPE", key="type")

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
            env_vars = []
            for _k, v in kwargs.items():
                if isinstance(v, str) and (m := env_re.match(v)):
                    env_vars.append(m.group(1) or m.group(2))
            return tuple(env_vars)

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
        data = _build_explore_data(self._entry, self._alias)
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
            expr = self._entry.expr
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
        revision_rows = []
        valid_revisions = []
        for i, (rev_entry, commit) in enumerate(raw_revisions):
            try:
                exists = rev_entry.exists()
            except Exception:
                exists = False
            col_count, cached, _ = (
                _safe_entry_info(rev_entry) if exists else (None, None, ())
            )
            revision_rows.append(
                RevisionRowData(
                    hash=rev_entry.name,
                    column_count=col_count,
                    cached=cached,
                    commit_date=datetime.fromtimestamp(commit.committed_date).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    is_current=(i == 0),
                )
            )
            valid_revisions.append((rev_entry, commit, exists))
        self.app.call_from_thread(
            self._render_revisions,
            tuple(revision_rows),
            tuple(valid_revisions),
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
        border: round $primary;
    }
    #catalog-table {
        height: 1fr;
    }
    #log-panel {
        height: 1fr;
        border: round $primary;
    }
    #log-table {
        height: 1fr;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    #breadcrumb {
        height: 1;
        background: $panel;
        color: $text-muted;
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
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    #data-table {
        height: 1fr;
    }
    #revisions-status {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    #revisions-table {
        height: 1fr;
    }
    #info-scroll {
        height: 1fr;
    }
    .info-section {
        border: round $primary;
        background: $surface;
        height: auto;
        margin: 0 1 1 1;
        padding: 1 2;
    }
    .info-section Static {
        height: auto;
    }
    #profiles-table {
        height: 1fr;
    }
    """

    def __init__(self, catalog):
        super().__init__()
        self._catalog = catalog

    def on_mount(self) -> None:
        self.push_screen(CatalogScreen())
