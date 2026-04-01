import re
import threading
import zipfile
from datetime import datetime
from pathlib import Path

import yaml12
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
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from xorq.catalog.tui.detail import get_detail_strategy
from xorq.catalog.tui.models import (
    CACHE_PANEL_COLUMNS,
    COLUMNS,
    DEFAULT_REFRESH_INTERVAL,
    GIT_LOG_COLUMNS,
    REVISION_COLUMNS,
    RUN_COLUMNS,
    SCHEMA_PREVIEW_COLUMNS,
    CacheRowData,
    CatalogRowData,
    ComposeConfig,
    RunConfig,
    RunRowData,
    _build_alias_multimap,
    _build_cache_rows,
    _build_git_log_rows,
    _build_run_rows,
    _format_run_detail,
    _format_size,
    _get_catalog_aliases,
    _get_catalog_list,
    _invalidate_catalog_caches,
    _load_catalog_row,
    _render_sql_dag,
    _revision_pair,
    _TogglePanelState,
)
from xorq.catalog.tui.panels.services import ServicesPanel
from xorq.catalog.tui.screens.modals import (
    ComposeScreen,
    ConfirmScreen,
    RunOptionsScreen,
)
from xorq.catalog.tui.screens.telemetry import TelemetryScreen


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
        ("s", "toggle_services", "Services"),
        ("x", "compose", "Compose"),
        ("t", "show_telemetry", "Telemetry"),
        ("shift+x", "clear_runs", "Clear Runs"),
        ("a", "toggle_alias_filter", "Aliases"),
        ("shift+d", "delete_entry", "Delete"),
        ("shift+a", "remove_alias", "Rm Alias"),
    )

    FOCUS_CYCLE = (
        "#catalog-table",
        "#runs-table",
        "#caches-table",
        "#services-table",
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
        self._current_highlighted_hash: str | None = None
        self._current_run_fingerprint: tuple[tuple[str, ...], ...] = ()
        self._entry_preview_cache: dict[str, str] = {}
        self._git_log_visible = False
        self._git_log_loaded = False
        self._refresh_lock = threading.Lock()
        self._data_preview = _TogglePanelState()
        self._profiles = _TogglePanelState()
        self._caches_visible = False
        self._caches_loaded = False
        self._cache_row_paths: dict[str, str] = {}
        self._cache_row_data: dict[str, CacheRowData] = {}
        self._alias_filter = True

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
                yield ServicesPanel(id="services-panel")
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

        services_panel = self.query_one("#services-panel", ServicesPanel)
        services_panel.display = False

        self.query_one("#status-bar", Static).update(" Loading catalog...")

        self.set_interval(self._refresh_interval, self._do_refresh)

    _PANEL_SWAP_IDS = frozenset(
        (
            "catalog-table",
            "runs-table",
            "caches-table",
        )
    )

    def on_descendant_focus(self, event) -> None:
        """React to any descendant focus change (Tab, click, programmatic).

        Routes to _on_panel_focus_changed when a left-column DataTable
        gains focus, swapping the right panel between SQL and data preview.
        """
        widget = event.widget
        if isinstance(widget, DataTable) and widget.id in self._PANEL_SWAP_IDS:
            self._on_panel_focus_changed(widget)

    # --- Catalog Row Highlighting ---

    @on(DataTable.RowHighlighted, "#catalog-table")
    def _on_catalog_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None:
            # Transient event from table.clear() during refresh.
            # Do NOT reset _current_highlighted_hash here — the cursor
            # will be restored momentarily by move_cursor and we want
            # the dedup guard to suppress the redundant re-render.
            return

        entry_hash = str(event.row_key.value)

        # Skip full re-render when the same entry is re-highlighted.
        # This happens every refresh cycle (clear + move_cursor) and
        # would otherwise destroy the data preview / profiles panels.
        if entry_hash == self._current_highlighted_hash:
            return
        self._current_highlighted_hash = entry_hash

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

        row_data = self._row_cache.get(entry_hash)
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
                    f"{len(schema_in)} in \u00b7 {len(row_data.schema_out)} out"
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
                        ).border_title = "Revisions \u2014 (alias not found)"
                    case _:
                        self.query_one(
                            "#revisions-panel"
                        ).border_title = f"Revisions \u2014 {first_alias}"
                        self._load_revisions_preview(catalog_alias)
            case _:
                self.query_one(
                    "#revisions-panel"
                ).border_title = "Revisions \u2014 (no alias)"

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

    # --- Refresh ---

    @work(thread=True, exit_on_error=False)
    def _do_refresh(self) -> None:
        if not self._refresh_lock.acquire(blocking=False):
            return
        try:
            self._do_refresh_locked()
        finally:
            self._refresh_lock.release()

    @work(thread=True, exit_on_error=False)
    def _do_force_refresh(self) -> None:
        """Refresh that blocks until the lock is available (used after mutations)."""
        with self._refresh_lock:
            self._do_refresh_locked()

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
        new_count = 0
        for entry_hash in new_keys:
            entry = catalog.get_catalog_entry(entry_hash)
            aliases = alias_multimap.get(entry_hash, ())
            row_data = _load_catalog_row(entry, aliases)
            self._row_cache[row_data.row_key] = row_data
            self.app.call_from_thread(self._render_catalog_row, row_data)
            new_count += 1

        if new_count:
            self.app.call_from_thread(self._notify_new_entries, new_count)

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
            filter_tag = " (aliased)" if self._alias_filter else ""
            self.query_one(
                "#catalog-panel"
            ).border_title = f"Expressions \u2014 {catalog_name}{filter_tag}"

            visible_rows = (
                tuple(r for r in cached_rows if r.aliases)
                if self._alias_filter
                else cached_rows
            )

            # If the currently highlighted entry was removed, clear the
            # dedup guard so the new entry at that cursor position gets
            # a full render.
            remaining_keys = frozenset(r.row_key for r in visible_rows)
            if (
                self._current_highlighted_hash is not None
                and self._current_highlighted_hash not in remaining_keys
            ):
                self._current_highlighted_hash = None

            table = self.query_one("#catalog-table", DataTable)
            saved_cursor = table.cursor_row
            table.clear()
            for row_data in visible_rows:
                table.add_row(*row_data.row, key=row_data.row_key)
            count = table.row_count
            if saved_cursor is not None and count > 0:
                table.move_cursor(row=min(saved_cursor, count - 1))

    def _render_catalog_row(self, row_data) -> None:
        with self.app.batch_update():
            table = self.query_one("#catalog-table", DataTable)
            if len(table.columns) < len(COLUMNS):
                return
            if self._alias_filter and not row_data.aliases:
                return
            table.add_row(*row_data.row, key=row_data.row_key)

    def _notify_new_entries(self, count: int) -> None:
        label = "entry" if count == 1 else "entries"
        self.notify(f"+{count} new {label}", timeout=4)
        panel = self.query_one("#catalog-panel")
        panel.border_subtitle = f"+{count} new"
        self.set_timer(5.0, self._clear_new_entries_subtitle)

    def _clear_new_entries_subtitle(self) -> None:
        self.query_one("#catalog-panel").border_subtitle = ""

    def _render_status(self, stamp, repo_path) -> None:
        count = self.query_one("#catalog-table", DataTable).row_count
        total = len(self._row_cache)
        filter_info = f" (filtered {count}/{total})" if self._alias_filter else ""
        self.query_one("#status-bar", Static).update(
            f" {count} entries{filter_info} \u00b7 {repo_path} \u00b7 {stamp}"
        )

    # --- Toggle: Alias Filter ---

    def action_toggle_alias_filter(self) -> None:
        self._alias_filter = not self._alias_filter
        self._current_highlighted_hash = None
        table = self.query_one("#catalog-table", DataTable)
        saved_cursor = table.cursor_row
        table.clear()
        visible = (
            tuple(r for r in self._row_cache.values() if r.aliases)
            if self._alias_filter
            else tuple(self._row_cache.values())
        )
        for row_data in visible:
            table.add_row(*row_data.row, key=row_data.row_key)
        count = table.row_count
        if saved_cursor is not None and count > 0:
            table.move_cursor(row=min(saved_cursor, count - 1))

        catalog = self.app._catalog
        if catalog is not None:
            catalog_name = Path(catalog.repo.working_dir).name
            filter_tag = " (aliased)" if self._alias_filter else ""
            self.query_one(
                "#catalog-panel"
            ).border_title = f"Expressions \u2014 {catalog_name}{filter_tag}"

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
                    " uncached \u2014 run to materialize"
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
                f" Data Preview \u2014 {total_rows} rows (max 50)"
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
            data = yaml12.safe_load(zf.read(member_path))
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

    # --- Toggle: Services ---

    def action_toggle_services(self) -> None:
        services_panel = self.query_one("#services-panel", ServicesPanel)
        services_panel.display = not services_panel.display

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
                        f"{len(rows)} files \u00b7 {_format_size(total_size)}"
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
        # Skip re-render when run data hasn't changed — avoids
        # clearing the table (which resets cursor and re-triggers
        # RowHighlighted, causing inline data preview to flicker).
        fingerprint = tuple(r.row for r in rows)
        if fingerprint == self._current_run_fingerprint:
            return
        self._current_run_fingerprint = fingerprint

        with self.app.batch_update():
            runs_table = self.query_one("#runs-table", DataTable)
            saved_cursor = runs_table.cursor_row
            runs_table.clear()
            self._run_row_cache.clear()
            for i, row_data in enumerate(rows):
                key = str(i)
                runs_table.add_row(*row_data.row, key=key)
                self._run_row_cache[key] = row_data
            count = runs_table.row_count
            if saved_cursor is not None and count > 0:
                runs_table.move_cursor(row=min(saved_cursor, count - 1))
            runs_panel = self.query_one("#runs-panel")
            match rows:
                case ():
                    runs_panel.border_subtitle = "no runs"
                case _:
                    runs_panel.border_subtitle = f"{len(rows)} runs"

    def _show_run_preview(self, run_data: RunRowData) -> bool:
        """Show data preview for a run. Returns True if preview was shown."""
        preview_path = self._find_run_preview_path(run_data)
        match preview_path:
            case str(p):
                self._show_inline_data(p, f"run {run_data.run_id_display}")
                return True
            case _:
                return False

    def _find_run_preview_path(self, run_data: RunRowData) -> str | None:
        """Find the best parquet path for a run's data preview.

        Once found for any run, the path is cached per-entry so all
        runs of the same entry share the same preview file.
        """
        entry_hash = self._current_highlighted_hash

        # Check per-entry cache first
        if entry_hash is not None:
            cached = self._entry_preview_cache.get(entry_hash)
            if cached is not None and Path(cached).exists():
                return cached

        result = self._discover_preview_path(run_data, entry_hash)

        # Cache the result for this entry
        if result is not None and entry_hash is not None:
            self._entry_preview_cache[entry_hash] = result

        return result

    def _discover_preview_path(
        self, run_data: RunRowData, entry_hash: str | None
    ) -> str | None:
        """Search for a parquet preview file across multiple strategies."""
        # 1. This run's snapshot
        match run_data.output_snapshot_path:
            case str(p) if Path(p).exists():
                return p
            case _:
                pass
        # 2. Any sibling run's snapshot
        for sibling in self._run_row_cache.values():
            match sibling.output_snapshot_path:
                case str(p) if Path(p).exists():
                    return p
                case _:
                    pass
        # 3. Entry-level cache paths
        if entry_hash is not None:
            row_data = self._row_cache.get(entry_hash)
            if row_data is not None:
                for p in row_data.entry.parquet_cache_paths:
                    if Path(p).exists():
                        return p
        # 4. Timestamp-based: find a cache file created during ANY run
        #    of this entry (not just the highlighted one)
        try:
            cache_dir = self._get_cache_dir()
            if cache_dir is not None:
                parquet_dir = cache_dir / "parquet"
                if parquet_dir.exists():
                    for run in self._run_row_cache.values():
                        meta_dict = dict(run.meta)
                        started = meta_dict.get("started_at", "")
                        completed = meta_dict.get("completed_at", "")
                        if not started or not completed:
                            continue
                        from datetime import datetime  # noqa: PLC0415

                        t_start = datetime.fromisoformat(started).timestamp() - 2
                        t_end = datetime.fromisoformat(completed).timestamp() + 2
                        for f in parquet_dir.glob("*.parquet"):
                            mtime = f.stat().st_mtime
                            if t_start <= mtime <= t_end:
                                return str(f)
        except Exception:
            pass
        return None

    @on(DataTable.RowHighlighted, "#runs-table")
    def _on_run_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None:
            return
        # Only update panels when runs table has focus. This prevents
        # _render_runs auto-highlighting rows from overriding the
        # catalog entry's SQL/info panels. The on_descendant_focus
        # handler covers the initial click/tab-into-runs-table case.
        if self.app.focused is not self.query_one("#runs-table", DataTable):
            return
        run_data = self._run_row_cache.get(str(event.row_key.value))
        if run_data is None:
            return
        self.query_one("#info-content", Static).update(_format_run_detail(run_data))
        if not self._show_run_preview(run_data):
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
        panel.border_title = f"Data Preview \u2014 {label}"
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
        # Stale worker — panel was hidden while loading
        if not self.query_one("#inline-data-panel").display:
            return
        self.query_one("#inline-data-status", Static).update(
            f" {total_rows:,} rows \u00b7 {len(columns)} cols (showing {len(rows)})"
        )
        table = self.query_one("#inline-data-table", DataTable)
        table.clear(columns=True)
        table.loading = False
        for col in columns:
            table.add_column(col, key=col)
        for i, row in enumerate(rows):
            table.add_row(*row, key=str(i))

    def _render_inline_data_error(self, message: str) -> None:
        if not self.query_one("#inline-data-panel").display:
            return
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

        strategy = get_detail_strategy(row_data.kind)
        actions = strategy.available_actions()

        match actions:
            case ():
                self.query_one("#status-bar", Static).update(
                    " No actions available \u2014 compose with a source first"
                )
                return
            case _:
                pass

        match row_data.aliases:
            case (first_alias, *_):
                entry_name = first_alias
            case _:
                entry_name = entry_hash

        # For now, default to run. ActionChooserModal will handle
        # multiple actions once sink/serve modals are implemented.
        self.app.push_screen(
            RunOptionsScreen(entry_name, entry_hash),
            callback=self._on_run_options_dismissed,
        )

    def _on_run_options_dismissed(self, config: RunConfig | None) -> None:
        if config is None:
            return
        self.query_one("#runs-panel").border_subtitle = "running..."
        self._execute_run(config)

    # --- Telemetry ---

    def action_show_telemetry(self) -> None:
        runs_table = self.query_one("#runs-table", DataTable)
        if runs_table.row_count == 0:
            self.query_one("#status-bar", Static).update(" No runs to show")
            return
        row_key, _ = runs_table.coordinate_to_cell_key(runs_table.cursor_coordinate)
        run_data = self._run_row_cache.get(str(row_key.value))
        if run_data is None:
            return
        entry_hash = self._current_highlighted_hash
        if entry_hash is None:
            return
        self.app.push_screen(
            TelemetryScreen(
                run_id=run_data.run_id,
                expr_hash=entry_hash,
            )
        )

    # --- Clear Runs ---

    def action_clear_runs(self) -> None:
        entry_hash = self._current_highlighted_hash
        if entry_hash is None:
            self.query_one("#status-bar", Static).update(" No entry selected")
            return
        self._do_clear_runs(entry_hash)

    @work(thread=True, exit_on_error=False)
    def _do_clear_runs(self, entry_hash: str) -> None:
        import shutil  # noqa: PLC0415

        from xorq.common.utils.logging_utils import get_xorq_runs_dir  # noqa: PLC0415

        runs_dir = get_xorq_runs_dir() / entry_hash
        if runs_dir.exists():
            shutil.rmtree(runs_dir)

        self._current_run_fingerprint = ()
        self._entry_preview_cache.pop(entry_hash, None)
        run_rows = _build_run_rows(entry_hash)
        self.app.call_from_thread(self._render_runs, run_rows)
        self.app.call_from_thread(
            self.query_one("#status-bar", Static).update,
            f" Cleared runs for {entry_hash[:12]}",
        )

    # --- Compose ---

    def action_compose(self) -> None:
        available = self._build_compose_entries()
        if not available:
            self.query_one("#status-bar", Static).update(
                " No catalog entries available"
            )
            return
        self.app.push_screen(
            ComposeScreen(available),
            callback=self._on_compose_dismissed,
        )

    def _build_compose_entries(self) -> tuple[tuple[str, str, str, dict, dict], ...]:
        """Build (display_name, kind, hash, schema_in, schema_out) tuples."""
        alias_multimap = _build_alias_multimap(self.catalog_aliases)
        return tuple(
            (
                alias_multimap.get(row_data.hash, (row_data.hash,))[0],
                row_data.kind,
                row_data.hash,
                dict(row_data.schema_in) if row_data.schema_in else {},
                dict(row_data.schema_out),
            )
            for row_data in sorted(self._row_cache.values(), key=lambda r: r.sort_key)
        )

    def _on_compose_dismissed(self, config: ComposeConfig | None) -> None:
        if config is None:
            return
        self.query_one("#status-bar", Static).update(" Composing...")
        self._execute_compose(config)

    @work(thread=True, exit_on_error=False)
    def _execute_compose(self, config: ComposeConfig) -> None:
        from xorq.catalog.composer import ExprComposer  # noqa: PLC0415
        from xorq.ibis_yaml.compiler import build_expr  # noqa: PLC0415

        catalog = self.app._catalog
        if catalog is None:
            return

        try:
            resolved = tuple(
                catalog.get_catalog_entry(name, maybe_alias=True)
                for name in config.entries
            )
            expr = ExprComposer(
                source=resolved[0],
                transforms=resolved[1:],
                code=config.code,
            ).expr

            build_path = build_expr(expr)
            entry_name = build_path.name
            aliases = (config.alias,) if config.alias else ()
            if catalog.contains(entry_name):
                if config.alias:
                    catalog.add_alias(entry_name, config.alias)
            else:
                catalog.add(build_path, aliases=aliases)

            # Build row data on worker thread so the main thread can insert
            # it directly into the DataTable without another refresh round-trip.
            _invalidate_catalog_caches()
            entry = catalog.get_catalog_entry(entry_name)
            alias_multimap = _build_alias_multimap(_get_catalog_aliases(catalog))
            entry_aliases = alias_multimap.get(entry_name, ())
            row_data = _load_catalog_row(entry, entry_aliases)

            label = config.alias or entry_name[:12]
            self.app.call_from_thread(
                self._render_compose_result, f"Composed as {label!r}", None, row_data
            )
        except Exception as e:
            import traceback  # noqa: PLC0415

            tb = traceback.format_exception(e)
            self.app.call_from_thread(
                self._render_compose_result, f"Compose error: {e}", "".join(tb)
            )

    def _render_compose_result(
        self,
        message: str,
        error_detail: str | None,
        row_data: CatalogRowData | None = None,
    ) -> None:
        self.query_one("#status-bar", Static).update(f" {message}")
        if error_detail is not None:
            self.query_one("#info-panel").border_title = "Compose Error"
            self.query_one("#info-content", Static).update(error_detail)
        else:
            if row_data is not None:
                self._row_cache[row_data.row_key] = row_data
                self._render_catalog_row(row_data)
            self._do_force_refresh()

    # --- Delete Entry / Remove Alias ---

    def _get_selected_entry(self) -> tuple[str, CatalogRowData] | None:
        """Return (entry_hash, row_data) for the currently highlighted row."""
        table = self.query_one("#catalog-table", DataTable)
        if table.row_count == 0:
            return None
        row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
        entry_hash = str(row_key.value)
        row_data = self._row_cache.get(entry_hash)
        if row_data is None:
            return None
        return entry_hash, row_data

    def action_delete_entry(self) -> None:
        sel = self._get_selected_entry()
        if sel is None:
            return
        entry_hash, row_data = sel
        label = row_data.aliases_display or entry_hash[:12]
        aliases_note = (
            f"\n aliases: {row_data.aliases_display}" if row_data.aliases else ""
        )
        self.app.push_screen(
            ConfirmScreen(
                f" delete {label}",
                f" {entry_hash[:12]}{aliases_note}",
            ),
            callback=lambda confirmed: self._on_delete_confirmed(confirmed, entry_hash),
        )

    def _on_delete_confirmed(self, confirmed: bool | None, entry_hash: str) -> None:
        if not confirmed:
            return
        self.query_one("#status-bar", Static).update(" Deleting...")
        self._execute_delete(entry_hash)

    @work(thread=True, exit_on_error=False)
    def _execute_delete(self, entry_hash: str) -> None:
        catalog = self.app._catalog
        if catalog is None:
            return
        try:
            catalog.remove(entry_hash, sync=False)
            self.app.call_from_thread(self._render_delete_result, entry_hash, None)
        except Exception as e:
            self.app.call_from_thread(self._render_delete_result, entry_hash, str(e))

    def _render_delete_result(self, entry_hash: str, error: str | None) -> None:
        if error is not None:
            self.query_one("#status-bar", Static).update(f" Delete error: {error}")
            return
        # Remove from row cache and DataTable immediately.
        self._row_cache.pop(entry_hash, None)
        table = self.query_one("#catalog-table", DataTable)
        try:
            table.remove_row(entry_hash)
        except Exception:
            pass
        self._current_highlighted_hash = None
        self.query_one("#status-bar", Static).update(" Deleted")
        _invalidate_catalog_caches()
        self._do_force_refresh()

    def action_remove_alias(self) -> None:
        sel = self._get_selected_entry()
        if sel is None:
            return
        entry_hash, row_data = sel
        if not row_data.aliases:
            self.query_one("#status-bar", Static).update(" No aliases on this entry")
            return
        alias = row_data.aliases[0]
        self.app.push_screen(
            ConfirmScreen(
                f" remove alias {alias!r}",
                f" from {entry_hash[:12]}",
            ),
            callback=lambda confirmed: self._on_remove_alias_confirmed(
                confirmed, alias
            ),
        )

    def _on_remove_alias_confirmed(self, confirmed: bool | None, alias: str) -> None:
        if not confirmed:
            return
        self.query_one("#status-bar", Static).update(" Removing alias...")
        self._execute_remove_alias(alias)

    @work(thread=True, exit_on_error=False)
    def _execute_remove_alias(self, alias: str) -> None:
        from xorq.catalog.catalog import CatalogAlias  # noqa: PLC0415

        catalog = self.app._catalog
        if catalog is None:
            return
        try:
            CatalogAlias.from_name(alias, catalog).remove()
            self.app.call_from_thread(self._render_remove_alias_result, alias, None)
        except Exception as e:
            self.app.call_from_thread(self._render_remove_alias_result, alias, str(e))

    def _render_remove_alias_result(self, alias: str, error: str | None) -> None:
        if error is not None:
            self.query_one("#status-bar", Static).update(
                f" Remove alias error: {error}"
            )
            return
        self.query_one("#status-bar", Static).update(f" Removed alias {alias!r}")
        _invalidate_catalog_caches()
        self._do_force_refresh()

    @work(thread=True, exit_on_error=False)
    def _execute_run(self, config: RunConfig) -> None:
        import subprocess  # noqa: PLC0415

        catalog = self.app._catalog
        if catalog is None:
            return

        try:
            catalog_path = catalog.repo.working_dir

            # Delegate to the CLI — identical execution, telemetry,
            # and run logging as `xorq catalog run`.
            from xorq.common.utils.caching_utils import (  # noqa: PLC0415
                get_xorq_cache_dir,
            )

            snapshots_dir = get_xorq_cache_dir() / "snapshots"
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = str(snapshots_dir / f"{config.expr_hash}.parquet")

            cmd = [
                "xorq",
                "catalog",
                "--path",
                catalog_path,
                "run",
                config.entry_name,
                "-o",
                snapshot_path,
                "-f",
                "parquet",
            ]

            import os  # noqa: PLC0415

            from xorq.common.utils.trace_utils import default_log_path  # noqa: PLC0415

            env = {**os.environ, "OTEL_EXPORTER_CONSOLE_FALLBACK": "1"}
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )

            # Append captured OTEL spans (stdout from ConsoleSpanExporter
            # fallback) to the otel log file for the telemetry viewer.
            if result.stdout.strip():
                default_log_path.parent.mkdir(parents=True, exist_ok=True)
                with default_log_path.open("a", encoding="utf-8") as f:
                    f.write(result.stdout)

            # Patch the newest run's meta.json with TUI-specific fields
            # that the CLI doesn't store (cache_type, output_snapshot_path).
            try:
                import json as _json  # noqa: PLC0415

                from xorq.common.utils.logging_utils import (  # noqa: PLC0415
                    Runs,
                    get_xorq_runs_dir,
                )

                runs = Runs(expr_dir=get_xorq_runs_dir() / config.expr_hash)
                if runs.runs:
                    newest = runs.runs[0]
                    meta = newest.read_meta()
                    if meta is not None:
                        meta["cache_type"] = config.cache_type
                        meta["output_snapshot_path"] = snapshot_path
                        if config.ttl is not None:
                            meta["ttl"] = config.ttl
                        newest._meta_path.write_text(_json.dumps(meta, indent=2) + "\n")
            except Exception:
                pass

            if result.returncode == 0:
                status = "ok"
                detail = result.stderr.strip() or "Run completed"
            else:
                status = "error"
                detail = result.stderr.strip() or "Unknown error"
        except subprocess.TimeoutExpired:
            status = "error"
            detail = "Run timed out (300s)"
        except Exception as e:
            status = "error"
            detail = str(e)

        run_rows = _build_run_rows(config.expr_hash)
        match status:
            case "ok":
                message = (
                    f"Run completed \u00b7 {detail}" if detail else "Run completed"
                )
            case _:
                message = f"Run {status} \u00b7 {detail}" if detail else f"Run {status}"
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
                # Show run detail info + inline data preview when available
                if runs_table.row_count > 0:
                    row_key, _ = runs_table.coordinate_to_cell_key(
                        runs_table.cursor_coordinate
                    )
                    run_data = self._run_row_cache.get(str(row_key.value))
                    if run_data is not None:
                        self.query_one("#info-content", Static).update(
                            _format_run_detail(run_data)
                        )
                        if self._show_run_preview(run_data):
                            return
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
