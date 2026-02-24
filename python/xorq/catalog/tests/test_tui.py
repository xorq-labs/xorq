"""Tests for the catalog TUI using Textual's Pilot test driver.

Strategy:
- Data classes and helpers: pure unit tests, no Pilot needed.
- Screen composition & navigation: mock the catalog at the boundary
  so we test the UI in isolation from real git repos / tarball I/O.
- Integration: optionally use the conftest catalog fixtures for
  end-to-end smoke tests.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from textual.widgets import DataTable, Static, TabbedContent

from xorq.catalog.tui import (
    COLUMNS,
    REFLOG_COLUMNS,
    CatalogRowData,
    CatalogScreen,
    CatalogTUI,
    ExploreData,
    ExploreScreen,
    GitLogRowData,
    RevisionRowData,
    _format_cached,
    _format_column_count,
    maybe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine in a fresh event loop (avoids pytest-asyncio)."""
    return asyncio.run(coro)


def _make_mock_entry(name="abc123", has_metadata=False):
    entry = MagicMock()
    entry.name = name
    entry.catalog_path = f"/tmp/fake/{name}.tgz"
    entry.metadata_path = MagicMock()
    entry.metadata_path.exists.return_value = has_metadata
    entry.expr = None
    return entry


def _make_mock_catalog(entries=(), aliases=()):
    catalog = MagicMock()
    catalog.repo.working_dir = "/tmp/fake-catalog"
    catalog.repo.head.log.return_value = []
    catalog.list.return_value = [e.name for e in entries]
    catalog.catalog_aliases = aliases
    for entry in entries:
        catalog.get_catalog_entry.side_effect = (
            lambda h, _entries={e.name: e for e in entries}: _entries[h]
        )
    return catalog


def _make_tui(catalog=None):
    if catalog is None:
        catalog = _make_mock_catalog()
    app = CatalogTUI(lambda: catalog)
    return app


SAMPLE_ROWS = (
    CatalogRowData(
        kind="expr",
        alias="my-model",
        hash="abc123",
        backends=("duckdb",),
        column_count=5,
        cached=True,
        tags=("v1", "latest"),
    ),
    CatalogRowData(
        kind="expr",
        alias="",
        hash="def456",
        backends=("postgres", "duckdb"),
        column_count=None,
        cached=False,
        tags=(),
    ),
)


# ---------------------------------------------------------------------------
# 1. Pure unit tests: helpers
# ---------------------------------------------------------------------------


class TestFormatCached:
    def test_true(self):
        assert _format_cached(True) == "●"

    def test_false(self):
        assert _format_cached(False) == "○"

    def test_none(self):
        assert _format_cached(None) == "—"


class TestFormatColumnCount:
    def test_none(self):
        assert _format_column_count(None) == "?"

    def test_int(self):
        assert _format_column_count(5) == "5 cols"

    def test_zero(self):
        assert _format_column_count(0) == "0 cols"


class TestMaybeDecorator:
    def test_returns_value_on_success(self):
        @maybe(default="fallback")
        def ok():
            return "good"

        assert ok() == "good"

    def test_returns_default_on_exception(self):
        @maybe(default="fallback")
        def bad():
            raise ValueError("boom")

        assert bad() == "fallback"


# ---------------------------------------------------------------------------
# 2. Pure unit tests: frozen data classes
# ---------------------------------------------------------------------------


class TestCatalogRowData:
    def test_row_tuple(self):
        row = SAMPLE_ROWS[0]
        assert row.row == (
            "expr",
            "my-model",
            "abc123",
            "duckdb",
            "5 cols",
            "●",
            "v1, latest",
        )

    def test_empty_fields(self):
        row = CatalogRowData()
        assert row.row == ("expr", "", "", "", "?", "—", "")

    def test_backends_deduped_and_sorted(self):
        row = CatalogRowData(backends=("duckdb", "postgres", "duckdb"))
        assert row.backends_display == "duckdb, postgres"

    def test_row_key_with_alias(self):
        row = SAMPLE_ROWS[0]
        assert row.row_key == "abc123|my-model"

    def test_row_key_without_alias(self):
        row = SAMPLE_ROWS[1]
        assert row.row_key == "def456"

    def test_frozen(self):
        row = SAMPLE_ROWS[0]
        with pytest.raises(AttributeError):
            row.alias = "new-name"


class TestGitLogRowData:
    def test_row_tuple(self):
        row = GitLogRowData(hash="abc123", date="2025-01-01", message="init")
        assert row.row == ("abc123", "2025-01-01", "init")


class TestRevisionRowData:
    def test_current(self):
        row = RevisionRowData(
            hash="abc123",
            column_count=3,
            cached=True,
            commit_date="2025-01-01",
            is_current=True,
        )
        assert row.status_display == "CURRENT →"
        assert row.row[0] == "CURRENT →"

    def test_not_current(self):
        row = RevisionRowData(hash="def456", is_current=False)
        assert row.status_display == ""


class TestExploreData:
    def test_frozen(self):
        data = ExploreData(hash="abc123", alias="test")
        with pytest.raises(AttributeError):
            data.hash = "new"


# ---------------------------------------------------------------------------
# 3. Pilot tests: CatalogTUI app
# ---------------------------------------------------------------------------


class TestCatalogTUIMount:
    def test_app_starts_and_pushes_catalog_screen(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                assert isinstance(app.screen, CatalogScreen)

        _run(_test())

    def test_app_has_custom_theme(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                assert app.theme == "xorq-dark"

        _run(_test())


class TestCatalogScreenComposition:
    def test_tables_have_correct_columns(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                catalog_table = app.screen.query_one("#catalog-table", DataTable)
                assert (
                    tuple(col.label.plain for col in catalog_table.columns.values())
                    == COLUMNS
                )

                log_table = app.screen.query_one("#log-table", DataTable)
                assert (
                    tuple(col.label.plain for col in log_table.columns.values())
                    == REFLOG_COLUMNS
                )

        _run(_test())

    def test_status_bar_exists(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                status = app.screen.query_one("#status-bar", Static)
                assert status is not None

        _run(_test())

    def test_panel_border_titles(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                # Before refresh completes, catalog-panel title is "Expressions"
                # (it gets updated to "Expressions — <name>" after refresh)
                catalog_panel = app.screen.query_one("#catalog-panel")
                assert "Expressions" in str(catalog_panel.border_title)

                log_panel = app.screen.query_one("#log-panel")
                assert log_panel.border_title == "Git Reflog"

        _run(_test())


class TestCatalogScreenNavigation:
    def _populate_table(self, screen):
        """Insert sample rows into the catalog table for navigation tests."""
        table = screen.query_one("#catalog-table", DataTable)
        for row_data in SAMPLE_ROWS:
            table.add_row(*row_data.row, key=row_data.row_key)
        screen._row_cache = {r.row_key: r for r in SAMPLE_ROWS}

    def test_quit_exits_app(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.press("q")
                # After pressing q, the app should be exiting
                # (run_test handles this gracefully)

        _run(_test())

    def test_j_k_moves_cursor(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                self._populate_table(app.screen)
                table = app.screen.query_one("#catalog-table", DataTable)
                assert table.cursor_row == 0

                await pilot.press("j")
                assert table.cursor_row == 1

                await pilot.press("k")
                assert table.cursor_row == 0

        _run(_test())

    def test_explore_on_empty_table_is_noop(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                # Table is empty, pressing enter should not crash
                await pilot.press("enter")
                assert isinstance(app.screen, CatalogScreen)

        _run(_test())


class TestExploreScreenComposition:
    def test_explore_screen_has_tabs(self):
        async def _test():
            entry = _make_mock_entry("abc123")
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                # Push ExploreScreen directly
                app.push_screen(ExploreScreen(entry, "my-model"))
                await pilot.pause()

                tabs = app.screen.query_one("#explore-tabs", TabbedContent)
                assert tabs is not None

        _run(_test())

    def test_explore_screen_breadcrumb(self):
        async def _test():
            entry = _make_mock_entry("abcdef123456")
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.push_screen(ExploreScreen(entry, "my-model"))
                await pilot.pause()

                breadcrumb = app.screen.query_one("#breadcrumb", Static)
                text = breadcrumb.content
                assert "abcdef123456" in text
                assert "my-model" in text

        _run(_test())

    def test_explore_screen_schema_table_has_columns(self):
        async def _test():
            entry = _make_mock_entry()
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.push_screen(ExploreScreen(entry, ""))
                await pilot.pause()

                schema_table = app.screen.query_one("#schema-table", DataTable)
                col_labels = tuple(
                    col.label.plain for col in schema_table.columns.values()
                )
                assert col_labels == ("NAME", "TYPE")

        _run(_test())

    def test_explore_screen_back_with_q(self):
        async def _test():
            entry = _make_mock_entry()
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.push_screen(ExploreScreen(entry, ""))
                await pilot.pause()
                assert isinstance(app.screen, ExploreScreen)

                await pilot.press("q")
                await pilot.pause()
                assert isinstance(app.screen, CatalogScreen)

        _run(_test())

    def test_explore_screen_back_with_escape(self):
        async def _test():
            entry = _make_mock_entry()
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.push_screen(ExploreScreen(entry, ""))
                await pilot.pause()
                assert isinstance(app.screen, ExploreScreen)

                await pilot.press("escape")
                await pilot.pause()
                assert isinstance(app.screen, CatalogScreen)

        _run(_test())


class TestExploreScreenTabNavigation:
    def test_number_keys_switch_tabs(self):
        async def _test():
            entry = _make_mock_entry()
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.push_screen(ExploreScreen(entry, ""))
                await pilot.pause()

                tabs = app.screen.query_one("#explore-tabs", TabbedContent)

                # Tab 1 = Schema (default)
                await pilot.press("1")
                await pilot.pause()
                assert tabs.active == "pane-schema"

                # Tab 4 = Info (always enabled)
                await pilot.press("4")
                await pilot.pause()
                assert tabs.active == "pane-info"

                # Tab 5 = Profiles (always enabled)
                await pilot.press("5")
                await pilot.pause()
                assert tabs.active == "pane-profiles"

        _run(_test())

    def test_disabled_data_tab_shows_notification(self):
        async def _test():
            entry = _make_mock_entry()
            app = _make_tui()
            async with app.run_test(size=(120, 40), notifications=True) as pilot:
                await pilot.pause()
                # Push explore with no cached data => Data tab disabled
                app.push_screen(ExploreScreen(entry, ""))
                await pilot.pause()

                await pilot.press("2")
                await pilot.pause()
                # Tab should not have switched (still on schema or wherever)
                tabs = app.screen.query_one("#explore-tabs", TabbedContent)
                assert tabs.active != "pane-data"

        _run(_test())

    def test_disabled_revisions_tab_without_alias(self):
        async def _test():
            entry = _make_mock_entry()
            app = _make_tui()
            async with app.run_test(size=(120, 40), notifications=True) as pilot:
                await pilot.pause()
                # No alias => Revisions tab disabled
                app.push_screen(ExploreScreen(entry, ""))
                await pilot.pause()

                await pilot.press("3")
                await pilot.pause()
                tabs = app.screen.query_one("#explore-tabs", TabbedContent)
                assert tabs.active != "pane-revisions"

        _run(_test())


class TestExploreScreenRender:
    def test_render_explore_populates_schema(self):
        async def _test():
            entry = _make_mock_entry()
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = ExploreScreen(entry, "test-alias")
                app.push_screen(screen)
                await pilot.pause()

                # Manually call _render_explore with test data
                data = ExploreData(
                    hash="abc123",
                    alias="test-alias",
                    schema_items=(("id", "int64"), ("name", "string")),
                    lineage_text="source → filter → output",
                    is_cached=False,
                    has_alias=True,
                )
                screen._render_explore(data)
                await pilot.pause()

                schema_table = screen.query_one("#schema-table", DataTable)
                # 2 schema rows + possibly 0 if the worker also ran
                assert schema_table.row_count >= 2

        _run(_test())

    def test_render_explore_uncached_disables_data(self):
        async def _test():
            entry = _make_mock_entry()
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = ExploreScreen(entry, "")
                app.push_screen(screen)
                await pilot.pause()

                data = ExploreData(
                    hash="abc123",
                    alias="",
                    is_cached=False,
                    has_alias=False,
                )
                screen._render_explore(data)
                await pilot.pause()

                status = screen.query_one("#data-status", Static)
                assert "uncached" in status.content

        _run(_test())

    def test_render_explore_cached_enables_data_tab(self):
        async def _test():
            entry = _make_mock_entry()
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = ExploreScreen(entry, "test-alias")
                app.push_screen(screen)
                await pilot.pause()

                data = ExploreData(
                    hash="abc123",
                    alias="test-alias",
                    is_cached=True,
                    cache_path="/tmp/cache/abc123",
                    has_alias=True,
                )
                screen._render_explore(data)
                await pilot.pause()

                from textual.widgets import TabPane

                pane = screen.query_one("#pane-data", TabPane)
                assert not pane.disabled

        _run(_test())

    def test_render_explore_shows_metadata_when_present(self):
        async def _test():
            entry = _make_mock_entry()
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = ExploreScreen(entry, "")
                app.push_screen(screen)
                await pilot.pause()

                data = ExploreData(
                    hash="abc123",
                    alias="",
                    metadata=(("author", "alice"), ("version", "1.0")),
                )
                screen._render_explore(data)
                await pilot.pause()

                section = screen.query_one("#metadata-section")
                assert section.display is True

        _run(_test())

    def test_render_explore_hides_metadata_when_empty(self):
        async def _test():
            entry = _make_mock_entry()
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = ExploreScreen(entry, "")
                app.push_screen(screen)
                await pilot.pause()

                data = ExploreData(hash="abc123", alias="", metadata=())
                screen._render_explore(data)
                await pilot.pause()

                section = screen.query_one("#metadata-section")
                assert section.display is False

        _run(_test())


class TestCatalogScreenRefresh:
    def test_render_refresh_populates_tables(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, CatalogScreen)

                reflog_rows = (
                    GitLogRowData(hash="aaa", date="2025-01-01", message="init"),
                    GitLogRowData(hash="bbb", date="2025-01-02", message="add entry"),
                )
                screen._render_refresh_start(reflog_rows, "/tmp/fake", SAMPLE_ROWS)
                await pilot.pause()

                catalog_table = screen.query_one("#catalog-table", DataTable)
                assert catalog_table.row_count == len(SAMPLE_ROWS)

                log_table = screen.query_one("#log-table", DataTable)
                assert log_table.row_count == len(reflog_rows)

        _run(_test())

    def test_render_refresh_uses_row_key(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, CatalogScreen)

                screen._render_refresh_start((), "/tmp/fake", SAMPLE_ROWS)
                await pilot.pause()

                table = screen.query_one("#catalog-table", DataTable)
                keys = [str(k.value) for k in table.rows.keys()]
                assert "abc123|my-model" in keys
                assert "def456" in keys

        _run(_test())

    def test_render_refresh_done_updates_status(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = app.screen
                screen._saved_cursor = 0
                screen._render_refresh_done("12:00:00", "/tmp/fake")
                await pilot.pause()

                status = screen.query_one("#status-bar", Static)
                text = status.content
                assert "12:00:00" in text
                assert "/tmp/fake" in text

        _run(_test())


class TestMultipleAliases:
    """Two aliases for the same hash should produce two distinct table rows."""

    def test_two_aliases_same_hash(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, CatalogScreen)

                row_v1 = CatalogRowData(
                    kind="expr",
                    alias="v1",
                    hash="abc123",
                    backends=("duckdb",),
                    column_count=5,
                    cached=True,
                )
                row_latest = CatalogRowData(
                    kind="expr",
                    alias="latest",
                    hash="abc123",
                    backends=("duckdb",),
                    column_count=5,
                    cached=True,
                )

                screen._render_refresh_start((), "/tmp/fake", (row_v1, row_latest))
                await pilot.pause()

                table = screen.query_one("#catalog-table", DataTable)
                assert table.row_count == 2

                keys = [str(k.value) for k in table.rows.keys()]
                assert "abc123|v1" in keys
                assert "abc123|latest" in keys

        _run(_test())

    def test_unaliased_row_uses_hash_as_key(self):
        async def _test():
            app = _make_tui()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = app.screen

                row = CatalogRowData(kind="expr", alias="", hash="def456")
                screen._render_refresh_start((), "/tmp/fake", (row,))
                await pilot.pause()

                table = screen.query_one("#catalog-table", DataTable)
                assert table.row_count == 1
                keys = [str(k.value) for k in table.rows.keys()]
                assert "def456" in keys

        _run(_test())
