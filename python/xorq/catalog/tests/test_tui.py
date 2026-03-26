"""Tests for the catalog TUI using Textual's Pilot test driver.

Strategy:
- Format helpers and frozen data classes: pure unit tests, no catalog needed.
- Screen composition, navigation, rendering: use a real Catalog backed by a
  temporary git repo so that CatalogEntry objects carry genuine expr_metadata,
  backends, and column info loaded from the zip archive.
- Git log: use the real repo that backs the catalog fixture.

IMPORTANT — populating the catalog table in pilot tests:
    Never wait for the async _do_refresh worker to populate rows.  It runs in
    a background thread on a timer and is inherently racy under test.  Instead,
    build CatalogRowData objects and call _render_refresh() directly — see the
    _populate_table() helper below.
"""

import asyncio
from pathlib import Path

import pytest
from textual.widgets import DataTable, Static, TabbedContent, TabPane

import xorq.api as xo
from xorq.caching import ParquetSnapshotCache
from xorq.catalog.tui import (
    ALIAS_COLUMNS,
    COLUMNS,
    GIT_LOG_COLUMNS,
    SCHEMA_PREVIEW_COLUMNS,
    CatalogRowData,
    CatalogScreen,
    CatalogTUI,
    ExploreData,
    ExploreScreen,
    GitLogRowData,
    RevisionRowData,
    _build_git_log_rows,
    _entry_info,
    _format_cached,
    _format_column_count,
    maybe,
)
from xorq.common.utils.defer_utils import deferred_read_parquet


def _run(coro):
    """Run an async coroutine in a fresh event loop (avoids pytest-asyncio)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def entry_a(catalog):
    """A three-column bound expression: id (int), name (str), score (float)."""
    expr = xo.memtable({"id": [1, 2], "name": ["alice", "bob"], "score": [9.5, 8.1]})
    return catalog.add(expr)


@pytest.fixture
def entry_b(catalog):
    """A single-column bound expression: value (int)."""
    expr = xo.memtable({"value": [10, 20, 30]})
    return catalog.add(expr)


@pytest.fixture
def entry_cached(catalog, tmp_path):
    """A memtable expression wrapped with ParquetSnapshotCache."""
    cache = ParquetSnapshotCache.from_kwargs(relative_path=tmp_path / "cache")
    expr = xo.memtable({"x": [1, 2, 3], "y": [4, 5, 6]}).cache(cache=cache)
    return catalog.add(expr)


@pytest.fixture
def alias_for_a(catalog, entry_a):
    """Add alias 'my-model' to entry_a and return the alias string."""
    alias = "my-model"
    catalog.add_alias(entry_a.name, alias)
    return alias


def _make_tui(catalog):
    return CatalogTUI(lambda: catalog)


async def _populate_table(pilot, catalog, *entries):
    """Deterministically populate the catalog table with the given entries.

    Use this instead of waiting for the async _do_refresh worker, which is
    racy under test.  Returns the CatalogScreen and the list of CatalogRowData.
    """
    await pilot.pause()
    screen = pilot.app.screen
    rows = tuple(CatalogRowData(entry=e) for e in entries)
    screen._render_refresh(catalog.repo.working_dir, rows)
    await pilot.pause()
    return screen, rows


# ---------------------------------------------------------------------------
# 1. Pure unit tests: format helpers
# ---------------------------------------------------------------------------


def test_format_cached_true():
    assert _format_cached(True) == "●"


def test_format_cached_false():
    assert _format_cached(False) == "○"


def test_format_cached_none():
    assert _format_cached(None) == "—"


def test_format_column_count_none():
    assert _format_column_count(None) == "?"


def test_format_column_count_int():
    assert _format_column_count(5) == "5 cols"


def test_format_column_count_zero():
    assert _format_column_count(0) == "0 cols"


def test_maybe_returns_value_on_success():
    @maybe(default="fallback")
    def ok():
        return "good"

    assert ok() == "good"


def test_maybe_returns_default_on_exception():
    @maybe(default="fallback")
    def bad():
        raise ValueError("boom")

    assert bad() == "fallback"


# ---------------------------------------------------------------------------
# 2. Unit tests: frozen data classes backed by real catalog entries
# ---------------------------------------------------------------------------


def test_row_shape(entry_a, alias_for_a):
    row = CatalogRowData(entry=entry_a, aliases=(alias_for_a,))
    kind, alias, hash_, backends, output, cached, root_tag = row.row
    assert kind == "source"
    assert alias == alias_for_a
    assert hash_ == entry_a.name
    assert isinstance(backends, str)
    assert output == "3 cols"
    assert cached == "—"  # simple memtable has no ParquetSnapshotCache
    assert root_tag == ""


def test_cached_is_none_for_plain_memtable(entry_a):
    row = CatalogRowData(entry=entry_a)
    assert row.cached is None


def test_column_count_single_column(entry_b):
    row = CatalogRowData(entry=entry_b)
    assert row.column_count == 1
    assert row.output_display == "1 cols"


def test_row_key_is_entry_name(entry_a, entry_b):
    row_a = CatalogRowData(entry=entry_a)
    row_b = CatalogRowData(entry=entry_b)
    assert row_a.row_key == entry_a.name
    assert row_b.row_key == entry_b.name
    assert row_a.row_key != row_b.row_key


def test_cached_with_parquet_snapshot(entry_cached):
    row = CatalogRowData(entry=entry_cached)
    assert row.cached is False
    assert row.cached_display == "○"

    entry_cached.expr.execute()

    row_after = CatalogRowData(entry=entry_cached)
    assert row_after.cached is True
    assert row_after.cached_display == "●"

    _, alias, _, _, _, cached_field, _ = row_after.row
    assert cached_field == "●"


def test_catalog_row_data_is_frozen(entry_a):
    row = CatalogRowData(entry=entry_a)
    with pytest.raises(AttributeError):
        row.aliases = ("new-name",)


def test_revision_row_data_current():
    row = RevisionRowData(
        hash="abc123",
        column_count=3,
        cached=True,
        commit_date="2025-01-01",
        is_current=True,
    )
    assert row.status_display == "CURRENT →"
    assert row.row[0] == "CURRENT →"


def test_revision_row_data_not_current():
    row = RevisionRowData(hash="def456", is_current=False)
    assert row.status_display == ""


def test_explore_data_is_frozen():
    data = ExploreData(hash="abc123", alias="test")
    with pytest.raises(AttributeError):
        data.hash = "new"


# ---------------------------------------------------------------------------
# 3. Pilot tests: CatalogTUI app
# ---------------------------------------------------------------------------


def test_app_starts_and_pushes_catalog_screen(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, CatalogScreen)

    _run(_test())


def test_app_has_custom_theme(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.theme == "xorq-dark"

    _run(_test())


def test_tables_have_correct_columns(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            catalog_table = app.screen.query_one("#catalog-table", DataTable)
            assert (
                tuple(col.label.plain for col in catalog_table.columns.values())
                == COLUMNS
            )

            schema_table = app.screen.query_one("#schema-preview-table", DataTable)
            assert (
                tuple(col.label.plain for col in schema_table.columns.values())
                == SCHEMA_PREVIEW_COLUMNS
            )

    _run(_test())


def test_status_bar_exists(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            status = app.screen.query_one("#status-bar", Static)
            assert status is not None

    _run(_test())


def test_panel_border_titles(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            catalog_panel = app.screen.query_one("#catalog-panel")
            assert "Expressions" in str(catalog_panel.border_title)

            schema_panel = app.screen.query_one("#schema-panel")
            assert schema_panel.border_title == "Schema"

    _run(_test())


def test_quit_exits_app(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("q")

    _run(_test())


def test_j_k_moves_cursor(catalog, entry_a, entry_b):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_table(pilot, catalog, entry_a, entry_b)

            table = screen.query_one("#catalog-table", DataTable)
            assert table.row_count == 2
            assert table.cursor_row == 0

            await pilot.press("j")
            assert table.cursor_row == 1

            await pilot.press("k")
            assert table.cursor_row == 0

    _run(_test())


def test_explore_on_empty_table_is_noop(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            assert isinstance(app.screen, CatalogScreen)

    _run(_test())


def test_explore_screen_has_tabs(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ExploreScreen(entry_a, "my-model"))
            await pilot.pause()

            tabs = app.screen.query_one("#explore-tabs", TabbedContent)
            assert tabs is not None

    _run(_test())


def test_explore_screen_breadcrumb(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ExploreScreen(entry_a, "my-model"))
            await pilot.pause()

            breadcrumb = app.screen.query_one("#breadcrumb", Static)
            text = breadcrumb.content
            assert entry_a.name[:12] in text
            assert "my-model" in text

    _run(_test())


def test_explore_screen_schema_table_has_columns(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ExploreScreen(entry_a, ""))
            await pilot.pause()

            schema_table = app.screen.query_one("#schema-table", DataTable)
            col_labels = tuple(col.label.plain for col in schema_table.columns.values())
            assert col_labels == ("NAME", "TYPE")

    _run(_test())


def test_explore_screen_back_with_q(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ExploreScreen(entry_a, ""))
            await pilot.pause()
            assert isinstance(app.screen, ExploreScreen)

            await pilot.press("q")
            await pilot.pause()
            assert isinstance(app.screen, CatalogScreen)

    _run(_test())


def test_explore_screen_back_with_escape(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ExploreScreen(entry_a, ""))
            await pilot.pause()
            assert isinstance(app.screen, ExploreScreen)

            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, CatalogScreen)

    _run(_test())


def test_number_keys_switch_tabs(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ExploreScreen(entry_a, ""))
            await pilot.pause()

            tabs = app.screen.query_one("#explore-tabs", TabbedContent)

            await pilot.press("2")
            await pilot.pause()
            assert tabs.active == "pane-schema"

            await pilot.press("4")
            await pilot.pause()
            assert tabs.active == "pane-info"

            await pilot.press("5")
            await pilot.pause()
            assert tabs.active == "pane-profiles"

            await pilot.press("6")
            await pilot.pause()
            assert tabs.active == "pane-aliases"

    _run(_test())


def test_disabled_data_tab_cannot_be_activated(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40), notifications=True) as pilot:
            await pilot.pause()
            app.push_screen(ExploreScreen(entry_a, ""))
            await pilot.pause()

            await pilot.press("3")
            await pilot.pause()
            tabs = app.screen.query_one("#explore-tabs", TabbedContent)
            assert tabs.active != "pane-data"

    _run(_test())


def test_disabled_revisions_tab_without_alias(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40), notifications=True) as pilot:
            await pilot.pause()
            app.push_screen(ExploreScreen(entry_a, ""))
            await pilot.pause()

            await pilot.press("2")
            await pilot.pause()
            tabs = app.screen.query_one("#explore-tabs", TabbedContent)
            assert tabs.active == "pane-schema"

            await pilot.press("1")
            await pilot.pause()
            assert tabs.active != "pane-revisions"

    _run(_test())


def test_render_explore_populates_schema(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = ExploreScreen(entry_a, "test-alias")
            app.push_screen(screen)
            await pilot.pause()

            data = ExploreData(
                hash=entry_a.name,
                alias="test-alias",
                schema_items=(("id", "int64"), ("name", "string")),
                lineage_text="source → output",
                is_cached=False,
            )
            screen._render_explore(data)
            await pilot.pause()

            schema_table = screen.query_one("#schema-table", DataTable)
            assert schema_table.row_count >= 2

    _run(_test())


def test_render_explore_uncached_shows_uncached_status(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = ExploreScreen(entry_a, "")
            app.push_screen(screen)
            await pilot.pause()

            data = ExploreData(hash=entry_a.name, alias="", is_cached=False)
            screen._render_explore(data)
            await pilot.pause()

            status = screen.query_one("#data-status", Static)
            assert "uncached" in status.content

    _run(_test())


def test_render_explore_cached_enables_data_tab(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = ExploreScreen(entry_a, "test-alias")
            app.push_screen(screen)
            await pilot.pause()

            data = ExploreData(
                hash=entry_a.name,
                alias="test-alias",
                is_cached=True,
                cache_path="/tmp/cache/abc123",
            )
            screen._render_explore(data)
            await pilot.pause()

            pane = screen.query_one("#pane-data", TabPane)
            assert not pane.disabled

    _run(_test())


def test_render_explore_shows_metadata_when_present(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = ExploreScreen(entry_a, "")
            app.push_screen(screen)
            await pilot.pause()

            data = ExploreData(
                hash=entry_a.name,
                alias="",
                metadata=(("author", "alice"), ("version", "1.0")),
            )
            screen._render_explore(data)
            await pilot.pause()

            section = screen.query_one("#metadata-section")
            assert section.display is True

    _run(_test())


def test_render_explore_hides_metadata_when_empty(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = ExploreScreen(entry_a, "")
            app.push_screen(screen)
            await pilot.pause()

            data = ExploreData(hash=entry_a.name, alias="", metadata=())
            screen._render_explore(data)
            await pilot.pause()

            section = screen.query_one("#metadata-section")
            assert section.display is False

    _run(_test())


def test_render_refresh_populates_table(catalog, entry_a, entry_b):
    async def _test():
        app = _make_tui(catalog)
        rows = (CatalogRowData(entry=entry_a), CatalogRowData(entry=entry_b))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, CatalogScreen)

            screen._render_refresh(catalog.repo.working_dir, rows)
            await pilot.pause()

            catalog_table = screen.query_one("#catalog-table", DataTable)
            assert catalog_table.row_count == 2

    _run(_test())


def test_render_refresh_uses_entry_name_as_row_key(catalog, entry_a, entry_b):
    async def _test():
        app = _make_tui(catalog)
        rows = (CatalogRowData(entry=entry_a), CatalogRowData(entry=entry_b))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen

            screen._render_refresh(catalog.repo.working_dir, rows)
            await pilot.pause()

            table = screen.query_one("#catalog-table", DataTable)
            keys = [str(k.value) for k in table.rows.keys()]
            assert entry_a.name in keys
            assert entry_b.name in keys

    _run(_test())


def test_render_status_updates_status_bar(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            repo_path = catalog.repo.working_dir
            screen._render_status("12:00:00", repo_path)
            await pilot.pause()

            status = screen.query_one("#status-bar", Static)
            text = status.content
            assert "12:00:00" in text
            assert repo_path in text

    _run(_test())


def test_two_aliases_same_entry_produce_one_row(catalog, entry_a):
    async def _test():
        catalog.add_alias(entry_a.name, "latest")
        catalog.add_alias(entry_a.name, "v1")
        app = _make_tui(catalog)
        row = CatalogRowData(entry=entry_a, aliases=("latest", "v1"))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, CatalogScreen)

            screen._render_refresh(catalog.repo.working_dir, (row,))
            await pilot.pause()

            table = screen.query_one("#catalog-table", DataTable)
            assert table.row_count == 1
            keys = [str(k.value) for k in table.rows.keys()]
            assert entry_a.name in keys

    _run(_test())


def test_unaliased_entry_uses_name_as_key(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        row = CatalogRowData(entry=entry_a, aliases=())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen

            screen._render_refresh(catalog.repo.working_dir, (row,))
            await pilot.pause()

            table = screen.query_one("#catalog-table", DataTable)
            assert table.row_count == 1
            keys = [str(k.value) for k in table.rows.keys()]
            assert entry_a.name in keys

    _run(_test())


def test_cursor_move_updates_schema_preview(catalog, entry_a, entry_b):
    async def _test():
        app = _make_tui(catalog)
        rows = (
            CatalogRowData(entry=entry_a, aliases=("a",)),
            CatalogRowData(entry=entry_b, aliases=("b",)),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, CatalogScreen)

            screen._render_refresh(catalog.repo.working_dir, rows)
            screen._row_cache = {r.row_key: r for r in rows}
            await pilot.pause()

            # Move to second row (entry_b: value)
            await pilot.press("j")
            await pilot.pause()
            schema_table = screen.query_one("#schema-preview-table", DataTable)
            assert schema_table.row_count == 1
            assert schema_table.get_cell_at((0, 0)) == "value"

            # Move back to first row (entry_a: id, name, score)
            await pilot.press("k")
            await pilot.pause()
            assert schema_table.row_count == 3
            col_names = [schema_table.get_cell_at((i, 0)) for i in range(3)]
            assert "id" in col_names
            assert "name" in col_names
            assert "score" in col_names

    _run(_test())


def test_schema_preview_empty_before_selection(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            schema_table = app.screen.query_one("#schema-preview-table", DataTable)
            assert schema_table.row_count == 0

    _run(_test())


def test_aliases_tab_has_correct_columns(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ExploreScreen(entry_a, ""))
            await pilot.pause()

            aliases_table = app.screen.query_one("#aliases-table", DataTable)
            col_labels = tuple(
                col.label.plain for col in aliases_table.columns.values()
            )
            assert col_labels == ALIAS_COLUMNS

    _run(_test())


def test_aliases_rendered_from_row_data(catalog, entry_a):
    async def _test():
        catalog.add_alias(entry_a.name, "latest")
        catalog.add_alias(entry_a.name, "v1")
        catalog.add_alias(entry_a.name, "prod")
        row_data = CatalogRowData(entry=entry_a, aliases=("latest", "v1", "prod"))
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = ExploreScreen(entry_a, "v1", row_data=row_data)
            app.push_screen(screen)
            await pilot.pause()

            data = ExploreData(hash=entry_a.name, alias="v1")
            screen._render_explore(data)
            await pilot.pause()

            aliases_table = screen.query_one("#aliases-table", DataTable)
            assert aliases_table.row_count == 3
            rows = [
                aliases_table.get_cell_at((i, 0))
                for i in range(aliases_table.row_count)
            ]
            assert "latest" in rows
            assert "v1" in rows
            assert "prod" in rows

    _run(_test())


def test_keybinding_6_activates_aliases_tab(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ExploreScreen(entry_a, ""))
            await pilot.pause()

            await pilot.press("6")
            await pilot.pause()
            tabs = app.screen.query_one("#explore-tabs", TabbedContent)
            assert tabs.active == "pane-aliases"

    _run(_test())


# ---------------------------------------------------------------------------
# 8. Git Log: unit tests
# ---------------------------------------------------------------------------


def test_git_log_row_data_row_tuple():
    row = GitLogRowData(
        hash="abc123def456", date="2025-01-15 10:30", message="initial commit"
    )
    assert row.row == ("abc123def456", "2025-01-15 10:30", "initial commit")


def test_git_log_row_data_defaults():
    row = GitLogRowData()
    assert row.row == ("", "", "")


def test_git_log_row_data_is_frozen():
    row = GitLogRowData(hash="abc")
    with pytest.raises(AttributeError):
        row.hash = "new"


def test_builds_from_real_catalog_commits(catalog, entry_a, entry_b):
    rows = _build_git_log_rows(catalog.repo, max_count=50)
    # init + add catalog.yaml + add entry_a + add entry_b = at least 4 commits
    assert len(rows) >= 4
    for row in rows:
        assert len(row.hash) == 12
        assert row.date != ""
        assert row.message != ""


def test_max_count_limits_output(catalog, entry_a, entry_b):
    one_row = _build_git_log_rows(catalog.repo, max_count=1)
    assert len(one_row) == 1


def test_empty_catalog_has_initial_commit(catalog):
    rows = _build_git_log_rows(catalog.repo)
    assert len(rows) >= 1


# ---------------------------------------------------------------------------
# 9. Git Log: pilot tests
# ---------------------------------------------------------------------------


def test_git_log_panel_hidden_by_default(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.screen.query_one("#git-log-panel")
            assert panel.display is False

    _run(_test())


def test_g_toggles_git_log_visibility(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.screen.query_one("#git-log-panel")
            assert panel.display is False

            await pilot.press("g")
            await pilot.pause()
            assert panel.display is True

            await pilot.press("g")
            await pilot.pause()
            assert panel.display is False

    _run(_test())


def test_git_log_table_has_correct_columns(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            git_table = app.screen.query_one("#git-log-table", DataTable)
            col_labels = tuple(col.label.plain for col in git_table.columns.values())
            assert col_labels == GIT_LOG_COLUMNS

    _run(_test())


def test_git_log_panel_border_title(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            panel = app.screen.query_one("#git-log-panel")
            assert panel.border_title == "Git Log"

    _run(_test())


def test_render_git_log_populates_table(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, CatalogScreen)

            rows = (
                GitLogRowData(hash="aabb", date="2025-01-01 10:00", message="first"),
                GitLogRowData(hash="ccdd", date="2025-01-02 11:00", message="second"),
            )
            screen._render_git_log(rows)
            await pilot.pause()

            git_table = screen.query_one("#git-log-table", DataTable)
            assert git_table.row_count == 2
            assert git_table.get_cell_at((0, 0)) == "aabb"
            assert git_table.get_cell_at((0, 2)) == "first"
            assert git_table.get_cell_at((1, 0)) == "ccdd"

    _run(_test())


def test_toggle_triggers_load_from_real_repo(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()

            await pilot.press("g")
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()

            git_table = app.screen.query_one("#git-log-table", DataTable)
            # at minimum: initial commit + add catalog.yaml + add entry_a
            assert git_table.row_count >= 3

    _run(_test())


# ---------------------------------------------------------------------------
# 10. _entry_info: reads from real CatalogEntry
# ---------------------------------------------------------------------------


def test_entry_info(entry_b):
    """_entry_info reads column count from expr_metadata; cached is None for plain memtables."""
    column_count, cached, root_tag, expr = _entry_info(entry_b)
    assert column_count == 1  # single column: value
    assert cached is None  # no ParquetSnapshotCache nodes in a plain memtable
    assert root_tag == ""
    assert expr is None


def test_entry_info_three_columns(entry_a):
    """_entry_info reports the correct column count for a multi-column expression."""
    column_count, cached, root_tag, expr = _entry_info(entry_a)
    assert column_count == 3  # id, name, score
    assert cached is None


def test_entry_info_scalar_expression_wraps_as_table(catalog):
    """Scalar expressions are wrapped with as_table() at catalog-save time so
    column_count is the number of columns of the resulting table (always 1)."""
    t = xo.memtable({"a": [1, 2, 3]})
    entry = catalog.add(t.a.sum())
    column_count, cached, root_tag, expr = _entry_info(entry)
    assert column_count == 1
    assert cached is None


def test_cached_false_before_execution(catalog, tmp_path, parquet_dir):
    con = xo.duckdb.connect()
    t = deferred_read_parquet(
        parquet_dir / "astronauts.parquet", con, table_name="astronauts"
    )
    cache = ParquetSnapshotCache.from_kwargs(relative_path=tmp_path / "cache")
    expr = t.cache(cache=cache)
    entry = catalog.add(expr)

    parquet_paths = entry.parquet_cache_paths
    assert parquet_paths, "entry must have parquet_cache_paths"
    assert not any(Path(p).exists() for p in parquet_paths)
    assert CatalogRowData(entry=entry).cached is False
    _, cached, _, _ = _entry_info(entry)
    assert cached is False


def test_cached_true_after_execution(catalog, tmp_path, parquet_dir):
    con = xo.duckdb.connect()
    t = deferred_read_parquet(
        parquet_dir / "astronauts.parquet", con, table_name="astronauts"
    )
    cache = ParquetSnapshotCache.from_kwargs(relative_path=tmp_path / "cache")
    expr = t.cache(cache=cache)
    entry = catalog.add(expr)
    entry.expr.execute()

    parquet_paths = entry.parquet_cache_paths
    assert all(Path(p).exists() for p in parquet_paths)
    assert CatalogRowData(entry=entry).cached is True
    _, cached, _, _ = _entry_info(entry)
    assert cached is True


def test_cached_display_reflects_execution_state(catalog, tmp_path, parquet_dir):
    con = xo.duckdb.connect()
    t = deferred_read_parquet(
        parquet_dir / "astronauts.parquet", con, table_name="astronauts"
    )
    cache = ParquetSnapshotCache.from_kwargs(relative_path=tmp_path / "cache")
    expr = t.cache(cache=cache)
    entry = catalog.add(expr)

    assert CatalogRowData(entry=entry).cached_display == "○"
    entry.expr.execute()
    assert CatalogRowData(entry=entry).cached_display == "●"


def test_memtable_cached_lifecycle(catalog, tmp_path):
    cache = ParquetSnapshotCache.from_kwargs(relative_path=tmp_path / "cache")
    expr = xo.memtable({"x": [1, 2, 3]}).cache(cache=cache)
    entry = catalog.add(expr)

    parquet_paths = entry.parquet_cache_paths
    assert parquet_paths, "entry must have parquet_cache_paths"
    assert CatalogRowData(entry=entry).cached is False

    entry.expr.execute()
    assert CatalogRowData(entry=entry).cached is True
