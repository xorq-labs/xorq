"""Tests for the catalog TUI using Textual's Pilot test driver.

Strategy:
- Format helpers and frozen data classes: pure unit tests, no catalog needed.
- Screen composition, navigation, rendering: use a real Catalog backed by a
  temporary git repo so that CatalogEntry objects carry genuine expr_metadata,
  backends, and column info loaded from the zip archive.
- Git log: use the real repo that backs the catalog fixture.

IMPORTANT — populating the catalog tree in pilot tests:
    Never wait for the async _do_refresh worker to populate rows.  It runs in
    a background thread on a timer and is inherently racy under test.  Instead,
    build CatalogRowData objects and call _render_refresh() directly — see the
    _populate_tree() helper below.
"""

import asyncio
from pathlib import Path

import pytest
from textual.widgets import DataTable, Input, Static, Tree

import xorq.api as xo
from xorq.caching import ParquetSnapshotCache
from xorq.catalog.tests.testing import (
    Assert,
    Press,
    run_script,
    settle,
    wait_until,
)
from xorq.catalog.tui import (
    GIT_LOG_COLUMNS,
    CatalogRowData,
    CatalogScreen,
    CatalogTUI,
    DataViewScreen,
    ExprStack,
    ExprStep,
    GitLogRowData,
    RevisionRowData,
    _build_git_log_rows,
    _entry_info,
    _format_cached,
    build_code,
    get_cache_key_path,
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


async def _populate_tree(pilot, catalog, *entries):
    """Deterministically populate the catalog tree with the given entries.

    Use this instead of waiting for the async _do_refresh worker, which is
    racy under test.  Returns the CatalogScreen and the list of CatalogRowData.
    """
    await settle(pilot)
    screen = pilot.app.screen
    rows = tuple(CatalogRowData(entry=e) for e in entries)
    screen._row_cache = {r.row_key: r for r in rows}
    screen._render_refresh(catalog.repo.working_dir, rows)
    await settle(pilot)
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


def test_revision_row_columns_display_none():
    assert RevisionRowData(column_count=None).columns_display == "?"


def test_revision_row_columns_display_int():
    assert RevisionRowData(column_count=5).columns_display == "5 cols"


def test_revision_row_columns_display_zero():
    assert RevisionRowData(column_count=0).columns_display == "0 cols"


# ---------------------------------------------------------------------------
# 2. Unit tests: frozen data classes backed by real catalog entries
# ---------------------------------------------------------------------------


def test_cached_is_none_for_plain_memtable(entry_a):
    row = CatalogRowData(entry=entry_a)
    assert not row.cached


def test_schema_out_single_column(entry_b):
    row = CatalogRowData(entry=entry_b)
    assert len(row.schema_out) == 1


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


def test_tree_label_with_alias(entry_a, alias_for_a):
    row = CatalogRowData(entry=entry_a, aliases=(alias_for_a,))
    assert alias_for_a in row.tree_label
    assert entry_a.name[:12] in row.tree_label


def test_tree_label_without_alias(entry_a):
    row = CatalogRowData(entry=entry_a)
    assert row.tree_label == entry_a.name[:12]


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


# ---------------------------------------------------------------------------
# 3. Pilot tests: app setup
# ---------------------------------------------------------------------------


def test_app_starts_and_pushes_catalog_screen(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            assert isinstance(app.screen, CatalogScreen)

    _run(_test())


def test_app_has_custom_theme(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            assert app.theme == "xorq-dark"

    _run(_test())


def test_catalog_tree_exists(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            tree = app.screen.query_one("#catalog-tree", Tree)
            assert tree is not None
            assert tree.show_root is False

    _run(_test())


def test_status_bar_exists(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            status = app.screen.query_one("#status-bar", Static)
            assert status is not None

    _run(_test())


def test_panel_border_titles(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            catalog_panel = app.screen.query_one("#catalog-panel")
            assert "Expressions" in str(catalog_panel.border_title)

            schema_panel = app.screen.query_one("#schema-panel")
            assert schema_panel.border_title == "Schema"

    _run(_test())


def test_quit_exits_app(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            await pilot.press("q")

    _run(_test())


def test_j_k_moves_cursor(catalog, entry_a, entry_b):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a, entry_b)
            tree = screen.query_one("#catalog-tree", Tree)

            # Tree structure: source (2) > entry_a, entry_b
            # Initial cursor on "source" branch (data=kind string)
            await run_script(
                pilot,
                Assert(lambda p: tree.cursor_node is not None),
                Assert(lambda p: tree.cursor_node.data == "source"),  # on branch
                Press(("j",)),
                Assert(lambda p: tree.cursor_node.data == entry_a.name),
                Press(("j",)),
                Assert(lambda p: tree.cursor_node.data == entry_b.name),
                Press(("k",)),
                Assert(lambda p: tree.cursor_node.data == entry_a.name),
            )

    _run(_test())


def test_data_preview_hidden_by_default(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            panel = app.screen.query_one("#data-preview-panel")
            assert panel.display is False

    _run(_test())


def test_info_panel_exists(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            info = app.screen.query_one("#info-panel")
            assert info.border_title == "Info"

    _run(_test())


def test_render_refresh_populates_tree(catalog, entry_a, entry_b):
    async def _test():
        app = _make_tui(catalog)
        rows = (CatalogRowData(entry=entry_a), CatalogRowData(entry=entry_b))
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen
            assert isinstance(screen, CatalogScreen)

            screen._render_refresh(catalog.repo.working_dir, rows)
            await settle(pilot)

            tree = screen.query_one("#catalog-tree", Tree)
            # Both entries are "source" kind → one branch with 2 leaves
            assert len(tree.root.children) == 1
            branch = tree.root.children[0]
            assert "source" in str(branch.label)
            assert len(branch.children) == 2

    _run(_test())


def test_render_refresh_uses_entry_name_as_node_data(catalog, entry_a, entry_b):
    async def _test():
        app = _make_tui(catalog)
        rows = (CatalogRowData(entry=entry_a), CatalogRowData(entry=entry_b))
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen

            screen._render_refresh(catalog.repo.working_dir, rows)
            await settle(pilot)

            hashes = screen._tree_entry_hashes()
            assert entry_a.name in hashes
            assert entry_b.name in hashes

    _run(_test())


def test_render_status_updates_status_bar(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen
            repo_path = catalog.repo.working_dir
            screen._render_status("12:00:00", repo_path)
            await settle(pilot)

            status = screen.query_one("#status-bar", Static)
            text = status.content
            assert "12:00:00" in text
            assert repo_path in text

    _run(_test())


def test_two_aliases_same_entry_produce_one_leaf(catalog, entry_a):
    async def _test():
        catalog.add_alias(entry_a.name, "latest")
        catalog.add_alias(entry_a.name, "v1")
        app = _make_tui(catalog)
        row = CatalogRowData(entry=entry_a, aliases=("latest", "v1"))
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen
            assert isinstance(screen, CatalogScreen)

            screen._render_refresh(catalog.repo.working_dir, (row,))
            await settle(pilot)

            hashes = screen._tree_entry_hashes()
            assert entry_a.name in hashes
            assert len(hashes) == 1

    _run(_test())


def test_unaliased_entry_uses_name_in_tree(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        row = CatalogRowData(entry=entry_a, aliases=())
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen

            screen._render_refresh(catalog.repo.working_dir, (row,))
            await settle(pilot)

            hashes = screen._tree_entry_hashes()
            assert entry_a.name in hashes
            assert len(hashes) == 1

    _run(_test())


def test_cursor_move_updates_schema_preview(catalog, entry_a, entry_b):
    async def _test():
        app = _make_tui(catalog)
        rows = (
            CatalogRowData(entry=entry_a, aliases=("a",)),
            CatalogRowData(entry=entry_b, aliases=("b",)),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen
            assert isinstance(screen, CatalogScreen)

            screen._row_cache = {r.row_key: r for r in rows}
            screen._render_refresh(catalog.repo.working_dir, rows)
            await settle(pilot)

            schema_table = screen.query_one("#schema-preview-table", DataTable)

            await run_script(
                pilot,
                # Move past branch to first leaf (entry_a: id, name, score)
                Press(("j",)),
                Assert(lambda p: schema_table.row_count == 3),
                Assert(
                    lambda p: "id"
                    in [schema_table.get_cell_at((i, 0)) for i in range(3)]
                ),
                Assert(
                    lambda p: "name"
                    in [schema_table.get_cell_at((i, 0)) for i in range(3)]
                ),
                Assert(
                    lambda p: "score"
                    in [schema_table.get_cell_at((i, 0)) for i in range(3)]
                ),
                # Move to second leaf (entry_b: value)
                Press(("j",)),
                Assert(lambda p: schema_table.row_count == 1),
                Assert(lambda p: schema_table.get_cell_at((0, 0)) == "value"),
                # Move back to first leaf (entry_a: id, name, score)
                Press(("k",)),
                Assert(lambda p: schema_table.row_count == 3),
            )

    _run(_test())


def test_schema_preview_empty_before_selection(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            schema_table = app.screen.query_one("#schema-preview-table", DataTable)
            assert schema_table.row_count == 0

    _run(_test())


def test_view_switching_1_2(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen

            sql_panel = screen.query_one("#sql-panel")
            data_panel = screen.query_one("#data-preview-panel")

            await run_script(
                pilot,
                # Default: SQL visible, data hidden
                Assert(lambda p: sql_panel.display is not False),
                Assert(lambda p: data_panel.display is False),
                # Switch to data
                Press(("2",)),
                Assert(lambda p: sql_panel.display is False),
                Assert(lambda p: data_panel.display is not False),
                # Switch back to SQL
                Press(("1",)),
                Assert(lambda p: sql_panel.display is not False),
                Assert(lambda p: data_panel.display is False),
            )

    _run(_test())


def test_v_toggles_revisions(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            panel = app.screen.query_one("#revisions-panel")

            await run_script(
                pilot,
                Assert(lambda p: panel.display is False),
                Press(("v",)),
                Assert(lambda p: panel.display is not False),
                Press(("v",)),
                Assert(lambda p: panel.display is False),
            )

    _run(_test())


def test_tree_entry_hashes_helper(catalog, entry_a, entry_b):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a, entry_b)
            hashes = screen._tree_entry_hashes()
            assert entry_a.name in hashes
            assert entry_b.name in hashes
            assert len(hashes) == 2

    _run(_test())


# ---------------------------------------------------------------------------
# 5. Git Log: unit tests
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
# 5. Git Log: pilot tests
# ---------------------------------------------------------------------------


def test_git_log_panel_hidden_by_default(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            panel = app.screen.query_one("#git-log-panel")
            assert panel.display is False

    _run(_test())


def test_g_toggles_git_log_visibility(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            panel = app.screen.query_one("#git-log-panel")

            await run_script(
                pilot,
                Assert(lambda p: panel.display is False),
                Press(("g",)),
                Assert(lambda p: panel.display is not False),
                Press(("g",)),
                Assert(lambda p: panel.display is False),
            )

    _run(_test())


def test_git_log_table_has_correct_columns(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            git_table = app.screen.query_one("#git-log-table", DataTable)
            col_labels = tuple(col.label.plain for col in git_table.columns.values())
            assert col_labels == GIT_LOG_COLUMNS

    _run(_test())


def test_git_log_panel_border_title(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            panel = app.screen.query_one("#git-log-panel")
            assert panel.border_title == "Git Log"

    _run(_test())


def test_render_git_log_populates_table(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen
            assert isinstance(screen, CatalogScreen)

            rows = (
                GitLogRowData(hash="aabb", date="2025-01-01 10:00", message="first"),
                GitLogRowData(hash="ccdd", date="2025-01-02 11:00", message="second"),
            )
            screen._render_git_log(rows)
            await settle(pilot)

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
            await settle(pilot)
            await pilot.press("g")
            git_table = app.screen.query_one("#git-log-table", DataTable)
            await wait_until(pilot, lambda: git_table.row_count >= 3)

    _run(_test())


# ---------------------------------------------------------------------------
# 6. _entry_info: reads from real CatalogEntry
# ---------------------------------------------------------------------------


def test_entry_info(entry_b):
    """_entry_info reads column count from expr_metadata; cached is None for plain memtables."""
    column_count, cached = _entry_info(entry_b)
    assert column_count == 1  # single column: value
    assert not cached


def test_entry_info_three_columns(entry_a):
    """_entry_info reports the correct column count for a multi-column expression."""
    column_count, cached = _entry_info(entry_a)
    assert column_count == 3  # id, name, score
    assert not cached


def test_entry_info_scalar_expression_wraps_as_table(catalog):
    """Scalar expressions are wrapped with as_table() at catalog-save time so
    column_count is the number of columns of the resulting table (always 1)."""
    t = xo.memtable({"a": [1, 2, 3]})
    entry = catalog.add(t.a.sum())
    column_count, cached = _entry_info(entry)
    assert column_count == 1
    assert not cached


def test_cached_false_before_execution(catalog, tmp_path, parquet_dir):
    con = xo.duckdb.connect()
    t = deferred_read_parquet(
        parquet_dir / "astronauts.parquet", con, table_name="astronauts"
    )
    cache = ParquetSnapshotCache.from_kwargs(relative_path=tmp_path / "cache")
    expr = t.cache(cache=cache)
    entry = catalog.add(expr)

    path = get_cache_key_path(entry.projected_cache_key)
    assert path is not None, "entry must have a cache key path"
    assert not Path(path).exists()
    assert CatalogRowData(entry=entry).cached is False
    _, cached = _entry_info(entry)
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

    path = get_cache_key_path(entry.projected_cache_key)
    assert path is not None and Path(path).exists()
    assert CatalogRowData(entry=entry).cached is True
    _, cached = _entry_info(entry)
    assert cached is True


# ---------------------------------------------------------------------------
# 11. DataViewScreen: pilot tests
# ---------------------------------------------------------------------------


@pytest.fixture
def _mock_catalog_run(monkeypatch):
    """Bypass xorq catalog subprocess — execute the expression in-process."""
    monkeypatch.setattr(
        DataViewScreen,
        "_run_catalog_subprocess",
        lambda self: self._entry.expr.limit(50_000).execute(),
    )



def test_data_view_screen_construction(entry_a):
    row_data = CatalogRowData(entry=entry_a)
    screen = DataViewScreen(entry=entry_a, row_data=row_data)
    assert screen._entry is entry_a
    assert screen._row_data is row_data
    assert screen._df is None


def test_e_pushes_data_view_screen(catalog, entry_a, _mock_catalog_run):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            tree = screen.query_one("#catalog-tree", Tree)

            await run_script(
                pilot,
                Press(("j",)),  # move to first leaf
                Assert(lambda p: tree.cursor_node.data == entry_a.name),
                Press(("e",)),
            )
            await settle(pilot)
            assert isinstance(app.screen, DataViewScreen)

    _run(_test())


def test_e_on_branch_does_nothing(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            tree = screen.query_one("#catalog-tree", Tree)

            # Cursor starts on branch node ("source")
            assert tree.cursor_node.data == "source"
            await pilot.press("e")
            await settle(pilot)
            assert isinstance(app.screen, CatalogScreen)

    _run(_test())


def test_data_view_escape_returns(catalog, entry_a, _mock_catalog_run):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)

            await run_script(
                pilot,
                Press(("j",)),
                Press(("e",)),
            )
            await settle(pilot)
            assert isinstance(app.screen, DataViewScreen)
            await pilot.press("escape")
            await settle(pilot)
            assert isinstance(app.screen, CatalogScreen)

    _run(_test())


def test_data_view_loads_data(catalog, entry_a, _mock_catalog_run):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)

            await run_script(
                pilot,
                Press(("j",)),
                Press(("e",)),
            )
            await settle(pilot)
            data_screen = app.screen
            assert isinstance(data_screen, DataViewScreen)

            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)
            # entry_a has 2 rows: alice, bob
            assert data_table.row_count == 2

    _run(_test())


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

    path = get_cache_key_path(entry.projected_cache_key)
    assert path is not None, "entry must have a cache key path"
    assert CatalogRowData(entry=entry).cached is False

    entry.expr.execute()
    assert CatalogRowData(entry=entry).cached is True


# ---------------------------------------------------------------------------
# 12. ExprStack: pure unit tests
# ---------------------------------------------------------------------------


def test_expr_step_is_frozen():
    step = ExprStep(
        verb="filter", user_input="source.x > 1", code="source.filter(source.x > 1)"
    )
    with pytest.raises(AttributeError):
        step.verb = "mutate"


def test_expr_stack_initial_state():
    base = xo.memtable({"x": [1, 2, 3]})
    stack = ExprStack(base_expr=base)
    assert stack.cursor == 0
    assert stack.steps == ()
    assert stack.current_code == ""
    assert not stack.can_undo
    assert not stack.can_redo


def test_expr_stack_push():
    base = xo.memtable({"x": [1, 2, 3]})
    stack = ExprStack(base_expr=base)
    step = ExprStep(
        verb="filter", user_input="source.x > 1", code="source.filter(source.x > 1)"
    )
    stack2 = stack.push(step)
    assert stack2.cursor == 1
    assert len(stack2.steps) == 1
    assert stack2.steps[0] is step
    # original unchanged (immutable)
    assert stack.cursor == 0
    assert len(stack.steps) == 0


def test_expr_stack_undo_redo():
    base = xo.memtable({"x": [1, 2, 3]})
    step = ExprStep(
        verb="filter", user_input="source.x > 1", code="source.filter(source.x > 1)"
    )
    stack = ExprStack(base_expr=base).push(step)
    assert stack.can_undo
    assert not stack.can_redo

    undone = stack.undo()
    assert undone.cursor == 0
    assert not undone.can_undo
    assert undone.can_redo

    redone = undone.redo()
    assert redone.cursor == 1
    assert redone.can_undo
    assert not redone.can_redo


def test_expr_stack_undo_at_zero():
    base = xo.memtable({"x": [1, 2, 3]})
    stack = ExprStack(base_expr=base)
    assert stack.undo().cursor == 0


def test_expr_stack_redo_at_end():
    base = xo.memtable({"x": [1, 2, 3]})
    step = ExprStep(
        verb="filter", user_input="source.x > 1", code="source.filter(source.x > 1)"
    )
    stack = ExprStack(base_expr=base).push(step)
    assert stack.redo().cursor == 1


def test_expr_stack_fork_discards_after_cursor():
    base = xo.memtable({"x": [1, 2, 3]})
    step1 = ExprStep(
        verb="filter", user_input="source.x > 1", code="source.filter(source.x > 1)"
    )
    step2 = ExprStep(
        verb="mutate", user_input="y=source.x * 2", code="source.mutate(y=source.x * 2)"
    )
    stack = ExprStack(base_expr=base).push(step1).push(step2)
    assert stack.cursor == 2

    # Undo to step 1, then push a new step — step2 should be discarded
    undone = stack.undo()
    assert undone.cursor == 1
    step3 = ExprStep(verb="select", user_input='"x"', code='source.select("x")')
    forked = undone.push(step3)
    assert forked.cursor == 2
    assert len(forked.steps) == 2
    assert forked.steps[1] is step3  # step2 was replaced


def test_expr_stack_current_expr_evaluates():
    base = xo.memtable({"x": [1, 2, 3]})
    step = ExprStep(
        verb="filter", user_input="source.x > 1", code="source.filter(source.x > 1)"
    )
    stack = ExprStack(base_expr=base).push(step)
    result = stack.current_expr()
    df = result.execute()
    assert len(df) == 2
    assert list(df["x"]) == [2, 3]


def test_expr_stack_current_code():
    base = xo.memtable({"x": [1, 2, 3]})
    step1 = ExprStep(
        verb="filter", user_input="source.x > 1", code="source.filter(source.x > 1)"
    )
    step2 = ExprStep(
        verb="mutate", user_input="y=source.x * 2", code="source.mutate(y=source.x * 2)"
    )
    stack = ExprStack(base_expr=base).push(step1).push(step2)
    code = stack.current_code
    assert "source.filter(source.x > 1)" in code
    assert "source.mutate(y=source.x * 2)" in code


def test_build_code_filter():
    assert build_code("filter", "source.x > 1") == "source.filter(source.x > 1)"


def test_build_code_mutate():
    assert build_code("mutate", "y=source.x * 2") == "source.mutate(y=source.x * 2)"


def test_build_code_select():
    assert build_code("select", '"x", "y"') == 'source.select("x", "y")'


def test_build_code_freeform():
    assert build_code("freeform", "source.distinct()") == "source.distinct()"


def test_build_code_agg():
    code = build_code("agg", "avg=source.x.mean()", group='"category"')
    assert code == 'source.group_by("category").agg(avg=source.x.mean())'


def test_build_code_agg_empty_group():
    code = build_code("agg", "total=source.x.sum()", group="")
    assert code == "source.group_by().agg(total=source.x.sum())"


# ---------------------------------------------------------------------------
# 13. DataViewScreen compose: pilot tests
# ---------------------------------------------------------------------------


def test_data_view_has_command_input_hidden(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)

            await run_script(
                pilot,
                Press(("j",)),
                Press(("e",)),
            )
            await settle(pilot)
            assert isinstance(app.screen, DataViewScreen)

            cmd = app.screen.query_one("#command-input", Input)
            assert cmd.display is False

    _run(_test())


def test_data_view_f_opens_filter_input(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)

            await run_script(
                pilot,
                Press(("j",)),
                Press(("e",)),
            )
            await settle(pilot)
            data_screen = app.screen
            assert isinstance(data_screen, DataViewScreen)

            # Wait for data to load first
            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            await pilot.press("f")
            await settle(pilot)

            cmd = data_screen.query_one("#command-input", Input)
            assert cmd.display is not False
            assert "filter" in str(cmd.border_title)

    _run(_test())


def test_data_view_escape_closes_command_input(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)

            await run_script(
                pilot,
                Press(("j",)),
                Press(("e",)),
            )
            await settle(pilot)
            data_screen = app.screen
            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            # Open command input
            await pilot.press("f")
            await settle(pilot)

            cmd = data_screen.query_one("#command-input", Input)
            assert cmd.display is not False

            # Escape closes it
            await pilot.press("escape")
            await settle(pilot)
            assert cmd.display is False
            # Still on DataViewScreen
            assert isinstance(app.screen, DataViewScreen)

    _run(_test())


def test_data_view_stack_browser_toggle(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)

            await run_script(
                pilot,
                Press(("j",)),
                Press(("e",)),
            )
            await settle(pilot)
            data_screen = app.screen
            assert isinstance(data_screen, DataViewScreen)

            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            panel = data_screen.query_one("#stack-browser-panel")
            assert panel.display is False

            await run_script(
                pilot,
                Press(("e",)),
                Assert(lambda p: panel.display is not False),
                Press(("e",)),
                Assert(lambda p: panel.display is False),
            )

    _run(_test())


def test_data_view_undo_redo_no_crash_when_empty(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)

            await run_script(
                pilot,
                Press(("j",)),
                Press(("e",)),
            )
            await settle(pilot)
            data_screen = app.screen
            assert isinstance(data_screen, DataViewScreen)

            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            # Undo/redo with empty stack should not crash
            await pilot.press("u")
            await settle(pilot)
            await pilot.press("ctrl+r")
            await settle(pilot)
            assert isinstance(app.screen, DataViewScreen)

    _run(_test())
