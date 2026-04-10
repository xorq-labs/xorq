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
from textual.widgets import DataTable, Static, Tree

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
    GitLogRowData,
    RevisionRowData,
    _build_git_log_rows,
    _entry_info,
    _format_cached,
    get_cache_keys_paths,
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
    assert row.cached is None


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
                    lambda p: (
                        "id" in [schema_table.get_cell_at((i, 0)) for i in range(3)]
                    )
                ),
                Assert(
                    lambda p: (
                        "name" in [schema_table.get_cell_at((i, 0)) for i in range(3)]
                    )
                ),
                Assert(
                    lambda p: (
                        "score" in [schema_table.get_cell_at((i, 0)) for i in range(3)]
                    )
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


def test_d_toggles_data_preview(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            tree = screen.query_one("#catalog-tree", Tree)
            data_panel = screen.query_one("#data-preview-panel")

            await run_script(
                pilot,
                # Move cursor to a leaf node
                Press(("j",)),
                Assert(lambda p: tree.cursor_node.data == entry_a.name),
                # Default: data hidden
                Assert(lambda p: data_panel.display is False),
                # Toggle on
                Press(("d",)),
                Assert(lambda p: data_panel.display is not False),
                # Toggle off
                Press(("d",)),
                Assert(lambda p: data_panel.display is False),
            )

    _run(_test())


def test_profiles_hidden_by_default(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            panel = app.screen.query_one("#profiles-panel")
            assert panel.display is False

    _run(_test())


def test_p_toggles_profiles_visibility(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            tree = screen.query_one("#catalog-tree", Tree)
            panel = screen.query_one("#profiles-panel")

            await run_script(
                pilot,
                # Move cursor to a leaf node
                Press(("j",)),
                Assert(lambda p: tree.cursor_node.data == entry_a.name),
                Assert(lambda p: panel.display is False),
                Press(("p",)),
                Assert(lambda p: panel.display is not False),
                Press(("p",)),
                Assert(lambda p: panel.display is False),
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
# 4. Git Log: unit tests
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
    assert cached is None  # no ParquetSnapshotCache nodes in a plain memtable


def test_entry_info_three_columns(entry_a):
    """_entry_info reports the correct column count for a multi-column expression."""
    column_count, cached = _entry_info(entry_a)
    assert column_count == 3  # id, name, score
    assert cached is None


def test_entry_info_scalar_expression_wraps_as_table(catalog):
    """Scalar expressions are wrapped with as_table() at catalog-save time so
    column_count is the number of columns of the resulting table (always 1)."""
    t = xo.memtable({"a": [1, 2, 3]})
    entry = catalog.add(t.a.sum())
    column_count, cached = _entry_info(entry)
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

    cache_keys_paths = get_cache_keys_paths(entry.parquet_snapshot_cache_keys)
    assert cache_keys_paths, "entry must have cache_keys_paths"
    assert not any(Path(p).exists() for p in cache_keys_paths)
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

    cache_keys_paths = get_cache_keys_paths(entry.parquet_snapshot_cache_keys)
    assert all(Path(p).exists() for p in cache_keys_paths)
    assert CatalogRowData(entry=entry).cached is True
    _, cached = _entry_info(entry)
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

    cache_keys_paths = get_cache_keys_paths(entry.parquet_snapshot_cache_keys)
    assert cache_keys_paths, "entry must have cache_keys_paths"
    assert CatalogRowData(entry=entry).cached is False

    entry.expr.execute()
    assert CatalogRowData(entry=entry).cached is True
