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
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import DataTable, Input, Select, Static, Tree

import xorq.api as xo
import xorq.config
from xorq.caching import ParquetSnapshotCache
from xorq.catalog.bind import _eval_code
from xorq.catalog.catalog import Catalog, CatalogAlias, CatalogEntry
from xorq.catalog.tests.testing import (
    Assert,
    Press,
    WaitUntil,
    run_script,
    settle,
    wait_until,
)
from xorq.catalog.tui import (
    GIT_LOG_COLUMNS,
    KIND_ORDER,
    KIND_STYLES,
    AddAliasScreen,
    AddEntryScreen,
    CatalogRowData,
    CatalogScreen,
    CatalogTUI,
    DataViewScreen,
    DeleteEntryScreen,
    ExprStack,
    ExprStep,
    GitLogRowData,
    RemoveAliasScreen,
    RevisionRowData,
    _build_git_log_rows,
    _entry_info,
    _format_cached,
    _get_catalog_aliases,
    _list_revisions_cached,
    _pygments_to_text,
    _pygments_tokens,
    _render_sql_dag,
    _render_sql_text,
    _styled_branch_label,
    get_cache_key_path,
)
from xorq.catalog.zip_utils import extract_build_zip_to
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.env_utils import (
    EnvConfigable,
    env_templates_dir,
)
from xorq.config import TUI, options
from xorq.ibis_yaml.enums import ExprKind


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


@pytest.mark.parametrize("kind", list(ExprKind))
def test_kind_order_and_styles_cover_every_expr_kind(kind):
    """KIND_ORDER and KIND_STYLES must include every ExprKind value.

    Drift here causes branch ordering to drop the kind and
    _styled_branch_label to KeyError at render time.
    """
    assert kind in KIND_ORDER
    assert kind in KIND_STYLES


@pytest.mark.parametrize("kind", list(ExprKind))
def test_styled_branch_label_renders_every_kind(kind):
    label = _styled_branch_label(kind, 1)
    assert kind in label.plain
    assert "(1)" in label.plain


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


def test_add_entry_from_build_directory_with_alias(catalog, entry_a, entry_b, tmp_path):
    async def _test():
        zip_path = catalog.get_zip(entry_b.name, dir_path=tmp_path)
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        build_dir = extract_build_zip_to(zip_path, extract_dir)
        catalog.remove(entry_b.name)
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await _populate_tree(pilot, catalog, entry_a)
            await pilot.press("a")
            await settle(pilot)

            assert isinstance(app.screen, AddEntryScreen)
            app.screen.query_one("#add-entry-path", Input).value = str(build_dir)
            app.screen.query_one("#add-entry-alias", Input).value = "restored"
            await pilot.press("ctrl+r")
            await wait_until(pilot, lambda: entry_b.name in catalog.list())

            assert isinstance(app.screen, CatalogScreen)
            assert "restored" in catalog.list_aliases()
            assert entry_b.name in app.screen._tree_entry_hashes()
            assert app.screen._row_cache[entry_b.name].aliases == ("restored",)

    _run(_test())


def test_add_entry_rejects_zip_path(catalog, tmp_path):
    async def _test():
        zip_path = tmp_path / "build.zip"
        zip_path.touch()
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            await pilot.press("a")
            await settle(pilot)

            assert isinstance(app.screen, AddEntryScreen)
            app.screen.query_one("#add-entry-path", Input).value = str(zip_path)
            await pilot.press("ctrl+r")
            await settle(pilot)

            assert isinstance(app.screen, AddEntryScreen)
            assert catalog.list() == []

    _run(_test())


def test_add_alias_to_selected_entry(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await _populate_tree(pilot, catalog, entry_a)
            await pilot.press("j", "A")
            await settle(pilot)

            assert isinstance(app.screen, AddAliasScreen)
            app.screen.query_one("#add-alias-name", Input).value = "new-alias"
            await pilot.press("ctrl+r")
            await wait_until(pilot, lambda: "new-alias" in catalog.list_aliases())

            assert isinstance(app.screen, CatalogScreen)
            assert entry_a.name in catalog.list()
            assert app.screen._row_cache[entry_a.name].aliases == ("new-alias",)
            assert "new-alias" in _leaf_label_for(app.screen, entry_a.name)

    _run(_test())


def test_add_alias_binding_only_visible_on_focused_entry(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)

            # The cursor starts on the kind branch, so Add Alias is hidden.
            assert "A" not in app.active_bindings

            await pilot.press("j")
            await settle(pilot)
            assert screen._selected_row_data() is not None
            assert "A" in app.active_bindings

            # Moving focus away from the entries panel hides the action again.
            await pilot.press("tab")
            await settle(pilot)
            assert "A" not in app.active_bindings

    _run(_test())


def test_delete_entry_can_be_cancelled(catalog, entry_a):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await _populate_tree(pilot, catalog, entry_a)
            await pilot.press("j", "d")
            await settle(pilot)

            assert isinstance(app.screen, DeleteEntryScreen)
            await pilot.press("escape")
            await settle(pilot)

            assert isinstance(app.screen, CatalogScreen)
            assert entry_a.name in catalog.list()

    _run(_test())


def test_delete_entry_removes_entry_and_aliases(catalog, entry_a):
    async def _test():
        catalog.add_alias(entry_a.name, "to-delete")
        app = _make_tui(catalog)
        row = CatalogRowData(entry=entry_a, aliases=("to-delete",))
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen
            screen._row_cache = {row.row_key: row}
            screen._render_refresh(catalog.repo.working_dir, (row,))
            await settle(pilot)

            await pilot.press("j", "d")
            await settle(pilot)
            assert isinstance(app.screen, DeleteEntryScreen)

            await pilot.press("ctrl+r")
            await wait_until(pilot, lambda: entry_a.name not in catalog.list())

            assert isinstance(app.screen, CatalogScreen)
            assert "to-delete" not in catalog.list_aliases()
            assert entry_a.name not in app.screen._tree_entry_hashes()

    _run(_test())


def test_remove_alias_keeps_entry_and_other_aliases(catalog, entry_a):
    async def _test():
        catalog.add_alias(entry_a.name, "keep-me")
        catalog.add_alias(entry_a.name, "remove-me")
        app = _make_tui(catalog)
        row = CatalogRowData(entry=entry_a, aliases=("keep-me", "remove-me"))
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen
            screen._row_cache = {row.row_key: row}
            screen._render_refresh(catalog.repo.working_dir, (row,))
            await settle(pilot)

            await pilot.press("j", "r")
            await settle(pilot)
            assert isinstance(app.screen, RemoveAliasScreen)

            select = app.screen.query_one("#remove-alias-select", Select)
            select.value = "remove-me"
            await pilot.press("ctrl+r")
            await wait_until(pilot, lambda: "remove-me" not in catalog.list_aliases())

            assert isinstance(app.screen, CatalogScreen)
            assert entry_a.name in catalog.list()
            assert "keep-me" in catalog.list_aliases()
            assert app.screen._row_cache[entry_a.name].aliases == ("keep-me",)

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


def _leaf_label_for(screen: CatalogScreen, entry_hash: str) -> str:
    tree = screen.query_one("#catalog-tree", Tree)
    return next(
        str(leaf.label)
        for branch in tree.root.children
        for leaf in branch.children
        if leaf.data == entry_hash
    )


def test_refresh_attaches_alias_added_after_entry_cached(
    catalog: Catalog, entry_a: CatalogEntry
) -> None:
    """A refresh landing mid-add caches the entry before its alias write;
    the next refresh must attach the alias to the cached row instead of
    leaving the entry permanently rendered as a bare hash."""

    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            # Phase 1 of the add: entry in catalog.yaml, alias not yet.
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            assert screen._row_cache[entry_a.name].aliases == ()

            # Phase 2: the alias write lands.
            catalog.add_alias(entry_a.name, "my-model")

            # Run the refresh body off the main thread, as the worker does.
            await asyncio.to_thread(screen._do_refresh_locked)
            await settle(pilot)

            assert screen._row_cache[entry_a.name].aliases == ("my-model",)
            assert "my-model" in _leaf_label_for(screen, entry_a.name)

    _run(_test())


def test_refresh_moves_alias_repointed_to_new_entry(
    catalog: Catalog, entry_a: CatalogEntry
) -> None:
    """Adding a new revision under an existing alias re-points the alias;
    the old entry's cached row must lose it and the new entry must show it."""

    async def _test():
        catalog.add_alias(entry_a.name, "latest")
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = pilot.app.screen
            row = CatalogRowData(entry=entry_a, aliases=("latest",))
            screen._row_cache = {row.row_key: row}
            screen._render_refresh(catalog.repo.working_dir, (row,))
            await settle(pilot)

            entry_c = catalog.add(xo.memtable({"z": [1, 2]}), aliases=("latest",))

            await asyncio.to_thread(screen._do_refresh_locked)
            await settle(pilot)

            assert screen._row_cache[entry_a.name].aliases == ()
            assert screen._row_cache[entry_c.name].aliases == ("latest",)
            assert "latest" not in _leaf_label_for(screen, entry_a.name)
            assert "latest" in _leaf_label_for(screen, entry_c.name)

    _run(_test())


def _alias_target(aliases: tuple, alias: str) -> str:
    return next(ca.catalog_entry.name for ca in aliases if ca.alias == alias)


def test_get_catalog_aliases_sees_pure_repoint(
    catalog: Catalog, entry_a: CatalogEntry, entry_b: CatalogEntry
) -> None:
    """A pure repoint rewrites only the aliases/ symlink; catalog.yaml already
    lists the alias, so its mtime alone must not key the cache."""
    catalog.add_alias(entry_a.name, "latest")
    yaml_path = catalog.catalog_yaml.yaml_path
    yaml_mtime = yaml_path.stat().st_mtime

    before = _get_catalog_aliases(catalog)
    assert _alias_target(before, "latest") == entry_a.name

    catalog.add_alias(entry_b.name, "latest")
    # the repoint must not have rewritten catalog.yaml, or this test would
    # pass via the yaml mtime without exercising the symlink-state key
    assert yaml_path.stat().st_mtime == yaml_mtime

    after = _get_catalog_aliases(catalog)
    assert _alias_target(after, "latest") == entry_b.name


def test_refresh_moves_alias_repointed_between_existing_entries(
    catalog: Catalog, entry_a: CatalogEntry, entry_b: CatalogEntry
) -> None:
    """A pure repoint between two existing entries touches only the aliases/
    symlink, not catalog.yaml; the refresh must still move the alias."""

    async def _test():
        catalog.add_alias(entry_a.name, "latest")
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = pilot.app.screen
            rows = (
                CatalogRowData(entry=entry_a, aliases=("latest",)),
                CatalogRowData(entry=entry_b),
            )
            screen._row_cache = {r.row_key: r for r in rows}
            screen._render_refresh(catalog.repo.working_dir, rows)
            await settle(pilot)

            # Prime the mtime-keyed alias cache with the pre-repoint state.
            await asyncio.to_thread(screen._do_refresh_locked)
            await settle(pilot)
            assert screen._row_cache[entry_a.name].aliases == ("latest",)

            catalog.add_alias(entry_b.name, "latest")

            await asyncio.to_thread(screen._do_refresh_locked)
            await settle(pilot)

            assert screen._row_cache[entry_a.name].aliases == ()
            assert screen._row_cache[entry_b.name].aliases == ("latest",)
            assert "latest" not in _leaf_label_for(screen, entry_a.name)
            assert "latest" in _leaf_label_for(screen, entry_b.name)

    _run(_test())


def test_cursor_move_updates_schema_preview(
    catalog: Catalog, entry_a: CatalogEntry, entry_b: CatalogEntry
) -> None:
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
                # Move past branch to first leaf (entry_a: id, name, score).
                # Panel render is debounced, so wait for it to settle.
                Press(("j",)),
                WaitUntil(lambda: schema_table.row_count == 3),
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
                WaitUntil(lambda: schema_table.row_count == 1),
                Assert(lambda p: schema_table.get_cell_at((0, 0)) == "value"),
                # Move back to first leaf (entry_a: id, name, score)
                Press(("k",)),
                WaitUntil(lambda: schema_table.row_count == 3),
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
                Assert(lambda p: sql_panel.display is True),
                Assert(lambda p: data_panel.display is False),
                # Switch to data
                Press(("2",)),
                Assert(lambda p: sql_panel.display is False),
                Assert(lambda p: data_panel.display is True),
                # Switch back to SQL
                Press(("1",)),
                Assert(lambda p: sql_panel.display is True),
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
                Assert(lambda p: panel.display is True),
                Press(("v",)),
                Assert(lambda p: panel.display is False),
            )

    _run(_test())


def test_list_revisions_cached_hits_and_invalidates(catalog, entry_a, alias_for_a):
    """Same (alias, HEAD sha) hits cache; a new commit invalidates it."""
    _list_revisions_cached.cache_clear()
    alias = CatalogAlias.from_name(alias_for_a, catalog)
    sha = catalog.repo.head.commit.hexsha

    first = _list_revisions_cached(alias, sha)
    # Identical args -> cache hit returns the same object (no re-walk).
    assert _list_revisions_cached(alias, sha) is first

    # A distinct but value-equal alias (CatalogAlias is @frozen) hits the same
    # cache entry -- the production path after _catalog_aliases_cached rebuilds
    # alias objects on a YAML mtime change.
    alias2 = CatalogAlias.from_name(alias_for_a, catalog)
    assert alias2 is not alias
    assert _list_revisions_cached(alias2, sha) is first

    # A new commit moves HEAD -> different key -> fresh walk.
    catalog.add_alias(entry_a.name, "another-alias")
    sha2 = catalog.repo.head.commit.hexsha
    assert sha2 != sha
    assert _list_revisions_cached(alias, sha2) is not first


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
                Assert(lambda p: panel.display is True),
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
        lambda self, code=None: self._entry.expr.limit(50_000).execute(),
    )


def _raise_load_error(self, code=None):
    raise RuntimeError("mock load error")


@pytest.fixture
def _mock_catalog_run_error(monkeypatch):
    """Make every subprocess call fail with RuntimeError."""
    monkeypatch.setattr(DataViewScreen, "_run_catalog_subprocess", _raise_load_error)


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
    # Each step is wrapped in a lambda so its `source` binds to the prior result.
    assert code == (
        "(lambda source: source.mutate(y=source.x * 2))"
        "((lambda source: source.filter(source.x > 1))(source))"
    )


def test_expr_stack_current_code_single_step():
    base = xo.memtable({"x": [1, 2, 3]})
    step = ExprStep(
        verb="filter", user_input="source.x > 1", code="source.filter(source.x > 1)"
    )
    stack = ExprStack(base_expr=base).push(step)
    assert stack.current_code == "(lambda source: source.filter(source.x > 1))(source)"


def test_expr_stack_current_code_evaluable():
    """current_code must be a single expression that _eval_code can evaluate."""

    base = xo.memtable({"x": [1, 2, 3], "y": [10, 20, 30]})
    step1 = ExprStep(
        verb="filter", user_input="source.x > 1", code="source.filter(source.x > 1)"
    )
    step2 = ExprStep(
        verb="filter", user_input="source.y < 25", code="source.filter(source.y < 25)"
    )
    stack = ExprStack(base_expr=base).push(step1).push(step2)
    result = _eval_code(stack.current_code, base)
    df = result.execute()
    assert len(df) == 1
    assert list(df["x"]) == [2]
    assert list(df["y"]) == [20]


def test_expr_stack_current_code_inner_source_rebinds():
    """Every `source` in a step must bind to the prior step's result, not only the first."""

    base = xo.memtable({"x": [1, 2, 3], "y": [10, 20, 30]})
    step1 = ExprStep(
        verb="freeform",
        user_input='source.select("x")',
        code='source.select("x")',
    )
    # This step references source.x; after select("x"), source.y is gone.
    # If `source` inside the mutate bound to the original base, this would
    # silently succeed using base.x rather than the selected projection.
    step2 = ExprStep(
        verb="freeform",
        user_input="source.mutate(z=source.x * 10)",
        code="source.mutate(z=source.x * 10)",
    )
    stack = ExprStack(base_expr=base).push(step1).push(step2)
    result = _eval_code(stack.current_code, base)
    df = result.execute()
    assert list(df.columns) == ["x", "z"]
    assert list(df["z"]) == [10, 20, 30]


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


def test_data_view_colon_opens_freeform_input(catalog, entry_a):
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

            await pilot.press(":")
            await settle(pilot)

            cmd = data_screen.query_one("#command-input", Input)
            assert cmd.display is True
            assert ":" in str(cmd.border_title)

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
            await pilot.press(":")
            await settle(pilot)

            cmd = data_screen.query_one("#command-input", Input)
            assert cmd.display is True

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
                Press(("s",)),
                Assert(lambda p: panel.display is True),
                Press(("s",)),
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


# ---------------------------------------------------------------------------
# 5. TUI options
# ---------------------------------------------------------------------------


def test_tui_options_defaults():
    cfg = TUI()
    assert cfg.left_ratio == 2
    assert cfg.right_ratio == 3
    assert cfg.revisions_open is False
    assert cfg.git_log_open is False
    assert cfg.row_limit == 10000


def test_tui_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XORQ_TUI_LEFT_RATIO", "7")
    monkeypatch.setenv("XORQ_TUI_REVISIONS_OPEN", "True")
    monkeypatch.setenv("XORQ_TUI_ROW_LIMIT", "42")
    fresh = EnvConfigable.subclass_from_env_file(
        env_templates_dir.joinpath(".env.xorq.template")
    ).from_env()
    assert fresh.XORQ_TUI_LEFT_RATIO == "7"
    assert fresh.XORQ_TUI_REVISIONS_OPEN == "True"
    assert fresh.XORQ_TUI_ROW_LIMIT == "42"


@pytest.mark.parametrize(
    ("row_limit", "expected"),
    [
        pytest.param(123, "123", id="normal"),
        pytest.param(1, "1", id="min"),
    ],
)
def test_catalog_run_cmd_uses_configured_row_limit(
    entry_a: CatalogEntry, row_limit: int, expected: str
) -> None:
    row_data = CatalogRowData(entry=entry_a)
    screen = DataViewScreen(entry=entry_a, row_data=row_data)
    with (
        patch.object(
            screen, "_catalog_base_cmd", return_value=["xorq", "catalog", "run", "x"]
        ),
        options.tui({"row_limit": row_limit}),
    ):
        cmd = screen._catalog_run_cmd()
    # assert on the contiguous flag/value pair, not a bare index lookup, so a
    # future combined --limit=<n> form fails loudly rather than silently.
    assert ["--limit", expected] in [cmd[i : i + 2] for i in range(len(cmd) - 1)], cmd


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("50000", 50000, id="passthrough"),
        pytest.param("1", 1, id="min"),
        pytest.param("0", 1, id="zero-clamped"),
        pytest.param("-5", 1, id="negative-clamped"),
        pytest.param(None, 10000, id="unset-default"),
        pytest.param("", 10000, id="empty-default"),
        pytest.param("abc", 10000, id="malformed-default"),
    ],
)
def test_row_limit_clamp_and_default(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int
) -> None:
    # Exercise the real TUI.row_limit field definition in config.py: the field
    # default is evaluated at import time against env_config, so reload the
    # module with XORQ_TUI_ROW_LIMIT set to re-run the class body. Reading
    # TUI().row_limit or options.tui({...}) would NOT hit the parse/clamp path
    # (import-time eval, and context-manager overrides bypass the max(...,1)
    # floor). Guards the floor (1) + default (10000) contract against drift.
    if raw is None:
        monkeypatch.delenv("XORQ_TUI_ROW_LIMIT", raising=False)
    else:
        monkeypatch.setenv("XORQ_TUI_ROW_LIMIT", raw)
    try:
        reloaded = importlib.reload(xorq.config)
        assert reloaded.TUI().row_limit == expected
    finally:
        # restore module-global env_config/options to the unset-env defaults so
        # later tests in this worker see the original config singletons.
        monkeypatch.undo()
        importlib.reload(xorq.config)


def test_tui_options_apply_column_widths(catalog):
    async def _test():
        with options.tui(
            {
                "left_ratio": 5,
                "right_ratio": 7,
                "revisions_open": True,
                "git_log_open": True,
            }
        ):
            app = _make_tui(catalog)
            async with app.run_test(size=(120, 40)) as pilot:
                await settle(pilot)
                screen = app.screen
                assert str(screen.query_one("#left-column").styles.width) == "5fr"
                assert str(screen.query_one("#right-column").styles.width) == "7fr"
                assert screen.query_one("#revisions-panel").display is True
                assert screen.query_one("#git-log-panel").display is True

    _run(_test())


def test_highlight_debounce_zero_renders_synchronously(catalog, entry_a, entry_b):
    """delay <= 0 takes the synchronous render branch; no timer is scheduled."""
    with options.tui({"highlight_debounce": 0.0}):

        async def _test():
            app = _make_tui(catalog)
            async with app.run_test(size=(120, 40)) as pilot:
                screen, _ = await _populate_tree(pilot, catalog, entry_a, entry_b)
                await run_script(
                    pilot,
                    Press(("j",)),
                    Assert(lambda p: screen._highlight_timer is None),
                )

        _run(_test())


def test_cancel_highlight_timer_stops_pending(catalog):
    """A pending timer is stopped and cleared by _cancel_highlight_timer."""

    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen
            screen._highlight_timer = screen.set_timer(100, lambda: None)
            screen._cancel_highlight_timer()
            assert screen._highlight_timer is None

    _run(_test())


def test_on_unmount_cancels_pending_timer(catalog):
    """Dismissing the screen with a timer pending cancels it (no NoMatches)."""

    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen
            screen._highlight_timer = screen.set_timer(100, lambda: None)
            screen.on_unmount()
            assert screen._highlight_timer is None

    _run(_test())


def test_render_highlighted_node_noop_when_detached():
    """An unmounted screen short-circuits before querying removed widgets."""
    screen = CatalogScreen()
    assert screen.is_attached is False
    # query_one would raise NoMatches if the is_attached guard did not return.
    screen._render_highlighted_node()
    assert screen._highlight_timer is None


def test_load_revisions_preview_renders_rows(catalog, entry_a, alias_for_a):
    """Highlighting an aliased entry runs the worker: resolve HEAD, walk, render."""
    _list_revisions_cached.cache_clear()
    # debounce 0 -> _populate_tree's highlight renders synchronously, leaving no
    # pending timer to fire mid-settle and clear the table after the worker runs.
    with options.tui({"highlight_debounce": 0.0}):

        async def _test():
            app = _make_tui(catalog)
            async with app.run_test(size=(120, 40)) as pilot:
                screen, _ = await _populate_tree(pilot, catalog, entry_a)
                alias = CatalogAlias.from_name(alias_for_a, catalog)
                screen._load_revisions_preview(alias)
                await settle(pilot)
                rev_table = screen.query_one("#revisions-preview-table", DataTable)
                assert rev_table.row_count >= 1

        _run(_test())


def test_load_revisions_preview_swallows_attribute_error(catalog, entry_a):
    """A catalog_alias missing the repo attribute chain is caught, not raised."""

    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            # object() has no .catalog_entry -> AttributeError inside the worker.
            screen._load_revisions_preview(object())
            await settle(pilot)
            rev_table = screen.query_one("#revisions-preview-table", DataTable)
            assert rev_table.row_count == 0

    _run(_test())


# ---------------------------------------------------------------------------
# 14. SQL highlight pipeline: pure unit tests
# ---------------------------------------------------------------------------


def test_pygments_tokens_cached_returns_same_object():
    r1 = _pygments_tokens("SELECT id FROM t")
    r2 = _pygments_tokens("SELECT id FROM t")
    assert r1 is r2


def test_pygments_tokens_keyword_bold():
    tokens = _pygments_tokens("SELECT 1")
    select_styles = [s for v, s in tokens if v.strip().upper() == "SELECT"]
    assert any("bold" in s for s in select_styles)


def test_pygments_to_text_fresh_object_per_call():
    t1 = _pygments_to_text("SELECT 1")
    t2 = _pygments_to_text("SELECT 1")
    assert t1 is not t2


def test_pygments_to_text_word_wrap_config():
    text = _pygments_to_text("SELECT 1")
    assert text.no_wrap is False
    assert text.overflow == "fold"


def test_render_sql_text_fallback_for_large_query():
    with options.tui({"sql_highlight_max_lines": 10}):
        big_sql = "SELECT 1\n" * 11
        text = _render_sql_text(big_sql)
        plain = text.plain
        assert plain.startswith("-- syntax highlighting disabled")
        assert "10 lines" in plain
        assert "SELECT 1" in plain


def test_render_sql_text_disabled_when_max_lines_zero():
    with options.tui({"sql_highlight_max_lines": 0}):
        text = _render_sql_text("SELECT 1")
        plain = text.plain
        assert plain.startswith("-- syntax highlighting disabled\n")
        assert "lines" not in plain.split("\n")[0]
        assert "SELECT 1" in plain


def test_render_sql_text_highlights_small_query():
    text = _render_sql_text("SELECT 1")
    assert text.no_wrap is False
    assert not text.plain.startswith("-- syntax highlighting disabled")


def test_render_sql_text_disabled_at_exact_boundary():
    # raw.count("\n") == max_lines triggers >= guard → highlighting disabled
    with options.tui({"sql_highlight_max_lines": 3}):
        at_boundary = "SELECT 1\nFROM t\nWHERE x = 1\n"
        assert at_boundary.count("\n") == 3
        text = _render_sql_text(at_boundary)
        assert text.plain.startswith("-- syntax highlighting disabled")


# ---------------------------------------------------------------------------
# 15. _render_sql_dag: pure unit tests
# ---------------------------------------------------------------------------


def test_render_sql_dag_empty():
    assert _render_sql_dag(()) == ""


def test_render_sql_dag_single():
    result = _render_sql_dag((("main", "duckdb", "SELECT 1"),))
    assert "-- [main] (duckdb)" in result
    assert "SELECT 1" in result
    assert "↓" not in result


def test_render_sql_dag_single_non_main_name():
    name = "abcdef1234567890abcdef12"
    result = _render_sql_dag(((name, "duckdb", "SELECT 1"),))
    assert f"-- [{name[:12]}]" in result
    assert "-- [main]" not in result


def test_render_sql_dag_multiple_no_deps():
    sqls = (
        ("main", "duckdb", "SELECT * FROM t"),
        ("abcdef1234567890abcdef12", "duckdb", "SELECT 1"),
    )
    result = _render_sql_dag(sqls)
    assert "↓" in result
    assert "-- [main]" in result
    assert "-- [abcdef123456]" in result


def test_render_sql_dag_dep_ordered_before_main():
    dep_hash = "ab" * 10  # 20 hex chars — matches FROM "..." regex
    main_sql = f'SELECT x FROM "{dep_hash}"'
    sqls = (
        ("main", "duckdb", main_sql),
        (dep_hash, "duckdb", "SELECT 1 AS x"),
    )
    result = _render_sql_dag(sqls)
    dep_pos = result.index(dep_hash[:12])
    main_pos = result.index("-- [main]")
    assert dep_pos < main_pos


# ---------------------------------------------------------------------------
# 16. _pygments_tokens: italic branch (SQL comments)
# ---------------------------------------------------------------------------


def test_pygments_tokens_italic_for_sql_comment():
    tokens = _pygments_tokens("-- this is a sql comment")
    all_styles = [s for _, s in tokens]
    assert any("italic" in s for s in all_styles)


# ---------------------------------------------------------------------------
# 17. CatalogScreen navigation: DataTable focus, cycle_focus, cached status
# ---------------------------------------------------------------------------


def test_tab_cycle_focus_no_crash(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            await pilot.press("tab")
            await settle(pilot)
            await pilot.press("shift+tab")
            await settle(pilot)
            assert isinstance(app.screen, CatalogScreen)

    _run(_test())


def test_h_l_with_datatable_focused(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            schema_table = app.screen.query_one("#schema-preview-table")
            schema_table.focus()
            await settle(pilot)
            await pilot.press("h")
            await settle(pilot)
            await pilot.press("l")
            await settle(pilot)
            assert isinstance(app.screen, CatalogScreen)

    _run(_test())


def test_j_k_with_datatable_focused(catalog):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            schema_table = app.screen.query_one("#schema-preview-table")
            schema_table.focus()
            await settle(pilot)
            await pilot.press("j")
            await settle(pilot)
            await pilot.press("k")
            await settle(pilot)
            assert isinstance(app.screen, CatalogScreen)

    _run(_test())


def test_render_status_includes_cached_count(catalog, entry_cached):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(pilot)
            screen = app.screen
            entry_cached.expr.execute()
            row = CatalogRowData(entry=entry_cached)
            screen._row_cache = {row.row_key: row}
            screen._render_status("09:00:00", catalog.repo.working_dir)
            await settle(pilot)
            status_text = str(screen.query_one("#status-bar", Static).content)
            assert "cached" in status_text

    _run(_test())


def test_tree_expand_collapses_then_enters_first_child(catalog, entry_a, entry_b):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a, entry_b)
            tree = screen.query_one("#catalog-tree")

            await run_script(
                pilot,
                # cursor on branch (source), press l to expand (already expanded) → enter first child
                Assert(lambda p: tree.cursor_node.data == "source"),
                Press(("l",)),
                Assert(lambda p: tree.cursor_node.data == entry_a.name),
                # press h on leaf → select parent branch, collapse it
                Press(("h",)),
                Assert(lambda p: tree.cursor_node.data == "source"),
            )

    _run(_test())


# ---------------------------------------------------------------------------
# 18. DataViewScreen: actions (sort, drop, undo, redo, navigate, persist)
# ---------------------------------------------------------------------------


def test_data_view_sort_asc_pushes_step(catalog, entry_a, _mock_catalog_run):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            await run_script(pilot, Press(("j",)), Press(("e",)))
            await settle(pilot)
            data_screen = app.screen
            assert isinstance(data_screen, DataViewScreen)
            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            await pilot.press("]")
            await settle(pilot)
            assert data_screen._stack.cursor == 1
            assert data_screen._stack.steps[0].verb == "order_by"
            assert "desc" not in data_screen._stack.steps[0].code

    _run(_test())


def test_data_view_sort_desc_pushes_step(catalog, entry_a, _mock_catalog_run):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            await run_script(pilot, Press(("j",)), Press(("e",)))
            await settle(pilot)
            data_screen = app.screen
            assert isinstance(data_screen, DataViewScreen)
            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            await pilot.press("[")
            await settle(pilot)
            assert data_screen._stack.cursor == 1
            assert data_screen._stack.steps[0].verb == "order_by"
            assert "desc" in data_screen._stack.steps[0].code

    _run(_test())


def test_data_view_drop_column_pushes_step(catalog, entry_a, _mock_catalog_run):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            await run_script(pilot, Press(("j",)), Press(("e",)))
            await settle(pilot)
            data_screen = app.screen
            assert isinstance(data_screen, DataViewScreen)
            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            await pilot.press("d")
            await settle(pilot)
            assert data_screen._stack.cursor == 1
            assert data_screen._stack.steps[0].verb == "drop"

    _run(_test())


def test_data_view_undo_after_step(catalog, entry_a, _mock_catalog_run):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            await run_script(pilot, Press(("j",)), Press(("e",)))
            await settle(pilot)
            data_screen = app.screen
            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            await pilot.press("]")
            await settle(pilot)
            assert data_screen._stack.cursor == 1

            await pilot.press("u")
            await settle(pilot)
            assert data_screen._stack.cursor == 0
            assert data_screen._stack.can_redo

    _run(_test())


def test_data_view_redo_after_undo(catalog, entry_a, _mock_catalog_run):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            await run_script(pilot, Press(("j",)), Press(("e",)))
            await settle(pilot)
            data_screen = app.screen
            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            await pilot.press("]")
            await settle(pilot)
            await pilot.press("u")
            await settle(pilot)
            assert data_screen._stack.cursor == 0

            await pilot.press("ctrl+r")
            await settle(pilot)
            assert data_screen._stack.cursor == 1

    _run(_test())


def test_data_view_navigation_keys_no_crash(catalog, entry_a, _mock_catalog_run):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            await run_script(pilot, Press(("j",)), Press(("e",)))
            await settle(pilot)
            data_screen = app.screen
            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            for key in ("j", "k", "h", "l"):
                await pilot.press(key)
                await settle(pilot)
            assert isinstance(app.screen, DataViewScreen)

    _run(_test())


def test_data_view_scroll_top_and_bottom(catalog, entry_a, _mock_catalog_run):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            await run_script(pilot, Press(("j",)), Press(("e",)))
            await settle(pilot)
            data_screen = app.screen
            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            await pilot.press("g")
            await settle(pilot)
            assert data_table.cursor_row == 0

            await pilot.press("shift+g")
            await settle(pilot)
            assert data_table.cursor_row == data_table.row_count - 1

    _run(_test())


def test_data_view_stack_browser_shows_step_content(
    catalog, entry_a, _mock_catalog_run
):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            await run_script(pilot, Press(("j",)), Press(("e",)))
            await settle(pilot)
            data_screen = app.screen
            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            await pilot.press("]")
            await settle(pilot)

            await pilot.press("s")
            await settle(pilot)
            panel = data_screen.query_one("#stack-browser-panel")
            assert panel.display is True
            content_widget = data_screen.query_one("#stack-browser-content", Static)
            assert "order_by" in str(content_widget.content)

    _run(_test())


def test_data_view_freeform_submit_pushes_step(catalog, entry_a, _mock_catalog_run):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            await run_script(pilot, Press(("j",)), Press(("e",)))
            await settle(pilot)
            data_screen = app.screen
            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            await pilot.press(":")
            await settle(pilot)
            cmd = data_screen.query_one("#command-input")
            cmd.value = 'source.select("id")'
            await pilot.press("enter")
            await settle(pilot)

            assert data_screen._stack.cursor == 1
            assert data_screen._stack.steps[0].verb == "freeform"

    _run(_test())


def test_data_view_persist_prompt_shows_after_step(catalog, entry_a, _mock_catalog_run):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            await run_script(pilot, Press(("j",)), Press(("e",)))
            await settle(pilot)
            data_screen = app.screen
            data_table = data_screen.query_one("#data-view-table", DataTable)
            await wait_until(pilot, lambda: data_table.row_count > 0)

            await pilot.press("]")
            await settle(pilot)
            assert data_screen._stack.cursor == 1

            await pilot.press("w")
            await settle(pilot)
            cmd = data_screen.query_one("#command-input")
            assert cmd.display is True
            assert "save" in str(cmd.border_title).lower()

    _run(_test())


def test_data_view_load_error_shows_in_status(
    catalog, entry_a, _mock_catalog_run_error
):
    async def _test():
        app = _make_tui(catalog)
        async with app.run_test(size=(120, 40)) as pilot:
            screen, _ = await _populate_tree(pilot, catalog, entry_a)
            await run_script(pilot, Press(("j",)), Press(("e",)))
            await settle(pilot)
            data_screen = app.screen
            assert isinstance(data_screen, DataViewScreen)
            status = data_screen.query_one("#data-view-status", Static)
            await wait_until(pilot, lambda: "Error" in str(status.content))

    _run(_test())
