"""`xorq catalog rebase` (#2323), over the sqlite world `test_drift` uses."""

from __future__ import annotations

import shutil
import sys
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
from click.testing import CliRunner, Result

import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.caching import ParquetCache
from xorq.catalog import drift
from xorq.catalog import rebase as rebase_module
from xorq.catalog.catalog import Catalog, CatalogAlias
from xorq.catalog.cli import cli
from xorq.catalog.enums import RebaseStatus
from xorq.catalog.exceptions import RebaseError
from xorq.catalog.rebase import rebase_entry, recorded_python_minor
from xorq.catalog.tests.conftest import (
    TEST_WHEEL_NAME,
    _annex_available,
    alias_target_hash,
)
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.expr.relations import pin_cache
from xorq.ibis_yaml.compiler import build_expr
from xorq.ibis_yaml.enums import DumpFiles


# Most tests exercise rebase's own logic, which doesn't vary with storage; the
# ones that write entries or move aliases run on every backend.
ALL_BACKENDS = pytest.mark.parametrize(
    "backend_type",
    (
        "git",
        pytest.param(
            "annex",
            marks=pytest.mark.skipif(
                not _annex_available, reason="git-annex not installed"
            ),
        ),
        "pointer",
    ),
)


@pytest.fixture
def backend_type() -> str:
    return "git"


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})
GROWN = RECORDED.append_column("c", pa.array([1.5, 2.5], pa.float64()))


def replace_t(world: SimpleNamespace, table: pa.Table) -> None:
    world.con.drop_table("t", force=True)
    world.con.create_table("t", table.to_pandas())


def commit_count(catalog: Catalog) -> int:
    return len(list(catalog.repo.iter_commits()))


def reopen(world: SimpleNamespace) -> Catalog:
    return Catalog.from_kwargs(path=world.catalog_path, init=False)


@pytest.fixture
def world(tmp_path: Path, catalog_path: str) -> SimpleNamespace:
    """A sqlite table, and an entry over it aliased `live` and `staging`."""
    db_path = tmp_path / "live.sqlite"
    con = SqliteBackend().connect(str(db_path))
    con.create_table("t", RECORDED.to_pandas())
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    t = con.table("t")
    entry = catalog.add(t.filter(t.a > 1), aliases=("live", "staging"))
    return SimpleNamespace(
        db_path=db_path,
        con=con,
        catalog=catalog,
        catalog_path=catalog_path,
        name=entry.name,
    )


def rebase(runner: CliRunner, world: SimpleNamespace, *args: str) -> Result:
    return runner.invoke(
        cli, ["--path", world.catalog_path, "rebase", *(args or (world.name,))]
    )


def assert_nothing_written(world: SimpleNamespace, commits: int) -> None:
    catalog = reopen(world)
    assert catalog.list() == [world.name]
    assert commit_count(catalog) == commits


@ALL_BACKENDS
def test_a_grown_source_rebases_to_a_new_entry(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, GROWN)

    result = rebase(runner, world)
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    assert new != world.name
    assert "DatabaseTable t: changed" in result.stderr
    assert f"Rebased {world.name} -> {new}" in result.stderr
    catalog = reopen(world)
    assert set(catalog.list()) == {world.name, new}
    assert catalog.get_catalog_entry(new).columns == ("a", "b", "c")
    assert catalog.get_catalog_entry(world.name).columns == ("a", "b")


@ALL_BACKENDS
def test_no_drift_prints_the_same_hash_and_commits_nothing(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    commits = commit_count(world.catalog)

    result = rebase(runner, world)
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{world.name}\n"
    assert "no drift" in result.stderr
    assert_nothing_written(world, commits)


@ALL_BACKENDS
def test_every_alias_moves_by_default(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, GROWN)

    new = rebase(runner, world).stdout.strip()
    catalog = reopen(world)
    assert alias_target_hash(catalog, "live") == new
    assert alias_target_hash(catalog, "staging") == new


@ALL_BACKENDS
def test_only_alias_narrows_and_alias_adds(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, GROWN)

    result = rebase(runner, world, "live", "--only-alias", "live", "-a", "v2")
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    catalog = reopen(world)
    assert alias_target_hash(catalog, "live") == new
    assert alias_target_hash(catalog, "v2") == new
    assert alias_target_hash(catalog, "staging") == world.name


def test_no_drift_says_the_alias_was_not_added(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    commits = commit_count(world.catalog)

    result = rebase(runner, world, world.name, "-a", "v2")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{world.name}\n"
    assert "Alias 'v2' not added" in result.stderr
    assert "v2" not in reopen(world).list_aliases()
    assert_nothing_written(world, commits)


def test_an_unknown_alias_to_move_writes_nothing(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, GROWN)
    commits = commit_count(world.catalog)

    result = rebase(runner, world, world.name, "--only-alias", "nope")
    assert result.exit_code == 1
    assert "no alias nope" in result.stderr
    assert_nothing_written(world, commits)


def fail_nth_alias_move(
    monkeypatch: pytest.MonkeyPatch, n: int, after_write: bool = False
) -> None:
    """Fail the ``n``th alias move; ``after_write`` fails it once the symlink is
    written, as a failed commit would."""
    add = CatalogAlias.add
    calls = []

    def failing_add(self: CatalogAlias) -> None:
        calls.append(self.alias)
        if len(calls) == n:
            if after_write:
                self._add()
            raise RuntimeError("alias move failed")
        return add(self)

    monkeypatch.setattr(CatalogAlias, "add", failing_add)


@ALL_BACKENDS
def test_a_failed_alias_move_rolls_back_the_new_entry(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    fail_nth_alias_move(monkeypatch, 2)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_entry(world.catalog.get_catalog_entry(world.name))
    monkeypatch.undo()

    catalog = reopen(world)
    assert catalog.list() == [world.name]
    assert alias_target_hash(catalog, "live") == world.name
    assert alias_target_hash(catalog, "staging") == world.name


@ALL_BACKENDS
def test_a_failed_alias_move_keeps_an_entry_that_already_existed(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    earlier = rebase_entry(world.catalog.get_catalog_entry(world.name)).new_entry.name
    catalog = reopen(world)
    for alias in ("live", "staging"):
        catalog.add_alias(world.name, alias)

    fail_nth_alias_move(monkeypatch, 2)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_entry(catalog.get_catalog_entry(world.name))
    monkeypatch.undo()

    catalog = reopen(world)
    assert set(catalog.list()) == {world.name, earlier}
    assert alias_target_hash(catalog, "live") == world.name
    assert alias_target_hash(catalog, "staging") == world.name


@ALL_BACKENDS
def test_a_failed_alias_move_restores_the_extra_alias(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = world.con.table("t")
    other = world.catalog.add(t.filter(t.a > 0), aliases=("v2",)).name
    replace_t(world, GROWN)

    fail_nth_alias_move(monkeypatch, 1)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_entry(world.catalog.get_catalog_entry(world.name), alias="v2")
    monkeypatch.undo()

    catalog = reopen(world)
    assert set(catalog.list()) == {world.name, other}
    assert alias_target_hash(catalog, "v2") == other
    assert alias_target_hash(catalog, "live") == world.name


def pull_then(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[Catalog], object]
) -> None:
    """Stand in for a sync whose pull applies ``change`` to the catalog."""

    @contextmanager
    def pulling(self, sync):
        if sync:
            change(self)
        yield

    monkeypatch.setattr(Catalog, "maybe_synchronizing", pulling)


@ALL_BACKENDS
def test_a_rollback_keeps_an_entry_the_pull_brought_in(
    world: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    pushed = rebase_entry(world.catalog.get_catalog_entry(world.name)).new_entry
    archive = Path(shutil.copy(pushed.catalog_path, tmp_path))
    catalog = reopen(world)
    catalog.remove(pushed.name)
    for alias in ("live", "staging"):
        catalog.add_alias(world.name, alias)

    pull_then(monkeypatch, lambda c: c.add(archive, sync=False))
    fail_nth_alias_move(monkeypatch, 2)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_entry(catalog.get_catalog_entry(world.name))
    monkeypatch.undo()

    assert set(reopen(world).list()) == {world.name, pushed.name}


@ALL_BACKENDS
def test_a_rollback_restores_an_alias_to_where_the_pull_moved_it(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = world.con.table("t")
    other = world.catalog.add(t.filter(t.a > 0)).name
    replace_t(world, GROWN)

    # `catalog.add` overwrites `fresh`; the second `add_alias`, moving
    # `live`, fails.
    pull_then(monkeypatch, lambda c: c.add_alias(other, "fresh", sync=False))
    fail_nth_alias_move(monkeypatch, 2)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_entry(world.catalog.get_catalog_entry(world.name), alias="fresh")
    monkeypatch.undo()

    catalog = reopen(world)
    assert alias_target_hash(catalog, "fresh") == other
    assert alias_target_hash(catalog, "live") == world.name
    assert alias_target_hash(catalog, "staging") == world.name


@ALL_BACKENDS
def test_a_rebase_leaves_an_alias_the_pull_moved_off_the_old_entry(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = world.con.table("t")
    other = world.catalog.add(t.filter(t.a > 0)).name
    replace_t(world, GROWN)

    pull_then(monkeypatch, lambda c: c.add_alias(other, "live", sync=False))
    result = rebase_entry(world.catalog.get_catalog_entry(world.name))
    monkeypatch.undo()

    assert result.moved_aliases == ("staging",)
    assert result.skipped_aliases == ("live",)
    catalog = reopen(world)
    assert alias_target_hash(catalog, "live") == other
    assert alias_target_hash(catalog, "staging") == result.new_entry.name


@ALL_BACKENDS
def test_a_rebase_leaves_an_alias_the_pull_removed(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)

    pull_then(monkeypatch, lambda c: c.remove_alias("live", sync=False))
    result = rebase_entry(world.catalog.get_catalog_entry(world.name))
    monkeypatch.undo()

    assert result.moved_aliases == ("staging",)
    assert result.skipped_aliases == ("live",)
    catalog = reopen(world)
    assert "live" not in catalog.list_aliases()
    assert alias_target_hash(catalog, "staging") == result.new_entry.name


@ALL_BACKENDS
def test_an_extra_alias_the_pull_moved_off_still_lands_on_the_new_entry(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = world.con.table("t")
    other = world.catalog.add(t.filter(t.a > 0)).name
    replace_t(world, GROWN)

    pull_then(monkeypatch, lambda c: c.add_alias(other, "live", sync=False))
    result = rebase_entry(world.catalog.get_catalog_entry(world.name), alias="live")
    monkeypatch.undo()

    assert result.skipped_aliases == ()
    assert set(result.moved_aliases) == {"live", "staging"}
    catalog = reopen(world)
    assert alias_target_hash(catalog, "live") == result.new_entry.name
    assert alias_target_hash(catalog, "staging") == result.new_entry.name


def test_the_cli_reports_an_alias_the_pull_moved_off(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = world.con.table("t")
    other = world.catalog.add(t.filter(t.a > 0)).name
    replace_t(world, GROWN)

    pull_then(monkeypatch, lambda c: c.add_alias(other, "live", sync=False))
    result = rebase(runner, world)
    monkeypatch.undo()

    assert result.exit_code == 0, result.output
    assert f"Alias 'live' not moved: no longer on {world.name}" in result.stderr
    assert alias_target_hash(reopen(world), "live") == other


@ALL_BACKENDS
def test_a_rollback_restores_an_alias_whose_commit_failed(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The symlink is written before the commit, so a failed commit leaves the
    alias on the new entry, where removing that entry would take it along."""
    fail_nth_alias_move(monkeypatch, 2, after_write=True)
    replace_t(world, GROWN)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_entry(world.catalog.get_catalog_entry(world.name))
    monkeypatch.undo()

    catalog = reopen(world)
    assert catalog.list() == [world.name]
    assert alias_target_hash(catalog, "live") == world.name
    assert alias_target_hash(catalog, "staging") == world.name


def test_a_failed_alias_move_exits_one_from_the_cli(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    fail_nth_alias_move(monkeypatch, 2)

    result = rebase(runner, world)
    assert result.exit_code == 1
    assert "alias move failed" in result.stderr
    assert reopen(world).list() == [world.name]


def test_a_failed_rollback_surfaces_the_error_that_caused_it(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    fail_nth_alias_move(monkeypatch, 2)

    def failing_remove(self, name, sync=True):
        raise OSError("rollback failed")

    monkeypatch.setattr(Catalog, "remove", failing_remove)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_entry(world.catalog.get_catalog_entry(world.name))


def test_a_str_cache_dir_rebases(
    runner: CliRunner, world: SimpleNamespace, tmp_path: Path
) -> None:
    replace_t(world, GROWN)

    result = rebase(runner, world, world.name, "--cache-dir", str(tmp_path / "c"))
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() != world.name


@pytest.mark.parametrize(
    "target", ("recorded_python_minor", "harvest_entry_from_zip", "make_profile")
)
def test_an_unreadable_archive_or_profile_exits_two(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    replace_t(world, GROWN)

    def unreadable(*_):
        raise ValueError("corrupt")

    monkeypatch.setattr(f"xorq.catalog.rebase.{target}", unreadable)
    commits = commit_count(world.catalog)

    result = rebase(runner, world)
    assert result.exit_code == 2, result.output
    assert f"{world.name} is unreadable: ValueError: corrupt" in result.stderr
    assert_nothing_written(world, commits)


@pytest.mark.parametrize(
    "dropped",
    (
        pytest.param(".whl", id="wheel"),
        pytest.param(DumpFiles.requirements, id="requirements"),
    ),
)
def test_an_archive_missing_its_bundle_exits_two(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    dropped: str,
) -> None:
    replace_t(world, GROWN)
    harvest = rebase_module.harvest_entry_from_zip

    def without(zf: zipfile.ZipFile, harvest_dir: Path) -> tuple:
        wheels, requirements, pin = harvest(zf, harvest_dir)
        if dropped == DumpFiles.requirements:
            return wheels, None, pin
        return [], requirements, pin

    monkeypatch.setattr(rebase_module, "harvest_entry_from_zip", without)
    commits = commit_count(world.catalog)

    result = rebase(runner, world)
    assert result.exit_code == 2, result.output
    assert "for the rebased entry" in result.stderr
    assert_nothing_written(world, commits)


def test_a_pinned_entry_exits_one_and_names_unpin(
    runner: CliRunner, world: SimpleNamespace, tmp_path: Path
) -> None:
    path = tmp_path / "t.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    t = deferred_read_parquet(path, xo.connect(), table_name="t")
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)
    pinned = world.catalog.add(
        pin_cache(t.filter(t.a > 1).cache(cache=cache), ensure_materialized=True)
    ).name
    commits = commit_count(world.catalog)

    result = rebase(runner, world, pinned)
    assert result.exit_code == 1
    assert f"xorq catalog unpin {pinned}" in result.stderr
    assert result.stdout == ""
    catalog = reopen(world)
    assert set(catalog.list()) == {world.name, pinned}
    assert commit_count(catalog) == commits


def test_a_deleted_database_exits_two_and_is_not_recreated(
    runner: CliRunner,
    world: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world.con.disconnect()
    world.db_path.unlink()
    commits = commit_count(world.catalog)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    result = rebase(runner, world)
    assert result.exit_code == 2, result.output
    assert "unreachable" in result.stderr
    assert not world.db_path.exists()
    assert not any(cwd.iterdir())
    assert_nothing_written(world, commits)


def test_an_unprobed_database_is_not_recreated(
    runner: CliRunner, world: SimpleNamespace, tmp_path: Path
) -> None:
    """A parquet read is probed without its sqlite profile; the load isn't."""
    path = tmp_path / "t.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    db = tmp_path / "ingest.sqlite"
    con = SqliteBackend().connect(str(db))
    name = world.catalog.add(
        deferred_read_parquet(path, con, table_name="ingested"), relocate_reads=False
    ).name
    con.disconnect()
    db.unlink()
    GROWN.to_pandas().to_parquet(path, index=False)
    commits = commit_count(world.catalog)

    result = rebase(runner, world, name)
    assert result.exit_code == 2, result.output
    assert f"database {db} does not exist" in result.stderr
    assert not db.exists()
    assert commit_count(reopen(world)) == commits


def test_a_dropped_referenced_column_is_a_conflict(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, pa.table({"b": ["x", "y"]}))
    commits = commit_count(world.catalog)

    result = rebase(runner, world)
    assert result.exit_code == 4, result.output
    assert "could not rebuild" in result.stderr
    assert_nothing_written(world, commits)


@pytest.mark.parametrize(
    "t_drift, exit_code",
    (
        pytest.param("drop", 4, id="gone-outranks-unreachable"),
        pytest.param("grow", 2, id="changed-refuses-nothing"),
    ),
)
def test_the_worst_refusal_picks_the_exit_code(
    runner: CliRunner,
    world: SimpleNamespace,
    tmp_path: Path,
    t_drift: str,
    exit_code: int,
) -> None:
    other_db = tmp_path / "other.sqlite"
    other = SqliteBackend().connect(str(other_db))
    other.create_table("u", RECORDED.to_pandas())
    t = world.con.table("t")
    name = world.catalog.add(t.union(other.table("u").into_backend(world.con))).name
    other.disconnect()
    other_db.unlink()
    if t_drift == "drop":
        world.con.drop_table("t")
    else:
        replace_t(world, GROWN)

    result = rebase(runner, world, name)
    assert result.exit_code == exit_code, result.output
    assert not other_db.exists()


def test_a_dropped_table_is_a_conflict(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    world.con.drop_table("t")
    commits = commit_count(world.catalog)

    result = rebase(runner, world)
    assert result.exit_code == 4, result.output
    assert "table-missing" in result.stderr
    assert_nothing_written(world, commits)


def test_a_python_minor_mismatch_refuses_unless_overridden(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    monkeypatch.setattr("xorq.catalog.rebase.recorded_python_minor", lambda _: (3, 0))
    commits = commit_count(world.catalog)

    refused = rebase(runner, world)
    assert refused.exit_code == 1
    assert "built on Python 3.0" in refused.stderr
    assert "--ignore-venv-mismatch" in refused.stderr
    assert_nothing_written(world, commits)

    overridden = rebase(runner, world, world.name, "--ignore-venv-mismatch")
    assert overridden.exit_code == 0, overridden.output
    assert overridden.stdout.strip() != world.name


def test_the_recorded_python_minor_is_read_from_the_archive(
    world: SimpleNamespace,
) -> None:
    entry = world.catalog.get_catalog_entry(world.name)
    assert recorded_python_minor(entry) == tuple(sys.version_info[:2])


@ALL_BACKENDS
def test_the_rebased_archive_inherits_wheels_and_requirements(
    runner: CliRunner,
    world: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = world.con.table("t")
    build_path = build_expr(t.filter(t.a > 0), builds_dir=tmp_path / "builds")
    (build_path / TEST_WHEEL_NAME).write_bytes(b"not a real wheel")
    (build_path / DumpFiles.requirements).write_text("pinned-dep==1.0\n")
    name = world.catalog.add(build_path).name
    replace_t(world, GROWN)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    result = rebase(runner, world, name)
    assert result.exit_code == 0, result.output
    new = reopen(world).get_catalog_entry(result.stdout.strip())
    with zipfile.ZipFile(new.catalog_path) as zf:
        members = {Path(member).name: member for member in zf.namelist()}
        requirements = zf.read(members[DumpFiles.requirements])
    assert [m for m in members if m.endswith(".whl")] == [TEST_WHEEL_NAME]
    assert requirements == b"pinned-dep==1.0\n"


def test_an_entry_with_only_unprobed_sources_is_attempted(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-derived and settled by the hash, but not declared drift-free."""
    monkeypatch.setattr(drift, "is_checkable", lambda leaf, record: False)
    commits = commit_count(world.catalog)

    result = rebase(runner, world)
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{world.name}\n"
    assert "No source could be probed (t)" in result.stderr
    assert "drift not ruled out" in result.stderr
    assert "no drift" not in result.stderr
    assert_nothing_written(world, commits)

    attempted = rebase_entry(world.catalog.get_catalog_entry(world.name))
    assert (attempted.status, attempted.unprobed) == (RebaseStatus.ATTEMPTED, ("t",))


def test_an_entry_with_some_unprobed_sources_is_refused(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unprobed source would come back unchanged beside the refreshed one."""
    u = world.con.create_table("u", RECORDED.to_pandas())
    t = world.con.table("t")
    name = world.catalog.add(t.union(u)).name
    monkeypatch.setattr(drift, "is_checkable", lambda leaf, record: leaf.name != "u")
    commits = commit_count(world.catalog)

    result = rebase(runner, world, name)
    assert result.exit_code == 1, result.output
    assert "u cannot be probed without writing to it" in result.stderr
    assert result.stdout == ""
    assert commit_count(reopen(world)) == commits


def test_a_rebase_error_carries_its_exit_code() -> None:
    error = RebaseError("refused", 2)
    assert (str(error), error.exit_code) == ("refused", 2)
