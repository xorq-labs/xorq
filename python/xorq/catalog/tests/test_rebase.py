"""`xorq catalog rebase` (#2323), over the sqlite world `test_drift` uses."""

from __future__ import annotations

import json
import shutil
import sys
import warnings
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import click
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
from xorq.catalog.enums import RebaseExit, RebaseStatus
from xorq.catalog.exceptions import RebaseError
from xorq.catalog.rebase import (
    RebaseConflict,
    RebaseResult,
    rebase_entry,
    recorded_python_minor,
)
from xorq.catalog.tests.conftest import (
    RECORDED,
    TEST_WHEEL_NAME,
    _annex_available,
    alias_target_hash,
)
from xorq.common.exceptions import UnmatchedSourceError
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.expr.relations import pin_cache
from xorq.ibis_yaml.compiler import ExprDumper, build_expr
from xorq.ibis_yaml.enums import DumpFiles


# Most tests exercise rebase's own logic, which doesn't vary with storage; the
# basic add, no-op and rollback paths run on every backend. Aliases are git
# symlinks on all of them.
ALL_BACKENDS = pytest.mark.parametrize(
    "backend_type",
    (
        pytest.param("git", id="git"),
        pytest.param(
            "annex",
            marks=pytest.mark.skipif(
                not _annex_available, reason="git-annex not installed"
            ),
            id="annex",
        ),
        pytest.param("pointer", id="pointer"),
    ),
)


@pytest.fixture
def backend_type() -> str:
    return "git"


@pytest.fixture
def world_aliases() -> tuple[str, ...]:
    return ("live", "staging")


GROWN = RECORDED.append_column("c", pa.array([1.5, 2.5], pa.float64()))


def replace_t(world: SimpleNamespace, table: pa.Table) -> None:
    world.con.drop_table("t", force=True)
    world.con.create_table("t", table.to_pandas())


def commit_count(catalog: Catalog) -> int:
    return len(list(catalog.repo.iter_commits()))


def reopen(world: SimpleNamespace) -> Catalog:
    return Catalog.from_kwargs(path=world.catalog_path, init=False)


def targets(world: SimpleNamespace, *aliases: str) -> tuple[str, ...]:
    catalog = reopen(world)
    return tuple(alias_target_hash(catalog, alias) for alias in aliases)


def rebase(runner: CliRunner, world: SimpleNamespace, *args: str) -> Result:
    return runner.invoke(
        cli, ["--path", world.catalog_path, "rebase", *(args or (world.name,))]
    )


def rebase_old(world: SimpleNamespace, **kwargs: object) -> rebase_module.RebaseResult:
    return rebase_entry(world.catalog.get_catalog_entry(world.name), **kwargs)


def other_entry(world: SimpleNamespace) -> str:
    t = world.con.table("t")
    return world.catalog.add(t.filter(t.a > 0)).name


@ALL_BACKENDS
def test_a_grown_source_rebases_to_a_new_entry(
    runner: CliRunner, world: SimpleNamespace, tmp_path: Path
) -> None:
    replace_t(world, GROWN)

    result = rebase(runner, world, world.name, "--cache-dir", str(tmp_path / "c"))
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    assert "DatabaseTable t: changed" in result.stderr
    assert f"Rebased {world.name} -> {new}" in result.stderr
    catalog = reopen(world)
    assert set(catalog.list()) == {world.name, new}
    assert catalog.get_catalog_entry(new).columns == ("a", "b", "c")
    assert catalog.get_catalog_entry(world.name).columns == ("a", "b")
    assert targets(world, "live", "staging") == (new, new)


@ALL_BACKENDS
def test_no_drift_prints_the_same_hash_and_commits_nothing(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    commits = commit_count(world.catalog)

    result = rebase(runner, world, world.name, "-a", "v2")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{world.name}\n"
    assert "no drift" in result.stderr
    assert "Alias 'v2' not added" in result.stderr
    assert reopen(world).list() == [world.name]
    assert "v2" not in reopen(world).list_aliases()
    assert commit_count(reopen(world)) == commits


def test_only_alias_narrows_and_alias_adds(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, GROWN)

    result = rebase(runner, world, "live", "--only-alias", "live", "-a", "v2")
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    assert targets(world, "live", "v2", "staging") == (new, new, world.name)


def test_an_alias_already_on_the_entry_moves_and_is_reported(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, GROWN)

    result = rebase(runner, world, world.name, "--only-alias", "staging", "-a", "live")
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    assert f"Moved alias 'live' -> {new}" in result.stderr
    assert f"Moved alias 'staging' -> {new}" in result.stderr
    assert targets(world, "live", "staging") == (new, new)


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
    assert commit_count(reopen(world)) == commits
    attempted = rebase_old(world)
    assert (attempted.status, attempted.unprobed) == (RebaseStatus.ATTEMPTED, ("t",))


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


def test_a_python_minor_mismatch_is_overridable(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = world.catalog.get_catalog_entry(world.name)
    assert recorded_python_minor(entry) == tuple(sys.version_info[:2])
    replace_t(world, GROWN)
    monkeypatch.setattr(rebase_module, "recorded_python_minor", lambda _: (3, 0))

    result = rebase(runner, world, world.name, "--ignore-venv-mismatch")
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() != world.name


@pytest.mark.parametrize(
    "drift_t, headline, op_name",
    (
        pytest.param(
            lambda w: replace_t(w, pa.table({"b": ["x", "y"]})),
            "conflict: Field cannot be rebuilt; changed sources in entry: DatabaseTable t",
            "Field",
            id="dropped-column",
        ),
        pytest.param(
            lambda w: w.con.drop_table("t"),
            "conflict: DatabaseTable t is gone",
            None,
            id="gone-table",
        ),
    ),
)
def test_a_conflict_names_the_op_and_source_with_check_sources_schemas(
    runner: CliRunner,
    world: SimpleNamespace,
    drift_t: Callable,
    headline: str,
    op_name: str | None,
) -> None:
    drift_t(world)
    checked = runner.invoke(
        cli, ["--path", world.catalog_path, "check-sources", world.name]
    )
    assert checked.exit_code == 3, checked.output
    pair = [line for line in checked.stdout.splitlines() if line.startswith("    ")]

    result = rebase(runner, world, world.name, "-a", "v2")
    assert result.exit_code == 4, result.output
    assert f"{world.name}: {headline}" in result.stderr
    assert pair and all(line in result.stderr.splitlines() for line in pair)
    assert "could not rebuild" not in result.stderr
    assert "Alias 'v2' not added" in result.stderr
    if op_name is not None:
        # Named once, by the headline; the cause follows it.
        assert result.stderr.count(op_name) == 1
        assert "XorqTypeError: Column 'a' is not found" in result.stderr
    conflicted = rebase_old(world)
    assert conflicted.status == RebaseStatus.CONFLICT
    assert conflicted.new_entry.name == world.name
    assert conflicted.conflict.op_name == op_name
    assert [r.leaf.name for r in conflicted.conflict.sources] == ["t"]


def test_conflict_status_requires_conflict(world: SimpleNamespace) -> None:
    entry = world.catalog.get_catalog_entry(world.name)
    record = drift.read_record(entry)
    conflict = RebaseConflict("gone", ())
    with pytest.raises(ValueError, match="conflict must be set"):
        RebaseResult(RebaseStatus.CONFLICT, entry, entry, (), record)
    with pytest.raises(ValueError, match="conflict must be set"):
        RebaseResult(RebaseStatus.NOOP, entry, entry, (), record, conflict=conflict)


def test_unmatched_live_key_is_refused(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)

    def unmatched(*_: object) -> None:
        raise UnmatchedSourceError("DatabaseTable", LookupError("t matched no source"))

    monkeypatch.setattr(rebase_module, "refresh_schemas", unmatched)
    result = rebase(runner, world)
    assert result.exit_code == 1, result.output
    assert "t matched no source" in result.stderr
    assert "conflict:" not in result.stderr


Setup = tuple[str, tuple[str, ...], tuple[Path, ...]]
RUNNING = ".".join(map(str, sys.version_info[:2]))
UNREADABLE = "{name} is unreadable: ValueError: corrupt"
NO_WHEEL = "{name} carries no wheel for the rebased entry"
NO_REQUIREMENTS = "{name} carries no requirements.txt for the rebased entry"


# Refusals. Each case sets `w` up (`w.monkeypatch` is the test's) and returns
# the entry to rebase, the extra CLI args, and files that must stay gone; a
# message can name them as `{gone[0]}`.
def refuse_unknown_alias(w: SimpleNamespace) -> Setup:
    replace_t(w, GROWN)
    return w.name, ("--only-alias", "nope"), ()


def refuse_pinned(w: SimpleNamespace) -> Setup:
    path = w.tmp_path / "t.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    t = deferred_read_parquet(path, xo.connect(), table_name="t")
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=w.tmp_path)
    expr = pin_cache(t.filter(t.a > 1).cache(cache=cache), ensure_materialized=True)
    return w.catalog.add(expr).name, (), ()


def refuse_python_minor(w: SimpleNamespace) -> Setup:
    replace_t(w, GROWN)
    w.monkeypatch.setattr(rebase_module, "recorded_python_minor", lambda _: (3, 0))
    return w.name, (), ()


def refuse_some_unprobed(w: SimpleNamespace) -> Setup:
    u = w.con.create_table("u", RECORDED.to_pandas())
    w.monkeypatch.setattr(drift, "is_checkable", lambda leaf, record: leaf.name != "u")
    return w.catalog.add(w.con.table("t").union(u)).name, (), ()


def refuse_unprobed_new_hash(w: SimpleNamespace) -> Setup:
    """No source probed, yet the re-derivation hashes anew: nothing to follow."""
    w.monkeypatch.setattr(drift, "is_checkable", lambda leaf, record: False)
    w.monkeypatch.setattr(ExprDumper, "expr_hash", property(lambda _: "0" * 12))
    return w.name, (), ()


def refuse_unreadable(target: str) -> Callable:
    def setup(w: SimpleNamespace) -> Setup:
        replace_t(w, GROWN)

        def unreadable(*_: object) -> None:
            raise ValueError("corrupt")

        w.monkeypatch.setattr(rebase_module, target, unreadable)
        return w.name, (), ()

    return setup


def rewrite_archive(w: SimpleNamespace, rewrite: Callable) -> None:
    """Rewrite the entry's zip, each member through ``rewrite(name, bytes)``.

    A member ``rewrite`` maps to ``None`` is dropped.
    """
    path = w.catalog.get_catalog_entry(w.name).catalog_path
    with zipfile.ZipFile(path) as zf:
        members = {info.filename: zf.read(info) for info in zf.infolist()}
    with zipfile.ZipFile(path, "w") as zf:
        for member, byts in members.items():
            if (byts := rewrite(Path(member).name, byts)) is not None:
                zf.writestr(member, byts)


def refuse_without(dropped: str, drift: bool = True) -> Callable:
    """An archive without ``dropped``; refused before the sweep, drift or not."""

    def setup(w: SimpleNamespace) -> Setup:
        if drift:
            replace_t(w, GROWN)
        rewrite_archive(w, lambda name, byts: None if name.endswith(dropped) else byts)
        return w.name, (), ()

    return setup


def refuse_corrupt_metadata(w: SimpleNamespace) -> Setup:
    replace_t(w, GROWN)
    rewrite_archive(
        w,
        lambda name, byts: b"{corrupt" if name == DumpFiles.build_metadata else byts,
    )
    return w.name, (), ()


def refuse_deleted_db(w: SimpleNamespace) -> Setup:
    w.con.disconnect()
    w.db_path.unlink()
    return w.name, (), (w.db_path,)


def refuse_unprobed_db(w: SimpleNamespace) -> Setup:
    """A parquet read is probed without its sqlite profile; the load isn't."""
    path = w.tmp_path / "t.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    db = w.tmp_path / "ingest.sqlite"
    con = SqliteBackend().connect(str(db))
    read = deferred_read_parquet(path, con, table_name="ingested")
    name = w.catalog.add(read, relocate_reads=False).name
    con.disconnect()
    db.unlink()
    GROWN.to_pandas().to_parquet(path, index=False)
    return name, (), (db,)


def refuse_dropped_column(w: SimpleNamespace) -> Setup:
    replace_t(w, pa.table({"b": ["x", "y"]}))
    return w.name, (), ()


def refuse_dropped_table(w: SimpleNamespace) -> Setup:
    w.con.drop_table("t")
    return w.name, (), ()


def refuse_beside_unreachable(t_drift: Callable) -> Callable:
    """``t_drift`` on `t`, beside a second source whose database is gone."""

    def setup(w: SimpleNamespace) -> Setup:
        other_db = w.tmp_path / "other.sqlite"
        other = SqliteBackend().connect(str(other_db))
        other.create_table("u", RECORDED.to_pandas())
        u = other.table("u").into_backend(w.con)
        name = w.catalog.add(w.con.table("t").union(u)).name
        other.disconnect()
        other_db.unlink()
        t_drift(w)
        return name, (), (other_db,)

    return setup


def test_gone_table_also_names_unreachable_source(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    name, _, _ = refuse_beside_unreachable(lambda w: w.con.drop_table("t"))(world)

    result = rebase(runner, world, name)
    assert result.exit_code == 4, result.output
    assert "DatabaseTable t is gone" in result.stderr
    assert "u is unreachable" in result.stderr


# `(setup, exit_code, message)`: each way a rebase ends without writing, short
# of a no-op, and what stderr says.
REFUSALS = (
    pytest.param(refuse_unknown_alias, 1, "no alias nope", id="unknown-alias"),
    pytest.param(refuse_pinned, 1, "xorq catalog unpin {name}", id="pinned"),
    pytest.param(
        refuse_python_minor,
        1,
        f"built on Python 3.0, this is {RUNNING}; pass --ignore-venv-mismatch",
        id="python-minor",
    ),
    pytest.param(
        refuse_some_unprobed,
        1,
        "u cannot be probed without writing to it",
        id="some-unprobed",
    ),
    pytest.param(
        refuse_unreadable("recorded_python_minor"),
        2,
        UNREADABLE,
        id="unreadable-metadata",
    ),
    pytest.param(
        refuse_unreadable("harvest_entry_from_zip"),
        2,
        UNREADABLE,
        id="unreadable-bundle",
    ),
    pytest.param(
        refuse_unreadable("make_profile"), 2, UNREADABLE, id="unreadable-profile"
    ),
    pytest.param(
        refuse_corrupt_metadata,
        2,
        "{name} is unreadable: JSONDecodeError",
        id="corrupt-metadata",
    ),
    pytest.param(refuse_without(".whl"), 2, NO_WHEEL, id="no-wheel"),
    pytest.param(
        refuse_without(DumpFiles.requirements),
        2,
        NO_REQUIREMENTS,
        id="no-requirements",
    ),
    pytest.param(
        refuse_without(".whl", drift=False), 2, NO_WHEEL, id="no-wheel-no-drift"
    ),
    pytest.param(refuse_deleted_db, 2, "unreachable", id="deleted-db"),
    pytest.param(
        refuse_unprobed_db, 2, "database {gone[0]} does not exist", id="unprobed-db"
    ),
    pytest.param(
        refuse_unprobed_new_hash,
        1,
        "no source could be probed (t)",
        id="unprobed-new-hash",
    ),
    pytest.param(refuse_dropped_column, 4, "cannot be rebuilt", id="column"),
    pytest.param(refuse_dropped_table, 4, "table-missing", id="table"),
    pytest.param(
        refuse_beside_unreachable(lambda w: w.con.drop_table("t")),
        4,
        "table-missing",
        id="gone-outranks-unreachable",
    ),
    pytest.param(
        refuse_beside_unreachable(lambda w: replace_t(w, GROWN)),
        2,
        "unreachable",
        id="unreachable-outranks-changed",
    ),
)


@pytest.mark.parametrize("setup, exit_code, message", REFUSALS)
def test_a_refused_rebase_writes_nothing(
    runner: CliRunner,
    world: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable,
    exit_code: int,
    message: str,
) -> None:
    world.monkeypatch = monkeypatch
    name, args, gone = setup(world)
    entries, commits = reopen(world).list(), commit_count(world.catalog)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    result = rebase(runner, world, name, *args)
    assert result.exit_code == exit_code, result.output
    assert message.format(name=name, gone=gone) in result.stderr
    assert result.stdout == ""
    assert reopen(world).list() == entries
    assert commit_count(reopen(world)) == commits
    assert not any(path.exists() for path in gone)
    assert not any(cwd.iterdir())


# Alias moves and their rollback.
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


def pull_then(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[Catalog], object]
) -> None:
    """Stand in for a sync whose pull applies ``change`` to the catalog."""

    @contextmanager
    def pulling(self: Catalog, sync: bool):
        if sync:
            change(self)
        yield

    monkeypatch.setattr(Catalog, "maybe_synchronizing", pulling)


@ALL_BACKENDS
@pytest.mark.parametrize(
    "after_write",
    (pytest.param(False, id="raised"), pytest.param(True, id="committing")),
)
def test_a_failed_alias_move_rolls_back_the_new_entry(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    after_write: bool,
) -> None:
    replace_t(world, GROWN)
    fail_nth_alias_move(monkeypatch, 2, after_write=after_write)
    result = rebase(runner, world)
    monkeypatch.undo()

    assert result.exit_code == 1, result.output
    assert "alias move failed" in result.stderr
    assert result.stdout == ""
    assert reopen(world).list() == [world.name]
    assert targets(world, "live", "staging") == (world.name, world.name)


@ALL_BACKENDS
@pytest.mark.parametrize(
    "via",
    (
        pytest.param("earlier-rebase", id="earlier-rebase"),
        pytest.param("pull", id="pull"),
    ),
)
def test_a_rollback_keeps_an_entry_it_did_not_add(
    world: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    via: str,
) -> None:
    replace_t(world, GROWN)
    earlier = rebase_old(world).new_entry
    catalog = reopen(world)
    if via == "pull":
        archive = Path(shutil.copy(earlier.catalog_path, tmp_path))
        catalog.remove(earlier.name)
    for alias in ("live", "staging"):
        catalog.add_alias(world.name, alias)
    if via == "pull":
        pull_then(monkeypatch, lambda c: c.add(archive, sync=False))

    fail_nth_alias_move(monkeypatch, 2)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_old(world)
    monkeypatch.undo()

    assert set(reopen(world).list()) == {world.name, earlier.name}
    assert targets(world, "live", "staging") == (world.name, world.name)


@pytest.mark.parametrize(
    "pulled",
    (pytest.param(False, id="already-there"), pytest.param(True, id="pulled")),
)
def test_a_rollback_restores_an_extra_alias_where_it_was(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, pulled: bool
) -> None:
    """`-a fresh` lands through `catalog.add`; the `live` move after it fails."""
    other = other_entry(world)
    replace_t(world, GROWN)
    if pulled:
        pull_then(monkeypatch, lambda c: c.add_alias(other, "fresh", sync=False))
    else:
        world.catalog.add_alias(other, "fresh")
    # The pull's own `add_alias` is a move too.
    fail_nth_alias_move(monkeypatch, 2 if pulled else 1)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_old(world, alias="fresh")
    monkeypatch.undo()

    assert set(reopen(world).list()) == {world.name, other}
    assert targets(world, "fresh", "live", "staging") == (
        other,
        world.name,
        world.name,
    )


def test_a_rollback_restores_an_alias_already_on_the_entry(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    # `-a live` lands through `catalog.add`; the `staging` move after it fails.
    fail_nth_alias_move(monkeypatch, 1)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_old(world, alias="live", move_aliases=("staging",))
    monkeypatch.undo()

    assert reopen(world).list() == [world.name]
    assert targets(world, "live", "staging") == (world.name, world.name)


@pytest.mark.parametrize(
    "pulled, args, live",
    (
        pytest.param("moved", (), "other", id="moved"),
        pytest.param("removed", (), None, id="removed"),
        pytest.param("moved", ("-a", "live"), "new", id="moved-but-requested"),
    ),
)
def test_an_alias_the_pull_took_off_the_old_entry_stays_put(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    pulled: str,
    args: tuple[str, ...],
    live: str | None,
) -> None:
    """Unless `-a` asks for it on the new entry."""
    other = other_entry(world)
    replace_t(world, GROWN)
    if pulled == "moved":
        pull_then(monkeypatch, lambda c: c.add_alias(other, "live", sync=False))
    else:
        pull_then(monkeypatch, lambda c: c.remove_alias("live", sync=False))

    result = rebase(runner, world, world.name, *args)
    monkeypatch.undo()
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    skipped = f"Alias 'live' not moved: no longer on {world.name}"
    assert (skipped in result.stderr) == (live != "new")
    assert f"Moved alias 'staging' -> {new}" in result.stderr
    taken = f"Moved alias 'live' from {other} -> {new}"
    assert (taken in result.stderr) == (live == "new")
    assert f"Moved alias 'live' -> {new}" not in result.stderr
    catalog = reopen(world)
    assert alias_target_hash(catalog, "staging") == new
    if live is None:
        assert "live" not in catalog.list_aliases()
    else:
        assert alias_target_hash(catalog, "live") == {"other": other, "new": new}[live]


@pytest.mark.parametrize(
    "pulled",
    (pytest.param(False, id="already-there"), pytest.param(True, id="pulled")),
)
def test_an_alias_taken_off_another_entry_is_reported(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    pulled: bool,
) -> None:
    other = other_entry(world)
    replace_t(world, GROWN)
    if pulled:
        pull_then(monkeypatch, lambda c: c.add_alias(other, "fresh", sync=False))
    else:
        world.catalog.add_alias(other, "fresh")

    result = rebase(runner, world, world.name, "-a", "fresh")
    monkeypatch.undo()
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    assert f"Moved alias 'fresh' from {other} -> {new}" in result.stderr
    assert f"Moved alias 'fresh' -> {new}" not in result.stderr
    assert alias_target_hash(reopen(world), "fresh") == new


def test_a_new_alias_is_not_reported_as_taken(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, GROWN)
    result = rebase(runner, world, world.name, "-a", "fresh")
    assert result.exit_code == 0, result.output
    assert "'fresh'" not in result.stderr


def test_a_failed_rollback_surfaces_the_error_that_caused_it(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    fail_nth_alias_move(monkeypatch, 2)

    def failing_remove(self: Catalog, name: str, sync: bool = True) -> None:
        raise OSError("rollback failed")

    monkeypatch.setattr(Catalog, "remove", failing_remove)
    result = rebase(runner, world)
    assert result.exit_code == 1
    assert "alias move failed" in result.stderr


@pytest.mark.parametrize("backend_type", ("pointer",))
def test_an_entry_whose_archive_is_not_local_is_fetched_first(
    world: SimpleNamespace,
) -> None:
    # A fresh clone of a pointer catalog has the pointer but not the archive.
    entry = world.catalog.get_catalog_entry(world.name)
    Path(entry.catalog_path).unlink()
    assert not entry.is_content_local

    result = rebase_old(world)

    assert result.status == RebaseStatus.NOOP
    assert entry.is_content_local


@pytest.mark.parametrize("backend_type", ("pointer",))
def test_an_archive_that_cannot_be_fetched_is_refused_as_retryable(
    world: SimpleNamespace,
) -> None:
    backend = world.catalog.backend
    entry = world.catalog.get_catalog_entry(world.name)
    Path(entry.catalog_path).unlink()
    shutil.rmtree(backend.content_store.directory)
    shutil.rmtree(backend.cache.cache_dir)

    with pytest.raises(RebaseError, match="could not be fetched") as exc_info:
        rebase_old(world)

    assert exc_info.value.exit_code == RebaseExit.UNREACHABLE


# `--json` (#2326).
ROOT_KEYS = {
    "state",
    "exit_code",
    "old_entry",
    "new_entry",
    "build_path",
    "failing_op",
    "drift",
    "aliases",
}
NO_ALIASES = {"moved": []}
FAILURE_STATES = {1: "refused", 2: "unreachable", 4: "conflict"}
# A usage error exits before any rebase, so it prints no document.
JSON_REFUSALS = tuple(p for p in REFUSALS if p.id != "both-alias-flags")


def rebase_json(runner: CliRunner, world: SimpleNamespace, *args: str) -> tuple:
    """``rebase --json``'s result and document, checked for what every one owes.

    stdout is exactly one document and nothing else, stderr is empty, and the
    document's exit code is the process's.
    """
    result = rebase(runner, world, *(args or (world.name,)), "--json")
    document = json.loads(result.stdout)
    assert result.stdout == json.dumps(document, indent=2) + "\n"
    assert result.stderr == ""
    assert document["exit_code"] == result.exit_code
    assert set(document) - {"error"} == ROOT_KEYS
    # Nothing on this branch keeps the build (#2325's `--no-add` will).
    assert document["build_path"] is None
    return result, document


def check_sources_entry(runner: CliRunner, world: SimpleNamespace, name: str) -> dict:
    checked = runner.invoke(
        cli, ["--path", world.catalog_path, "check-sources", name, "--json"]
    )
    return json.loads(checked.stdout)["entries"][name]


def json_noop(w: SimpleNamespace) -> tuple[str, ...]:
    return ("-a", "v2")


def json_unprobed(w: SimpleNamespace) -> tuple[str, ...]:
    w.monkeypatch.setattr(drift, "is_checkable", lambda leaf, record: False)
    return ()


def json_rebased(w: SimpleNamespace) -> tuple[str, ...]:
    w.other = other_entry(w)
    replace_t(w, GROWN)
    return ("-a", "fresh")


@pytest.mark.parametrize(
    "setup, state, drift_state, rebased",
    (
        pytest.param(json_noop, "noop", "equal", False, id="noop"),
        pytest.param(json_unprobed, "attempted", None, False, id="attempted"),
        pytest.param(json_rebased, "rebased", "changed", True, id="rebased"),
    ),
)
def test_json_reports_a_completed_rebase(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable,
    state: str,
    drift_state: str | None,
    rebased: bool,
) -> None:
    world.monkeypatch = monkeypatch
    args = setup(world)
    before = check_sources_entry(runner, world, world.name)

    result, document = rebase_json(runner, world, world.name, *args)
    assert (document["state"], result.exit_code) == (state, 0)
    assert document["old_entry"] == world.name
    assert document["failing_op"] is None
    assert "error" not in document
    # The sweep's own document, the one `check-sources --json` publishes.
    assert document["drift"] == before
    assert document["drift"]["state"] == drift_state
    if not rebased:
        # A no-op, proven or not, names the existing hash both times.
        assert document["new_entry"] == world.name
        assert document["aliases"] == NO_ALIASES
        assert alias_target_hash(reopen(world), "live") == world.name
        return
    new = document["new_entry"]
    assert new != world.name
    assert set(reopen(world).list()) == {world.name, world.other, new}
    assert document["aliases"] == {"moved": ["live", "staging"]}
    assert targets(world, "live", "staging", "fresh") == (new, new, new)


def test_json_reports_only_the_aliases_left_on_the_old_entry_after_the_pull(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = other_entry(world)
    replace_t(world, GROWN)
    pull_then(monkeypatch, lambda c: c.add_alias(other, "live", sync=False))

    _, document = rebase_json(runner, world, world.name)
    monkeypatch.undo()
    assert document["state"] == "rebased"
    assert document["aliases"] == {"moved": ["staging"]}


@pytest.mark.parametrize("setup, exit_code, message", JSON_REFUSALS)
def test_json_reports_a_rebase_that_wrote_nothing(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable,
    exit_code: int,
    message: str,
) -> None:
    world.monkeypatch = monkeypatch
    name, args, gone = setup(world)
    old_entry = reopen(world).get_catalog_entry(name, maybe_alias=True).name
    entries, commits = reopen(world).list(), commit_count(world.catalog)

    result, document = rebase_json(runner, world, name, *args)
    assert result.exit_code == exit_code
    assert document["state"] == FAILURE_STATES[exit_code]
    assert (document["old_entry"], document["new_entry"]) == (old_entry, None)
    assert document["aliases"] == NO_ALIASES
    if exit_code == 4:
        # The headline's facts are the drift document; `error` is the detail.
        assert document["drift"]["state"] in ("changed", "table-missing")
        assert document["error"]
    else:
        # A refusal returns no sweep to report.
        assert document["drift"] is None
        assert document["failing_op"] is None
        assert message.format(name=name, gone=gone) in document["error"]
    assert reopen(world).list() == entries
    assert commit_count(reopen(world)) == commits
    assert not any(path.exists() for path in gone)


@pytest.mark.parametrize(
    "drift_t, failing_op, error",
    (
        pytest.param(
            lambda w: replace_t(w, pa.table({"b": ["x", "y"]})),
            "Field",
            "XorqTypeError: Column 'a' is not found",
            id="dropped-column",
        ),
        pytest.param(lambda w: w.con.drop_table("t"), None, "t", id="gone-table"),
    ),
)
def test_json_conflict_embeds_the_check_sources_entry_document(
    runner: CliRunner,
    world: SimpleNamespace,
    drift_t: Callable,
    failing_op: str | None,
    error: str,
) -> None:
    drift_t(world)
    checked = check_sources_entry(runner, world, world.name)

    result, document = rebase_json(runner, world, world.name, "-a", "v2")
    assert (document["state"], result.exit_code) == ("conflict", 4)
    assert document["failing_op"] == failing_op
    assert document["drift"] == checked
    assert error in document["error"]
    assert "v2" not in reopen(world).list_aliases()


def test_json_keeps_the_local_path_warning_off_stderr(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """An external local read is dumped as recorded, without the relocate hint.

    pytest captures warnings itself, so the CLI's stderr is checked through a
    recorder: outside pytest the warning would print there.
    """
    path = world.tmp_path / "ext.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    con = SqliteBackend().connect(str(world.tmp_path / "ingest.sqlite"))
    read = deferred_read_parquet(path, con, table_name="ingested")
    name = world.catalog.add(read.filter(read.a > 1), relocate_reads=False).name
    GROWN.to_pandas().to_parquet(path, index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, document = rebase_json(runner, world, name)
    assert document["state"] == "rebased"
    assert not [w for w in caught if "local filesystem path" in str(w.message)]


def test_json_reports_a_failed_write_as_one_document(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that fails midway still ends in one whole document."""
    replace_t(world, GROWN)
    fail_nth_alias_move(monkeypatch, 2)
    result, document = rebase_json(runner, world, world.name)
    monkeypatch.undo()

    assert (document["state"], result.exit_code) == ("failed", 1)
    assert (document["old_entry"], document["new_entry"]) == (world.name, None)
    assert document["error"] == "RuntimeError: alias move failed"
    assert reopen(world).list() == [world.name]


def test_json_prints_nothing_until_the_rebase_ends(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    ended = []
    echoed = []
    real_rebase_entry = rebase_module.rebase_entry
    real_echo = click.echo

    def rebasing(*args: object, **kwargs: object) -> RebaseResult:
        result = real_rebase_entry(*args, **kwargs)
        ended.append(True)
        return result

    def echoing(*args: object, **kwargs: object) -> None:
        echoed.append(bool(ended))
        real_echo(*args, **kwargs)

    monkeypatch.setattr(rebase_module, "rebase_entry", rebasing)
    monkeypatch.setattr(click, "echo", echoing)
    _, document = rebase_json(runner, world)
    assert document["state"] == "rebased"
    assert echoed == [True]


@pytest.mark.parametrize(
    "args, exit_code",
    (
        pytest.param(("nope",), 1, id="unresolved-name"),
        pytest.param(("--bogus",), 2, id="usage-error"),
    ),
)
@pytest.mark.parametrize(
    "command",
    (
        pytest.param("rebase", id="rebase"),
        pytest.param("check-sources", id="check-sources"),
    ),
)
def test_json_never_started_prints_no_document_like_check_sources(
    runner: CliRunner,
    world: SimpleNamespace,
    command: str,
    args: tuple[str, ...],
    exit_code: int,
) -> None:
    """No entry resolved, so there is no rebase whose outcome to report."""
    result = runner.invoke(
        cli, ["--path", world.catalog_path, command, *args, "--json"]
    )
    assert result.exit_code == exit_code
    assert result.stdout == ""
    assert result.stderr
