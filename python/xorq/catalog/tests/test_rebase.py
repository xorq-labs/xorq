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
from xorq.catalog.rebase import rebase_entry, recorded_python_minor
from xorq.catalog.tests.conftest import (
    TEST_WHEEL_NAME,
    _annex_available,
    alias_target_hash,
)
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.expr.relations import pin_cache
from xorq.ibis_yaml.compiler import ExprDumper, build_expr
from xorq.ibis_yaml.enums import DumpFiles


# Most tests exercise rebase's own logic, which doesn't vary with storage; the
# ones that add or remove entries run on every backend. Aliases are git
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


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})
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
        tmp_path=tmp_path,
    )


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
    assert commit_count(reopen(world)) == commits


def test_only_alias_narrows_and_alias_adds(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, GROWN)

    result = rebase(runner, world, "live", "--only-alias", "live", "-a", "v2")
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    assert targets(world, "live", "v2", "staging") == (new, new, world.name)


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


Setup = tuple[str, tuple[str, ...], tuple[Path, ...]]
UNREADABLE = "{name} is unreadable: ValueError: corrupt"
NO_BUNDLE = "for the rebased entry"


# Refusals. Each case sets `w` up (`w.monkeypatch` is the test's) and returns
# the entry to rebase, the extra CLI args, and files that must stay gone.
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


def refuse_without(dropped: str, drift: bool = True) -> Callable:
    """An archive without ``dropped``; refused before the sweep, drift or not."""

    def setup(w: SimpleNamespace) -> Setup:
        if drift:
            replace_t(w, GROWN)
        members = rebase_module.bundle_members

        def without(catalog_entry: object) -> tuple[str, ...]:
            return tuple(
                name for name in members(catalog_entry) if not name.endswith(dropped)
            )

        w.monkeypatch.setattr(rebase_module, "bundle_members", without)
        return w.name, (), ()

    return setup


def refuse_corrupt_metadata(w: SimpleNamespace) -> Setup:
    replace_t(w, GROWN)
    path = w.catalog.get_catalog_entry(w.name).catalog_path
    with zipfile.ZipFile(path) as zf:
        members = {info.filename: zf.read(info) for info in zf.infolist()}
    members = {
        member: b"{corrupt" if Path(member).name == DumpFiles.build_metadata else byts
        for member, byts in members.items()
    }
    with zipfile.ZipFile(path, "w") as zf:
        for member, byts in members.items():
            zf.writestr(member, byts)
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


@pytest.mark.parametrize(
    "setup, exit_code, message",
    (
        pytest.param(refuse_unknown_alias, 1, "no alias nope", id="unknown-alias"),
        pytest.param(refuse_pinned, 1, "xorq catalog unpin {name}", id="pinned"),
        pytest.param(refuse_python_minor, 1, "built on Python 3.0", id="python-minor"),
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
        pytest.param(refuse_without(".whl"), 2, NO_BUNDLE, id="no-wheel"),
        pytest.param(
            refuse_without(DumpFiles.requirements), 2, NO_BUNDLE, id="no-requirements"
        ),
        pytest.param(
            refuse_without(".whl", drift=False), 2, NO_BUNDLE, id="no-wheel-no-drift"
        ),
        pytest.param(refuse_deleted_db, 2, "unreachable", id="deleted-db"),
        pytest.param(refuse_unprobed_db, 2, "does not exist", id="unprobed-db"),
        pytest.param(
            refuse_unprobed_new_hash,
            4,
            "no source could be probed (t)",
            id="unprobed-new-hash",
        ),
        pytest.param(refuse_dropped_column, 4, "could not rebuild", id="column"),
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
    ),
)
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
    assert message.format(name=name) in result.stderr
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
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, after_write: bool
) -> None:
    replace_t(world, GROWN)
    fail_nth_alias_move(monkeypatch, 2, after_write=after_write)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_old(world)
    monkeypatch.undo()

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


def test_a_rollback_restores_an_extra_alias_where_the_pull_left_it(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = other_entry(world)
    replace_t(world, GROWN)
    pull_then(monkeypatch, lambda c: c.add_alias(other, "fresh", sync=False))
    # The first move is `-a fresh` through `catalog.add`; `live`, second, fails.
    fail_nth_alias_move(monkeypatch, 2)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_old(world, alias="fresh")
    monkeypatch.undo()

    assert targets(world, "fresh", "live") == (other, world.name)


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
    catalog = reopen(world)
    assert alias_target_hash(catalog, "staging") == new
    if live is None:
        assert "live" not in catalog.list_aliases()
    else:
        assert alias_target_hash(catalog, "live") == {"other": other, "new": new}[live]


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
