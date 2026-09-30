"""`xorq catalog rebase` (#2323), over the sqlite world `test_drift` uses."""

from __future__ import annotations

import shutil
import sys
import zipfile
from collections.abc import Callable
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
from xorq.catalog.catalog import Catalog, CatalogAlias, CatalogEntry
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
from xorq.ibis_yaml.compiler import build_expr
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


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})
GROWN = RECORDED.append_column("c", pa.array([1.5, 2.5], pa.float64()))
# `a` renamed to `x`.
RENAMED = RECORDED.rename_columns(["x", "b"])


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
    # No alias moves unless asked.
    assert targets(world, "live", "staging") == (world.name, world.name)
    assert "alias" not in result.stderr.lower()


def test_move_aliases_moves_them_all(runner: CliRunner, world: SimpleNamespace) -> None:
    replace_t(world, GROWN)

    result = rebase(runner, world, world.name, "--move-aliases")
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    assert f"Moved alias 'live' -> {new}" in result.stderr
    assert f"Moved alias 'staging' -> {new}" in result.stderr
    assert targets(world, "live", "staging") == (new, new)


@ALL_BACKENDS
@pytest.mark.parametrize("moving", (("--move-aliases",), ("--only-alias", "live")))
def test_no_drift_prints_the_same_hash_and_commits_nothing(
    runner: CliRunner, world: SimpleNamespace, moving: tuple[str, ...]
) -> None:
    commits = commit_count(world.catalog)

    result = rebase(runner, world, world.name, "-a", "v2", *moving)
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{world.name}\n"
    assert "no drift" in result.stderr
    assert "Alias 'v2' not added" in result.stderr
    assert "Aliases not moved: nothing to rebase" in result.stderr
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


@pytest.mark.parametrize(
    "args",
    (
        pytest.param(("--move-aliases", "-a", "live"), id="move-aliases"),
        pytest.param(("--only-alias", "live", "-a", "live"), id="only-alias"),
        pytest.param(("--only-alias", "live", "--only-alias", "live"), id="repeated"),
    ),
)
def test_an_alias_named_twice_moves_once(
    runner: CliRunner, world: SimpleNamespace, args: tuple[str, ...]
) -> None:
    replace_t(world, GROWN)

    result = rebase(runner, world, world.name, *args)
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    assert result.stderr.count(f"Moved alias 'live' -> {new}") == 1
    assert targets(world, "live") == (new,)


def test_an_entry_with_only_unprobed_sources_is_not_loaded(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing can be refreshed, so nothing is done; not declared drift-free."""
    monkeypatch.setattr(drift, "is_checkable", lambda leaf, record: False)

    def no_load(*_: object, **__: object) -> None:
        raise AssertionError("an unprobed entry must not be loaded")

    monkeypatch.setattr(CatalogEntry, "load_expr", no_load)
    commits = commit_count(world.catalog)

    result = rebase(runner, world, world.name, "-a", "v2")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{world.name}\n"
    assert f"{world.name}: no source can be probed (t); nothing done" in result.stderr
    assert "no drift" not in result.stderr
    assert "Alias 'v2' not added: nothing to rebase" in result.stderr
    assert "Aliases not moved" not in result.stderr
    assert commit_count(reopen(world)) == commits
    unprobed = rebase_old(world)
    assert (unprobed.status, unprobed.unprobed) == (RebaseStatus.UNPROBED, ("t",))


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
    assert f"WARNING: {world.name} was built on Python 3.0" in result.stderr


def test_an_unrecorded_python_minor_is_overridable_with_a_warning(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    monkeypatch.setattr(rebase_module, "recorded_python_minor", lambda _: None)

    result = rebase(runner, world, world.name, "--ignore-venv-mismatch")
    assert result.exit_code == 0, result.output
    assert f"WARNING: {world.name} records no Python minor" in result.stderr


def test_rebase_entry_checks_its_alias_arguments(world: SimpleNamespace) -> None:
    entry = world.catalog.get_catalog_entry(world.name)
    # 0.4.5 took the names to move as `move_aliases`; they must not mean "all".
    with pytest.raises(TypeError, match="only_aliases"):
        rebase_entry(entry, move_aliases=("staging",))
    with pytest.raises(ValueError, match="mutually exclusive"):
        rebase_entry(entry, move_aliases=True, only_aliases=("live",))
    # An empty iterator names no alias.
    noop = rebase_entry(entry, move_aliases=True, only_aliases=iter(()))
    assert noop.status == RebaseStatus.NOOP


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


def refuse_elsewhere_alias_without_drift(w: SimpleNamespace) -> Setup:
    """With nothing to rebase, `--only-alias` must name one of the entry's."""
    w.catalog.add_alias(other_entry(w), "fresh")
    return w.name, ("--only-alias", "fresh"), ()


def refuse_taken_alias(w: SimpleNamespace) -> Setup:
    """`-a` names an alias on another entry: refused, not taken."""
    w.catalog.add_alias(other_entry(w), "fresh")
    replace_t(w, GROWN)
    return w.name, ("-a", "fresh"), ()


def refuse_both_alias_flags(w: SimpleNamespace) -> Setup:
    replace_t(w, GROWN)
    return w.name, ("--move-aliases", "--only-alias", "live"), ()


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


def refuse_no_python_minor(w: SimpleNamespace) -> Setup:
    replace_t(w, GROWN)
    w.monkeypatch.setattr(rebase_module, "recorded_python_minor", lambda _: None)
    return w.name, (), ()


def refuse_pull(w: SimpleNamespace) -> Setup:
    replace_t(w, GROWN)

    def failing_pull(self: Catalog) -> None:
        raise OSError("remote gone")

    w.monkeypatch.setattr(Catalog, "pull", failing_pull)
    return w.name, (), ()


def refuse_rename(table: pa.Table | None, *args: str) -> Callable:
    """``--rename args`` over `t` replaced by ``table`` (left as is for ``None``)."""

    def setup(w: SimpleNamespace) -> Setup:
        if table is not None:
            replace_t(w, table)
        return w.name, ("--rename", *args), ()

    return setup


def refuse_rename_unprobed(w: SimpleNamespace) -> Setup:
    w.monkeypatch.setattr(drift, "is_checkable", lambda leaf, record: False)
    return w.name, ("--rename", "t", "a", "x"), ()


def refuse_rename_beside_a_lost_column(w: SimpleNamespace) -> Setup:
    """`a` renamed to `x` and `b` dropped: the hint names `b` only."""
    t = w.con.table("t")
    name = w.catalog.add(t.filter(t.a > 1).select("b")).name
    replace_t(w, pa.table({"x": pa.array([1, 2], pa.int64())}))
    return name, ("--rename", "t", "a", "x"), ()


def refuse_rename_ambiguous(w: SimpleNamespace) -> Setup:
    """One table name `t` on two connections."""
    other = SqliteBackend().connect(str(w.tmp_path / "other.sqlite"))
    other.create_table("t", RECORDED.to_pandas())
    name = w.catalog.add(
        w.con.table("t").union(other.table("t").into_backend(w.con))
    ).name
    replace_t(w, RENAMED)
    return name, ("--rename", "t", "a", "x"), ()


def refuse_some_unprobed(w: SimpleNamespace) -> Setup:
    u = w.con.create_table("u", RECORDED.to_pandas())
    w.monkeypatch.setattr(drift, "is_checkable", lambda leaf, record: leaf.name != "u")
    return w.catalog.add(w.con.table("t").union(u)).name, (), ()


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


def refuse_dangling_profile(*args: str) -> Callable:
    """`t`'s profile gone from the record: unreadable, with ``args`` or not."""

    def setup(w: SimpleNamespace) -> Setup:
        replace_t(w, RENAMED)
        rewrite_archive(
            w, lambda name, byts: b"{}\n" if name == DumpFiles.profiles else byts
        )
        return w.name, args, ()

    return setup


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
        pytest.param(
            refuse_elsewhere_alias_without_drift,
            1,
            "no alias fresh",
            id="elsewhere-alias-no-drift",
        ),
        pytest.param(
            refuse_taken_alias,
            1,
            "alias 'fresh' points at",
            id="taken-alias",
        ),
        pytest.param(
            refuse_both_alias_flags,
            2,
            "--move-aliases and --only-alias are mutually exclusive",
            id="both-alias-flags",
        ),
        pytest.param(refuse_pinned, 1, "xorq catalog unpin {name}", id="pinned"),
        pytest.param(
            refuse_python_minor,
            1,
            f"built on Python 3.0, this is {RUNNING}; its UDFs may not load; "
            "pass --ignore-venv-mismatch",
            id="python-minor",
        ),
        pytest.param(
            refuse_no_python_minor,
            1,
            f"records no Python minor, this is {RUNNING}",
            id="no-python-minor",
        ),
        pytest.param(
            refuse_pull,
            2,
            "pull failed: OSError: remote gone; nothing written",
            id="pull",
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
        pytest.param(
            refuse_dangling_profile(),
            2,
            "{name}: t is unreadable: ValueError: node",
            id="dangling-profile",
        ),
        pytest.param(
            refuse_dangling_profile("--rename", "t", "a", "x"),
            2,
            "{name}: t is unreadable: ValueError: node",
            id="rename-dangling-profile",
        ),
        pytest.param(refuse_deleted_db, 2, "unreachable", id="deleted-db"),
        pytest.param(
            refuse_unprobed_db, 2, "database {gone[0]} does not exist", id="unprobed-db"
        ),
        pytest.param(
            refuse_rename(RENAMED, "nope", "a", "x"),
            1,
            "--rename names no source 'nope'; its sources: t",
            id="rename-no-source",
        ),
        pytest.param(
            refuse_rename_ambiguous,
            1,
            "--rename 't' names 2 sources",
            id="rename-ambiguous",
        ),
        pytest.param(
            refuse_rename(RENAMED, "t", "zz", "x"),
            1,
            "--rename t: 'zz' is not a recorded column",
            id="rename-not-recorded",
        ),
        pytest.param(
            refuse_rename_unprobed,
            1,
            "--rename needs a live schema, and no source can be probed (t)",
            id="rename-unprobed",
        ),
        pytest.param(
            refuse_rename(RENAMED, "t", "a", "q"),
            1,
            "--rename t: 'q' is not a live column",
            id="rename-not-live",
        ),
        pytest.param(
            refuse_rename(None, "t", "a", "b"),
            1,
            "--rename t: 'a' is still a live column, so nothing was renamed",
            id="rename-no-drift",
        ),
        pytest.param(
            refuse_rename(RENAMED, "t", "a", "x", "--rename", "t", "a", "b"),
            1,
            "--rename t: 'a' or 'b' is renamed twice",
            id="rename-twice",
        ),
        pytest.param(
            refuse_rename_beside_a_lost_column,
            4,
            "t: recorded columns gone: b; live columns new: -\nif a column was renamed",
            id="rename-hint-less-renames",
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
    monkeypatch.setattr(Catalog, "pull", lambda self: change(self))


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
    result = rebase(runner, world, world.name, "--move-aliases")
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
    if via == "pull":
        archive = Path(shutil.copy(earlier.catalog_path, tmp_path))
        # Before `pull_then`: `remove` syncs.
        reopen(world).remove(earlier.name)
        pull_then(monkeypatch, lambda c: c.add(archive, sync=False))

    # `-a fresh` lands on the kept entry through `catalog.add`, so only the
    # rollback can take it off; the `staging` move after `live`'s fails.
    fail_nth_alias_move(monkeypatch, 2)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_old(world, alias="fresh", move_aliases=True)
    monkeypatch.undo()

    catalog = reopen(world)
    assert set(catalog.list()) == {world.name, earlier.name}
    assert "fresh" not in catalog.list_aliases()
    assert targets(world, "live", "staging") == (world.name, world.name)


def test_a_rollback_removes_an_alias_it_registered(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-a fresh` lands through `catalog.add`; the `live` move after it fails."""
    replace_t(world, GROWN)
    fail_nth_alias_move(monkeypatch, 1)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_old(world, alias="fresh", move_aliases=True)
    monkeypatch.undo()

    catalog = reopen(world)
    assert catalog.list() == [world.name]
    assert "fresh" not in catalog.list_aliases()
    assert targets(world, "live", "staging") == (world.name, world.name)


def test_a_rollback_restores_an_alias_already_on_the_entry(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    # `-a live` lands through `catalog.add`; the `staging` move after it fails.
    fail_nth_alias_move(monkeypatch, 1)
    with pytest.raises(RuntimeError, match="alias move failed"):
        rebase_old(world, alias="live", only_aliases=("staging",))
    monkeypatch.undo()

    assert reopen(world).list() == [world.name]
    assert targets(world, "live", "staging") == (world.name, world.name)


@pytest.mark.parametrize(
    "pulled", (pytest.param("moved", id="moved"), pytest.param("removed", id="removed"))
)
def test_move_aliases_moves_those_on_the_old_entry_after_the_pull(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    pulled: str,
) -> None:
    other = other_entry(world)
    replace_t(world, GROWN)
    if pulled == "moved":
        pull_then(monkeypatch, lambda c: c.add_alias(other, "live", sync=False))
    else:
        pull_then(monkeypatch, lambda c: c.remove_alias("live", sync=False))

    result = rebase(runner, world, world.name, "--move-aliases")
    monkeypatch.undo()
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    assert f"Moved alias 'staging' -> {new}" in result.stderr
    assert "'live'" not in result.stderr
    catalog = reopen(world)
    assert alias_target_hash(catalog, "staging") == new
    if pulled == "moved":
        assert alias_target_hash(catalog, "live") == other
    else:
        assert "live" not in catalog.list_aliases()


@pytest.mark.parametrize(
    "entry, args, message",
    (
        pytest.param(
            None, ("--only-alias", "live"), "{name} has no alias live", id="only-alias"
        ),
        pytest.param(
            None, ("-a", "live"), "alias 'live' points at {other};", id="alias"
        ),
        # ENTRY `live` meant the old entry before the pull, not after it.
        pytest.param(
            "live",
            ("--move-aliases",),
            "alias 'live' points at {other} after the pull, not {name}",
            id="entry",
        ),
    ),
)
def test_a_named_alias_the_pull_moved_away_is_refused(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    entry: str | None,
    args: tuple[str, ...],
    message: str,
) -> None:
    other = other_entry(world)
    replace_t(world, GROWN)
    pull_then(monkeypatch, lambda c: c.add_alias(other, "live", sync=False))

    result = rebase(runner, world, entry or world.name, *args)
    monkeypatch.undo()
    assert result.exit_code == 1, result.output
    assert message.format(name=world.name, other=other) in result.stderr
    assert result.stdout == ""
    catalog = reopen(world)
    assert set(catalog.list()) == {world.name, other}
    assert targets(world, "live", "staging") == (other, world.name)


@pytest.mark.parametrize(
    "alias, args", (("fresh", ("-a",)), ("live", ("--only-alias",)))
)
def test_a_rerun_finds_the_alias_already_on_the_new_entry(
    runner: CliRunner, world: SimpleNamespace, alias: str, args: tuple[str, ...]
) -> None:
    """As after a failed push, or a pull of an earlier rebase of the entry."""
    replace_t(world, GROWN)
    new = rebase_old(world, alias="fresh", only_aliases=("live",)).new_entry.name

    result = rebase(runner, world, world.name, *args, alias)
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == new
    assert "Moved alias" not in result.stderr
    assert f"Alias {alias!r} already on {new}" in result.stderr
    assert targets(world, alias) == (new,)


def test_a_new_alias_is_reported_as_added(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, GROWN)
    result = rebase(runner, world, world.name, "-a", "fresh")
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    # One line, the one `add-alias` prints.
    assert [line for line in result.stderr.splitlines() if "'fresh'" in line] == [
        f"Added alias 'fresh' -> {new}"
    ]


def test_a_failed_rollback_surfaces_the_error_that_caused_it(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    fail_nth_alias_move(monkeypatch, 2)

    def failing_remove(self: Catalog, name: str, sync: bool = True) -> None:
        raise OSError("rollback failed")

    monkeypatch.setattr(Catalog, "remove", failing_remove)
    result = rebase(runner, world, world.name, "--move-aliases")
    assert result.exit_code == 1
    assert "RuntimeError: alias move failed; rollback failed: OSError" in result.stderr
    # It names what it left: the new entry, with the alias that moved.
    assert "cataloged, aliases [live] -> " in result.stderr


# Outcomes the output states: the output schema, a kept entry, the push.
@pytest.mark.parametrize(
    "build, table, output",
    (
        pytest.param(
            lambda t: t.filter(t.a >= 1), GROWN, "Output: c float64 added", id="added"
        ),
        pytest.param(
            lambda t: t.mutate(a2=t.a * 2),
            pa.table({"a": [1.5, 2.5], "b": ["x", "y"]}),
            "Output: a int64 -> float64, a2 int64 -> float64",
            id="retyped",
        ),
        pytest.param(
            lambda t: t.select("a"), GROWN, "Output: unchanged", id="unchanged"
        ),
    ),
)
def test_the_output_schema_change_is_reported(
    runner: CliRunner,
    world: SimpleNamespace,
    build: Callable,
    table: pa.Table,
    output: str,
) -> None:
    name = world.catalog.add(build(world.con.table("t"))).name
    replace_t(world, table)

    result = rebase(runner, world, name)
    assert result.exit_code == 0, result.output
    assert f"{output}\n" in result.stderr
    new = reopen(world).get_catalog_entry(result.stdout.strip())
    changes = rebase_module.output_changes(
        world.catalog.get_catalog_entry(name).load_expr().schema(),
        new.load_expr().schema(),
    )
    assert output == f"Output: {', '.join(map(str, changes)) or 'unchanged'}"


def test_an_entry_already_cataloged_is_kept_and_said_so(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, GROWN)
    first = rebase_old(world)
    assert first.created

    result = rebase(runner, world, world.name, "--move-aliases")
    assert result.exit_code == 0, result.output
    new = first.new_entry.name
    assert result.stdout == f"{new}\n"
    assert (
        f"Rebased {world.name} -> {new} (already cataloged; existing archive kept)"
        in result.stderr
    )
    assert targets(world, "live", "staging") == (new, new)
    assert not rebase_old(world).created


def test_a_failed_push_exits_5_with_the_local_rebase_reported(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)

    def failing_push(self: Catalog) -> None:
        raise OSError("rejected")

    monkeypatch.setattr(Catalog, "push", failing_push)
    result = rebase(runner, world, world.name, "--move-aliases")
    monkeypatch.undo()
    assert result.exit_code == 5, result.output
    new = result.stdout.strip()
    assert f"Rebased {world.name} -> {new}" in result.stderr
    assert (
        f"rebased to {new} locally; push failed: OSError: rejected; run "
        "`xorq catalog push`" in result.stderr
    )
    # Committed locally, aliases moved; nothing rolled back.
    assert set(reopen(world).list()) == {world.name, new}
    assert targets(world, "live", "staging") == (new, new)


def test_pushed_is_false_without_a_remote(world: SimpleNamespace) -> None:
    replace_t(world, GROWN)
    result = rebase_old(world)
    assert (result.status, result.pushed) == (RebaseStatus.REBASED, False)


# `--rename`: a renamed column followed under its recorded name.
def test_a_renamed_column_is_followed_under_its_recorded_name(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, RENAMED)

    result = rebase(runner, world, world.name, "--rename", "t", "a", "x")
    assert result.exit_code == 0, result.output
    assert "Renamed t: a <- x\nOutput: unchanged\n" in result.stderr
    new = reopen(world).get_catalog_entry(result.stdout.strip())
    assert new.columns == ("a", "b")
    assert list(new.load_expr().execute()["a"]) == [2]
    # The API takes the `Rename`s a result hands back.
    renames = (rebase_module.Rename("t", "a", "x"),)
    assert rebase_old(world, renames=renames).renames == renames


def test_a_source_recorded_at_two_schemas_is_renamed_under_both(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """`t` bound before and after it grew: one source, two records."""
    before = world.con.table("t")
    replace_t(world, GROWN)
    after = world.con.table("t")
    name = world.catalog.add(before.select("a", "b").union(after.select("a", "b"))).name
    replace_t(world, RENAMED)

    result = rebase(runner, world, name, "--rename", "t", "a", "x")
    assert result.exit_code == 0, result.output
    assert "Renamed t: a <- x\nOutput: unchanged\n" in result.stderr


def test_a_conflict_over_a_lost_column_lists_what_changed(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, RENAMED)

    result = rebase(runner, world)
    assert result.exit_code == 4, result.output
    assert (
        "t: recorded columns gone: a; live columns new: x\n"
        "if a column was renamed, pass --rename <source> <old> <new>"
    ) in result.stderr
