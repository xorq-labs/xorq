"""`xorq catalog rebase` (#2323), over the sqlite world `test_drift` uses."""

from __future__ import annotations

import json
import re
import shutil
import sys
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import click
import pyarrow as pa
import pytest
from attr import evolve
from click.testing import CliRunner, Result
from git import Repo

import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.caching import ParquetCache
from xorq.catalog import drift
from xorq.catalog import rebase as rebase_module
from xorq.catalog.aliases import AliasPlan
from xorq.catalog.catalog import Catalog, CatalogAlias, CatalogPullError
from xorq.catalog.cli import cli
from xorq.catalog.constants import MAIN_BRANCH
from xorq.catalog.enums import RebaseExit, RebaseFailure, RebaseStatus
from xorq.catalog.exceptions import RebaseError
from xorq.catalog.rebase import rebase_entry, recorded_python_minor
from xorq.catalog.tests.conftest import (
    RECORDED,
    TEST_WHEEL_NAME,
    _annex_available,
    add_expr_entry,
    alias_target_hash,
    fail_nth_alias_move,
    pull_then,
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
    # No alias moves unless asked, and what stays behind is said.
    assert targets(world, "live", "staging") == (world.name, world.name)
    assert "Moved alias" not in result.stderr
    assert (
        f"Aliases left on {world.name}: live, staging; pass --move-aliases"
        in result.stderr
    )


def test_move_aliases_moves_them_all(runner: CliRunner, world: SimpleNamespace) -> None:
    replace_t(world, GROWN)

    result = rebase(runner, world, world.name, "--move-aliases")
    assert result.exit_code == 0, result.output
    new = result.stdout.strip()
    assert f"Moved alias 'live' -> {new}" in result.stderr
    assert f"Moved alias 'staging' -> {new}" in result.stderr
    assert "Aliases left on" not in result.stderr
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
    # What --only-alias leaves behind is named, as it is with no alias option.
    assert (
        f"Aliases left on {world.name}: staging; pass --move-aliases" in result.stderr
    )


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


def test_an_entry_with_only_unprobed_sources_is_attempted(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-derived and settled by the hash, but not declared drift-free."""
    monkeypatch.setattr(drift, "is_checkable", lambda leaf, record: False)
    commits = commit_count(world.catalog)

    result = rebase(runner, world, world.name, "--move-aliases")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{world.name}\n"
    assert "No source could be probed (t)" in result.stderr
    assert "drift not ruled out" in result.stderr
    assert "no drift" not in result.stderr
    assert "Aliases not moved: nothing to rebase" in result.stderr
    assert targets(world, "live", "staging") == (world.name, world.name)
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


def test_rebase_entry_checks_its_alias_arguments(world: SimpleNamespace) -> None:
    entry = world.catalog.get_catalog_entry(world.name)
    # 0.4.5 took the names to move as `move_aliases`; they must not mean "all".
    with pytest.raises(TypeError, match="only_aliases"):
        rebase_entry(entry, move_aliases=("staging",))
    with pytest.raises(ValueError, match="mutually exclusive"):
        rebase_entry(entry, move_aliases=True, only_aliases=("live",))
    # A bare string would otherwise be read as one alias per character.
    with pytest.raises(TypeError, match="only_aliases"):
        rebase_entry(entry, only_aliases="live")
    # An empty iterator names no alias.
    noop = rebase_entry(entry, move_aliases=True, only_aliases=iter(()))
    assert noop.status == RebaseStatus.NOOP


def test_an_unregistered_alias_is_named_by_when_it_was_read(
    world: SimpleNamespace,
) -> None:
    """Whether the pull removed it or it never existed is the read's, not the
    name's: a plan that knows the new name but was read before the pull."""
    entry = world.catalog.get_catalog_entry(world.name)
    plan = AliasPlan(entry, {}, new="0" * 12, only_aliases=("nope",))
    with pytest.raises(RebaseError, match=f"{world.name} has no alias nope"):
        plan.checked()
    with pytest.raises(RebaseError, match="points at nothing after the pull"):
        plan.evolved(pulled=True)


def conflict_dropped_column(w: SimpleNamespace) -> str:
    replace_t(w, pa.table({"b": ["x", "y"]}))
    return w.name


def conflict_gone_table(w: SimpleNamespace) -> str:
    w.con.drop_table("t")
    return w.name


def two_source_entry(w: SimpleNamespace) -> str:
    w.con.create_table("u", RECORDED.to_pandas())
    t, u = w.con.table("t"), w.con.table("u")
    return w.catalog.add(t.select("a").union(u.select("a"))).name


def conflict_two_changed(w: SimpleNamespace) -> str:
    name = two_source_entry(w)
    replace_t(w, pa.table({"b": ["x", "y"]}))
    w.con.drop_table("u")
    w.con.create_table("u", GROWN.to_pandas())
    return name


def conflict_two_gone(w: SimpleNamespace) -> str:
    name = two_source_entry(w)
    w.con.drop_table("t")
    w.con.drop_table("u")
    return name


def conflict_gone_beside_changed(w: SimpleNamespace) -> str:
    name = two_source_entry(w)
    w.con.drop_table("u")
    replace_t(w, GROWN)
    return name


def conflict_gone_beside_unreachable(w: SimpleNamespace) -> str:
    name, _, _ = refuse_beside_unreachable(lambda w: w.con.drop_table("t"))(w)
    return name


@pytest.mark.parametrize(
    "setup, headline, op_name, gone, changed",
    (
        pytest.param(
            conflict_dropped_column,
            "Field cannot be rebuilt; changed sources in entry:",
            "Field",
            [],
            ["t"],
            id="dropped-column",
        ),
        pytest.param(
            conflict_gone_table,
            "is gone",
            None,
            ["t"],
            [],
            id="gone-table",
        ),
        pytest.param(
            conflict_two_changed,
            "Field cannot be rebuilt; changed sources in entry:",
            "Field",
            [],
            ["t", "u"],
            id="two-changed",
        ),
        pytest.param(
            conflict_two_gone,
            "are gone",
            None,
            ["t", "u"],
            [],
            id="two-gone",
        ),
        pytest.param(
            conflict_gone_beside_changed,
            "is gone",
            None,
            ["u"],
            ["t"],
            id="gone-beside-changed",
        ),
        pytest.param(
            conflict_gone_beside_unreachable,
            "is gone",
            None,
            ["t"],
            ["u"],
            id="gone-beside-unreachable",
        ),
    ),
)
def test_a_conflict_names_the_op_and_source_with_check_sources_schemas(
    runner: CliRunner,
    world: SimpleNamespace,
    setup: Callable,
    headline: str,
    op_name: str | None,
    gone: list[str],
    changed: list[str],
) -> None:
    name = setup(world)
    checked = runner.invoke(cli, ["--path", world.catalog_path, "check-sources", name])
    assert checked.exit_code == 3, checked.output
    pair = [line for line in checked.stdout.splitlines() if line.startswith("    ")]

    result = rebase(runner, world, name, "-a", "v2")
    assert result.exit_code == 4, result.output
    # The headline names the gone sources, or every changed one when an op failed.
    first = result.stderr.splitlines()[0]
    assert first.startswith(f"{name}: conflict: ") and headline in first
    assert all(f"DatabaseTable {n}" in first for n in (gone or changed))
    assert all(f"DatabaseTable {n}" not in first for n in gone and changed)
    # Every source the sweep did not find equal gets its check-sources block.
    assert pair and all(line in result.stderr.splitlines() for line in pair)
    assert "could not rebuild" not in result.stderr
    assert "Alias 'v2' not added" in result.stderr
    if op_name is not None:
        # Named once, by the headline; the cause follows it.
        assert result.stderr.count(op_name) == 1
        assert "XorqTypeError: Column 'a' is not found" in result.stderr
    else:
        # The blocks already say it; the cause is not echoed a third time.
        assert " is table-missing" not in result.stderr
    conflicted = rebase_entry(reopen(world).get_catalog_entry(name))
    assert conflicted.status == RebaseStatus.CONFLICT
    assert conflicted.new_entry.name == name
    assert conflicted.conflict.op_name == op_name
    assert sorted(r.leaf.name for r in conflicted.conflict.sources) == sorted(
        gone + changed
    )


@pytest.mark.parametrize("moving", (("--move-aliases",), ("--only-alias", "live")))
def test_a_conflict_says_the_aliases_were_not_moved(
    runner: CliRunner, world: SimpleNamespace, moving: tuple[str, ...]
) -> None:
    name = conflict_dropped_column(world)

    result = rebase(runner, world, name, *moving)
    assert result.exit_code == 4, result.output
    assert "Aliases not moved: the rebase conflicted" in result.stderr
    assert targets(world, "live") == (name,)


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
    w.other = other_entry(w)
    w.catalog.add_alias(w.other, "fresh")
    return w.name, ("--only-alias", "fresh"), ()


def refuse_taken_alias(w: SimpleNamespace) -> Setup:
    """`-a` names an alias on another entry: refused, not taken."""
    w.catalog.add_alias(other_entry(w), "fresh")
    replace_t(w, GROWN)
    return w.name, ("-a", "fresh"), ()


def refuse_both_alias_flags(w: SimpleNamespace) -> Setup:
    replace_t(w, GROWN)
    return w.name, ("--move-aliases", "--only-alias", "live"), ()


def refuse_alias_add_failing(w: SimpleNamespace) -> Setup:
    """`catalog.add` fails registering `-a` after staging the entry: rolled back."""
    replace_t(w, GROWN)
    entry = w.catalog.get_catalog_entry(w.name)
    alias_path = CatalogAlias("fresh", entry).alias_path
    alias_path.parent.mkdir(exist_ok=True)
    alias_path.write_text("not a symlink")
    return w.name, ("--only-alias", "live", "-a", "fresh"), ()


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


def refuse_unmatched_source(w: SimpleNamespace) -> Setup:
    """The rewrite disagrees with the sweep: a bug, refused as one."""
    replace_t(w, GROWN)

    def unmatched(*_: object) -> None:
        raise UnmatchedSourceError("DatabaseTable", LookupError("t matched no source"))

    w.monkeypatch.setattr(rebase_module, "refresh_schemas", unmatched)
    return w.name, (), ()


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


REFUSALS = (
    pytest.param(refuse_unknown_alias, 1, "no alias nope", id="unknown-alias"),
    pytest.param(
        refuse_elsewhere_alias_without_drift,
        1,
        "alias 'fresh' points at {other}, not {name}",
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
    pytest.param(
        refuse_alias_add_failing,
        1,
        "non symlink already exists",
        id="alias-add-failing",
    ),
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
    pytest.param(
        refuse_unmatched_source, 1, "t matched no source", id="unmatched-source"
    ),
    pytest.param(refuse_dropped_column, 4, "cannot be rebuilt", id="column"),
    pytest.param(refuse_dropped_table, 4, "table-missing", id="table"),
    pytest.param(
        refuse_beside_unreachable(lambda w: w.con.drop_table("t")),
        4,
        "DatabaseTable u: unreachable",
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
    other = getattr(world, "other", None)
    assert message.format(name=name, gone=gone, other=other) in result.stderr
    assert result.stdout == ""
    assert reopen(world).list() == entries
    assert commit_count(reopen(world)) == commits
    assert not any(path.exists() for path in gone)
    assert not any(cwd.iterdir())


# Alias moves and their rollback.
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
    # The user saw `live` on the old entry; where it went is said, not refused.
    assert "Moved alias 'live'" not in result.stderr
    assert (
        f"Alias 'live' not moved: no longer on {world.name} after the pull"
        in result.stderr
    )
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
            None,
            ("--only-alias", "live"),
            "alias 'live' points at {other}, not {name}",
            id="only-alias",
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


def test_an_entry_alias_the_pull_moved_onto_the_new_entry_is_accepted(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A teammate rebased `live` first and pushed; ours lands on the same entry."""
    replace_t(world, GROWN)
    new = rebase_old(world).new_entry.name
    pull_then(monkeypatch, lambda c: c.add_alias(new, "live", sync=False))

    result = rebase(runner, world, "live", "--move-aliases")
    monkeypatch.undo()
    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == new
    assert f"Moved alias 'staging' -> {new}" in result.stderr
    # The user saw `live` on the old entry; where the pull put it is said.
    assert f"Alias 'live' already on {new}" in result.stderr
    assert targets(world, "live", "staging") == (new, new)


def test_an_only_alias_the_pull_removed_is_refused_after_the_pull(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the sweep `live` was on the old entry; the pull removed it."""
    replace_t(world, GROWN)
    pull_then(monkeypatch, lambda c: c.remove_alias("live", sync=False))

    result = rebase(runner, world, world.name, "--only-alias", "live")
    monkeypatch.undo()
    assert result.exit_code == 1, result.output
    assert (
        f"alias 'live' points at nothing after the pull, not {world.name}"
        in result.stderr
    )
    assert result.stdout == ""
    catalog = reopen(world)
    assert catalog.list() == [world.name]
    assert "live" not in catalog.list_aliases()


@pytest.mark.parametrize("args", (("-a", "fresh"), ("--only-alias", "fresh")))
def test_an_alias_on_another_entry_is_refused_before_the_pull(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[str, ...],
) -> None:
    """Once the new name is known, nothing is dumped or pulled for a refusal."""
    other = other_entry(world)
    world.catalog.add_alias(other, "fresh")
    replace_t(world, GROWN)
    pulls = []
    pull_then(monkeypatch, pulls.append)
    monkeypatch.setattr(
        ExprDumper, "dump_expr", lambda self: pytest.fail("dumped before refusing")
    )

    result = rebase(runner, world, world.name, *args)
    monkeypatch.undo()
    assert result.exit_code == 1, result.output
    assert f"alias 'fresh' points at {other}" in result.stderr
    assert pulls == []
    assert set(reopen(world).list()) == {world.name, other}


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


def push_to_origin(world: SimpleNamespace) -> Path:
    """Give the world's catalog a bare origin it is in step with."""
    origin = world.tmp_path / "origin.git"
    Repo.init(origin, bare=True, initial_branch=MAIN_BRANCH)
    world.catalog.repo.create_remote(name="origin", url=str(origin))
    world.catalog.repo.git.push("-u", "origin", MAIN_BRANCH)
    return origin


def origin_moved_on(world: SimpleNamespace, origin: Path) -> None:
    """Another clone pushes after our pull: our push is rejected."""
    other = Catalog.clone_from(
        url=str(origin), repo_path=world.tmp_path / "other", annex=False
    )
    add_expr_entry(other, "elsewhere", value=1)
    other.push()


def origin_gone(world: SimpleNamespace, origin: Path) -> None:
    shutil.rmtree(origin)


@pytest.mark.parametrize(
    "push_fails, raised",
    (
        pytest.param(origin_moved_on, "CatalogPushError", id="rejected"),
        pytest.param(origin_gone, "GitCommandError", id="unreachable"),
    ),
)
def test_a_failed_push_keeps_the_rebase_and_exits_5(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    push_fails: Callable,
    raised: str,
) -> None:
    origin = push_to_origin(world)
    replace_t(world, GROWN)
    # The remote changes between the pull and the push.
    pull_then(monkeypatch, lambda c: push_fails(world, origin))

    result = rebase(runner, world, world.name, "--move-aliases")
    monkeypatch.undo()
    assert result.exit_code == RebaseExit.PUSH_FAILED, result.output
    new = result.stdout.strip()
    assert (
        f"Rebased {world.name} -> {new} locally; push failed: {raised}: "
        in result.stderr
    )
    assert "run `xorq catalog sync`" in result.stderr
    assert f"Moved alias 'live' -> {new}" in result.stderr
    # Committed locally, not rolled back.
    catalog = reopen(world)
    assert set(catalog.list()) == {world.name, new}
    assert targets(world, "live", "staging") == (new, new)


def test_the_hint_finishes_a_rejected_push(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = push_to_origin(world)
    replace_t(world, GROWN)
    pull_then(monkeypatch, lambda c: origin_moved_on(world, origin))
    new = rebase(runner, world, world.name).stdout.strip()
    monkeypatch.undo()

    result = runner.invoke(cli, ["--path", world.catalog_path, "sync"])
    assert result.exit_code == 0, result.output
    pushed = Repo(origin).git.ls_tree("-r", "--name-only", MAIN_BRANCH)
    assert f"entries/{new}" in pushed


def test_a_failed_pull_is_retryable_and_writes_nothing(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace_t(world, GROWN)
    before = commit_count(world.catalog)

    def failing_pull(self: Catalog) -> None:
        raise CatalogPullError("fetch failed")

    monkeypatch.setattr(Catalog, "pull", failing_pull)
    result = rebase(runner, world, world.name, "--move-aliases")
    assert result.exit_code == RebaseExit.UNREACHABLE, result.output
    assert "the sync's pull failed: CatalogPullError: fetch failed" in result.stderr
    assert result.stdout == ""
    assert commit_count(reopen(world)) == before
    assert targets(world, "live", "staging") == (world.name, world.name)


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


# `--json`.
ROOT_KEYS = {
    "state",
    "exit_code",
    "old_entry",
    "new_entry",
    "build_path",
    "failing_op",
    "drift",
    "aliases",
    "warnings",
}
ALIAS_KEYS = ("moved", "added", "already", "unmoved", "left")
FAILURE_STATES = {1: "refused", 2: "unreachable", 4: "conflict"}
# A usage error exits before the entry resolves, so it prints no document.
JSON_REFUSALS = tuple(p for p in REFUSALS if p.id != "both-alias-flags")


def aliases_document(**names: list[str]) -> dict[str, list[str]]:
    return {key: names.get(key, []) for key in ALIAS_KEYS}


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
    return ("--move-aliases", "-a", "fresh")


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
        assert document["aliases"] == aliases_document()
        assert alias_target_hash(reopen(world), "live") == world.name
        return
    new = document["new_entry"]
    assert new != world.name
    assert set(reopen(world).list()) == {world.name, world.other, new}
    assert document["aliases"] == aliases_document(
        moved=["live", "staging"], added=["fresh"]
    )
    assert targets(world, "live", "staging", "fresh") == (new, new, new)


def test_json_reports_the_aliases_the_pull_took_off_the_old_entry(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = other_entry(world)
    replace_t(world, GROWN)
    pull_then(monkeypatch, lambda c: c.add_alias(other, "live", sync=False))

    _, document = rebase_json(runner, world, world.name, "--move-aliases")
    monkeypatch.undo()
    assert document["state"] == "rebased"
    assert document["aliases"] == aliases_document(moved=["staging"], unmoved=["live"])


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
    # A write that failed and was rolled back exits 1 like a refusal.
    failed = setup is refuse_alias_add_failing
    assert document["state"] == ("failed" if failed else FAILURE_STATES[exit_code])
    assert (document["old_entry"], document["new_entry"]) == (old_entry, None)
    assert document["aliases"] == aliases_document()
    assert re.fullmatch(r"\w+: .+", document["error"], re.DOTALL)
    if exit_code == 4:
        # The headline's facts are the drift document; `error` is the detail.
        assert document["drift"]["state"] in ("changed", "table-missing")
        assert document["error"]
    else:
        # A refusal returns no sweep to report.
        assert document["drift"] is None
        assert document["failing_op"] is None
        other = getattr(world, "other", None)
        assert message.format(name=name, gone=gone, other=other) in document["error"]
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
    assert re.fullmatch(r"\w+: .+", document["error"], re.DOTALL)
    assert "v2" not in reopen(world).list_aliases()


def test_json_reports_a_failed_push_with_the_entry_it_kept(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written locally, rejected by the remote: the document names the entry."""
    origin = push_to_origin(world)
    replace_t(world, GROWN)
    before = check_sources_entry(runner, world, world.name)
    pull_then(monkeypatch, lambda c: origin_moved_on(world, origin))

    result, document = rebase_json(runner, world, world.name, "--move-aliases")
    monkeypatch.undo()
    assert (document["state"], result.exit_code) == ("push-failed", 5)
    new = document["new_entry"]
    assert new not in (None, world.name)
    assert document["aliases"] == aliases_document(moved=["live", "staging"])
    assert document["drift"] == before
    assert document["error"].startswith("CatalogPushError: push failed")
    assert set(reopen(world).list()) == {world.name, new}


def test_json_reports_a_push_that_raised_before_pushing_as_push_failed(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A push that fails its own checks still follows a committed write."""

    def inconsistent(catalog: Catalog) -> None:
        raise AssertionError("catalog inconsistent")

    replace_t(world, GROWN)
    monkeypatch.setattr(Catalog, "push", inconsistent)
    result, document = rebase_json(runner, world, world.name, "--move-aliases")
    monkeypatch.undo()

    assert (document["state"], result.exit_code) == ("push-failed", 5)
    new = document["new_entry"]
    assert new not in (None, world.name)
    assert document["aliases"] == aliases_document(moved=["live", "staging"])
    assert document["error"] == "AssertionError: catalog inconsistent"
    assert set(reopen(world).list()) == {world.name, new}


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

    _, document = rebase_json(runner, world, name)
    assert document["state"] == "rebased"
    assert not [w for w in document["warnings"] if "local filesystem path" in w]


def test_a_plain_rebase_keeps_the_local_path_warning_off_stderr(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """Without --json the relocate hint is not raised either.

    pytest captures warnings itself, so stderr is checked through a recorder:
    outside pytest the warning would print there.
    """
    path = world.tmp_path / "ext.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    con = SqliteBackend().connect(str(world.tmp_path / "ingest.sqlite"))
    read = deferred_read_parquet(path, con, table_name="ingested")
    name = world.catalog.add(read.filter(read.a > 1), relocate_reads=False).name
    GROWN.to_pandas().to_parquet(path, index=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = rebase(runner, world, name)
    assert result.exit_code == 0, result.output
    assert not [w for w in caught if "local filesystem path" in str(w.message)]


@pytest.mark.parametrize(
    "setup, state",
    (
        pytest.param(lambda w: replace_t(w, GROWN), "rebased", id="rebased"),
        pytest.param(
            lambda w: w.monkeypatch.setattr(
                rebase_module, "check_bundle", failing_check
            ),
            "refused",
            id="refused",
        ),
    ),
)
def test_json_puts_what_the_rebase_warned_in_the_document(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable,
    state: str,
) -> None:
    """Any warning, not just the local-path one, is reported, not printed.

    pytest swaps in its own warning display, so stderr alone cannot show it.
    """
    world.monkeypatch = monkeypatch
    real_fetch_archive = rebase_module.fetch_archive

    def warning_fetch(catalog_entry: object) -> None:
        warnings.warn("deprecated knob", FutureWarning, stacklevel=1)
        real_fetch_archive(catalog_entry)

    monkeypatch.setattr(rebase_module, "fetch_archive", warning_fetch)
    setup(world)
    _, document = rebase_json(runner, world)
    monkeypatch.undo()

    assert document["state"] == state
    # Membership: a collected object's ResourceWarning may land there too.
    assert "FutureWarning: deprecated knob" in document["warnings"]


def failing_check(catalog_entry: object) -> None:
    raise RebaseError("no bundle", RebaseExit.REFUSED)


def test_json_reports_a_failed_write_as_one_document(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that fails midway still ends in one whole document."""
    replace_t(world, GROWN)
    fail_nth_alias_move(monkeypatch, 2)
    result, document = rebase_json(runner, world, world.name, "--move-aliases")
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

    def rebasing(*args: object, **kwargs: object) -> rebase_module.RebaseResult:
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


@pytest.mark.parametrize(
    "exit_code", (RebaseExit.CONFLICT, RebaseExit.PUSH_FAILED, 0, 3)
)
def test_a_rebase_error_refuses_only_with_a_refusal_code(exit_code: int) -> None:
    """A conflict or failed push is a result; a raise with its code has no state."""
    with pytest.raises(ValueError, match="exit 1 or 2"):
        RebaseError("refused", exit_code)


def test_every_rebase_state_has_its_own_exit_code() -> None:
    """Only a completed rebase exits 0; a new state must name its code."""
    assert {state: state.exit_code for state in (*RebaseStatus, *RebaseFailure)} == {
        RebaseStatus.NOOP: 0,
        RebaseStatus.ATTEMPTED: 0,
        RebaseStatus.REBASED: 0,
        RebaseStatus.CONFLICT: RebaseExit.CONFLICT,
        RebaseStatus.PUSH_FAILED: RebaseExit.PUSH_FAILED,
        RebaseFailure.REFUSED: RebaseExit.REFUSED,
        RebaseFailure.UNREACHABLE: RebaseExit.UNREACHABLE,
        RebaseFailure.FAILED: RebaseExit.REFUSED,
    }


def test_a_result_and_a_failure_never_share_a_state() -> None:
    """Both fill the document's `state`; one value must name one outcome."""
    assert not set(map(str, RebaseStatus)) & set(map(str, RebaseFailure))


@pytest.mark.parametrize("status", (RebaseStatus.CONFLICT, RebaseStatus.PUSH_FAILED))
def test_a_result_cannot_claim_a_status_its_fields_contradict(
    world: SimpleNamespace, status: RebaseStatus
) -> None:
    """A conflict without its detail, or a failed push without its error, would
    print a document whose state disagrees with its own fields."""
    replace_t(world, GROWN)
    rebased = rebase_old(world, sync=False)
    assert rebased.status == RebaseStatus.REBASED
    with pytest.raises(ValueError, match=str(status)):
        evolve(rebased, status=status)
