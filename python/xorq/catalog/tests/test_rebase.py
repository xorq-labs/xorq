"""`xorq catalog rebase` (#2323), over the sqlite world `test_drift` uses."""

from __future__ import annotations

import json
import pickle
import re
import shutil
import sys
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pyarrow as pa
import pytest
from attr import evolve
from click.testing import CliRunner, Result
from git import Repo

import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.caching import ParquetCache
from xorq.catalog import cli as catalog_cli
from xorq.catalog import drift
from xorq.catalog import rebase as rebase_module
from xorq.catalog.aliases import AliasPlan
from xorq.catalog.catalog import (
    Catalog,
    CatalogAlias,
    CatalogEntry,
    CatalogPullError,
)
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
from xorq.cli import cli as top_cli
from xorq.common.exceptions import LocalReadPathWarning, UnmatchedSourceError
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
RETYPED = pa.table({"a": pa.array([1.0, 2.0], pa.float64()), "b": ["x", "y"]})


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


def test_a_rebase_onto_an_entry_already_cataloged_keeps_it_and_says_so(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """The old entry still drifts, so a second rebase derives the same hash."""
    replace_t(world, GROWN)
    _, first = rebase_json(runner, world, world.name, "--no-sync")
    new = first["new_entry"]
    archive = Path(reopen(world).get_catalog_entry(new).catalog_path)
    mtime = archive.stat().st_mtime_ns

    again = rebase(runner, world, world.name, "--no-sync")
    _, again_document = rebase_json(runner, world, world.name, "--no-sync")
    assert (first["created"], again_document["created"]) == (True, False)
    assert again.exit_code == 0, again.output
    assert again.stdout == f"{new}\n"
    assert (
        f"Rebased {world.name} -> {new} (already cataloged; existing archive kept)"
        in again.stderr
    )
    assert archive.stat().st_mtime_ns == mtime


@pytest.mark.parametrize(
    "live, output",
    (
        pytest.param(GROWN, "Output: c float64 added", id="added"),
        pytest.param(RETYPED, "Output: a int64 -> float64", id="retyped"),
    ),
)
def test_a_rebase_states_how_the_output_schema_changed(
    runner: CliRunner, world: SimpleNamespace, live: pa.Table, output: str
) -> None:
    replace_t(world, live)
    result = rebase(runner, world)
    assert result.exit_code == 0, result.output
    assert output in result.stderr.splitlines()


def test_the_output_names_a_derived_column_its_source_retyped(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """The sweep compares sources; only the output shows `a2` became float64."""
    t = world.con.table("t")
    name = world.catalog.add(t.select(a2=t.a + 1)).name
    replace_t(world, RETYPED)
    _, document = rebase_json(runner, world, name, "--no-sync")
    assert document["output_changes"] == [
        {"name": "a2", "before": "int64", "after": "float64"}
    ]


@pytest.mark.parametrize(
    "make, change",
    (
        pytest.param(
            lambda t: (t.a + 1).name("a2"),
            {"name": "a2", "before": "int64", "after": "float64"},
            id="column",
        ),
        pytest.param(
            lambda t: t.a.max().name("top"),
            {"name": "top", "before": "int64", "after": "float64"},
            id="scalar",
        ),
    ),
)
def test_a_column_or_scalar_entry_states_its_output_change(
    runner: CliRunner, world: SimpleNamespace, make: Callable, change: dict
) -> None:
    """Compared as a table, as an entry's metadata is."""
    name = world.catalog.add(make(world.con.table("t"))).name
    replace_t(world, RETYPED)
    _, document = rebase_json(runner, world, name, "--no-sync")
    assert document["state"] == "rebased"
    assert document["output_changes"] == [change]


def test_an_unprobed_result_names_its_sources(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An UNPROBED result with no names would print an empty list."""
    monkeypatch.setattr(drift, "is_checkable", lambda leaf, record: False)
    unprobed = rebase_old(world)
    with pytest.raises(ValueError, match="unprobed"):
        evolve(unprobed, unprobed=())


@pytest.mark.parametrize("args", (("--dry-run",), ("--no-add",)))
def test_an_unprobed_entry_writes_nothing_under_dry_run_or_no_add(
    runner: CliRunner,
    world: SimpleNamespace,
    cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[str, ...],
) -> None:
    monkeypatch.setattr(drift, "is_checkable", lambda leaf, record: False)
    result = rebase(runner, world, world.name, *args)
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{world.name}\n"
    assert not any(cwd.iterdir())


@pytest.mark.parametrize(
    "make, live, output, change",
    (
        pytest.param(
            lambda t: t.select("a"),
            GROWN,
            "Output: unchanged",
            None,
            id="unchanged",
        ),
        pytest.param(
            None,
            pa.table({"a": pa.array([1, 2], pa.int64())}),
            "Output: b removed",
            {"name": "b", "before": "string", "after": None},
            id="removed",
        ),
    ),
)
def test_the_output_line_says_unchanged_or_removed(
    runner: CliRunner,
    world: SimpleNamespace,
    make: Callable,
    live: pa.Table,
    output: str,
    change: dict | None,
) -> None:
    # The world's own entry keeps every column; `None` rebases that one.
    name = (
        world.name
        if make is None
        else world.catalog.add(make(world.con.table("t"))).name
    )
    replace_t(world, live)
    result = rebase(runner, world, name, "--no-sync")
    assert result.exit_code == 0, result.output
    assert output in result.stderr.splitlines()
    _, document = rebase_json(runner, world, name, "--no-sync")
    assert document["output_changes"] == ([] if change is None else [change])


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


def test_an_entry_with_only_unprobed_sources_is_left_alone(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing can be refreshed, so it is not loaded, nor declared drift-free."""
    monkeypatch.setattr(drift, "is_checkable", lambda leaf, record: False)
    commits = commit_count(world.catalog)
    loads = []
    real_load = CatalogEntry.load_expr
    monkeypatch.setattr(
        CatalogEntry,
        "load_expr",
        lambda self, **kw: loads.append(self.name) or real_load(self, **kw),
    )

    result = rebase(runner, world, world.name, "--move-aliases")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{world.name}\n"
    assert "no source can be probed (t); nothing done" in result.stderr
    assert "drift not ruled out" in result.stderr
    assert "no drift" not in result.stderr
    assert "Aliases not moved: nothing to rebase" in result.stderr
    assert targets(world, "live", "staging") == (world.name, world.name)
    assert commit_count(reopen(world)) == commits
    assert loads == []
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


@pytest.mark.parametrize(
    "recorded, problem",
    (
        pytest.param((3, 0), "was built on Python 3.0", id="other-minor"),
        pytest.param(None, "records no Python minor", id="unrecorded"),
    ),
)
def test_a_python_minor_mismatch_is_overridable_and_warned(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    recorded: tuple[int, int] | None,
    problem: str,
) -> None:
    entry = world.catalog.get_catalog_entry(world.name)
    assert recorded_python_minor(entry) == tuple(sys.version_info[:2])
    replace_t(world, GROWN)
    monkeypatch.setattr(rebase_module, "recorded_python_minor", lambda _: recorded)

    result = rebase(runner, world, world.name, "--ignore-venv-mismatch")
    assert result.exit_code == 0, result.output
    assert f"PythonMinorWarning: {world.name} {problem}" in result.stderr
    assert result.stdout.strip() != world.name
    _, document = rebase_json(
        runner, world, world.name, "--ignore-venv-mismatch", "--no-sync"
    )
    assert [w for w in document["warnings"] if w.startswith("PythonMinorWarning")]


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


# Preview, build-only, and provenance (#2325).
@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty working directory, for `./builds` and to show nothing lands."""
    path = tmp_path / "cwd"
    path.mkdir()
    monkeypatch.chdir(path)
    return path


@ALL_BACKENDS
def test_a_preview_of_a_drifted_entry_reports_both_hashes_and_writes_nothing(
    runner: CliRunner, world: SimpleNamespace, cwd: Path
) -> None:
    replace_t(world, GROWN)
    commits = commit_count(world.catalog)

    result = rebase(runner, world, world.name, "--dry-run", "-a", "v2")
    assert result.exit_code == 3, result.output
    new = result.stdout.strip()
    assert new != world.name
    assert f"Would rebase {world.name} -> {new}; nothing cataloged" in result.stderr
    assert f"Would add alias 'v2' -> {new}" in result.stderr
    assert (
        f"Aliases would stay on {world.name}: live, staging; pass --move-aliases "
        "or --only-alias to move them"
    ) in result.stderr
    assert reopen(world).list() == [world.name]
    assert "v2" not in reopen(world).list_aliases()
    assert targets(world, "live", "staging") == (world.name, world.name)
    assert commit_count(reopen(world)) == commits
    assert not any(cwd.iterdir())
    # The hash previewed is the one the rebase then catalogs.
    rebased = rebase(runner, world)
    assert rebased.exit_code == 0, rebased.output
    assert rebased.stdout.strip() == new


@ALL_BACKENDS
@pytest.mark.parametrize(
    "drift_t, exit_code",
    (
        pytest.param(lambda w: None, 0, id="undrifted"),
        pytest.param(
            lambda w: replace_t(w, pa.table({"b": ["x", "y"]})), 4, id="conflicted"
        ),
    ),
)
def test_a_preview_without_a_rebase_to_make_exits_as_the_rebase_would(
    runner: CliRunner,
    world: SimpleNamespace,
    cwd: Path,
    drift_t: Callable,
    exit_code: int,
) -> None:
    drift_t(world)
    commits = commit_count(world.catalog)

    result = rebase(runner, world, world.name, "--dry-run")
    assert result.exit_code == exit_code, result.output
    assert "Would rebase" not in result.stderr
    assert result.stdout == ("" if exit_code else f"{world.name}\n")
    assert reopen(world).list() == [world.name]
    assert commit_count(reopen(world)) == commits
    assert not any(cwd.iterdir())


def test_a_result_cannot_name_a_hash_its_status_contradicts(
    world: SimpleNamespace,
) -> None:
    """A preview of the old hash, or a rebase under another entry's hash, would
    print a document that disagrees with itself."""
    replace_t(world, GROWN)
    previewed = rebase_old(world, dry_run=True)
    with pytest.raises(ValueError, match="new_hash"):
        evolve(previewed, new_hash=world.name)
    rebased = rebase_old(world, sync=False)
    with pytest.raises(ValueError, match="new_hash"):
        evolve(rebased, new_hash=previewed.new_hash + "x")


def test_a_preview_result_names_the_new_hash(world: SimpleNamespace) -> None:
    replace_t(world, GROWN)

    previewed = rebase_old(world, dry_run=True)
    assert previewed.status == RebaseStatus.PREVIEW
    assert previewed.new_entry.name == world.name
    assert previewed.build_path is None
    assert previewed.new_hash == rebase_old(world).new_entry.name


@ALL_BACKENDS
def test_build_only_prints_a_directory_that_adds_to_the_same_hash(
    runner: CliRunner, world: SimpleNamespace, cwd: Path
) -> None:
    replace_t(world, GROWN)
    commits = commit_count(world.catalog)

    result = rebase(runner, world, world.name, "--no-add")
    assert result.exit_code == 3, result.output
    build_path = Path(result.stdout.strip())
    assert build_path == cwd / "builds" / build_path.name
    assert (build_path / DumpFiles.expr).is_file()
    assert f"Built {world.name} -> {build_path.name}, not cataloged" in result.stderr
    assert reopen(world).list() == [world.name]
    assert targets(world, "live", "staging") == (world.name, world.name)
    assert commit_count(reopen(world)) == commits

    # Adding it by hand needs nothing from the cwd: the bundle is staged.
    added = runner.invoke(cli, ["--path", world.catalog_path, "add", str(build_path)])
    assert added.exit_code == 0, added.output
    catalog = reopen(world)
    assert set(catalog.list()) == {world.name, build_path.name}
    entry = catalog.get_catalog_entry(build_path.name)
    assert entry.columns == ("a", "b", "c")
    assert entry.rebased_from == world.name
    # And a rebase lands on that same entry.
    rebased = rebase(runner, world)
    assert rebased.exit_code == 0, rebased.output
    assert rebased.stdout.strip() == build_path.name


def test_build_only_keeps_an_external_read_external_through_add(
    runner: CliRunner, world: SimpleNamespace, tmp_path: Path, cwd: Path
) -> None:
    """`catalog add` of the directory (a Path) relocates nothing, so the hash
    holds even where relocating the read would move it into the archive."""
    path = tmp_path / "t.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    read = deferred_read_parquet(path, xo.connect(), table_name="t")
    name = world.catalog.add(read.filter(read.a > 1), relocate_reads=False).name
    GROWN.to_pandas().to_parquet(path, index=False)

    result = rebase(runner, world, name, "--no-add")
    assert result.exit_code == 3, result.output
    build_path = Path(result.stdout.strip())
    added = runner.invoke(cli, ["--path", world.catalog_path, "add", str(build_path)])
    assert added.exit_code == 0, added.output

    entry = reopen(world).get_catalog_entry(build_path.name)
    assert entry.columns == ("a", "b", "c")
    assert entry.rebased_from == name
    (leaf,) = drift.read_record(entry).external_leaves
    assert leaf.name == str(path)
    rebased = rebase(runner, world, name)
    assert rebased.exit_code == 0, rebased.output
    assert rebased.stdout.strip() == build_path.name


def test_build_only_writes_into_builds_dir(
    runner: CliRunner, world: SimpleNamespace, tmp_path: Path, cwd: Path
) -> None:
    replace_t(world, GROWN)
    builds_dir = tmp_path / "elsewhere"

    result = rebase(
        runner, world, world.name, "--no-add", "--builds-dir", str(builds_dir)
    )
    assert result.exit_code == 3, result.output
    build_path = Path(result.stdout.strip())
    assert build_path.parent == builds_dir
    assert not any(cwd.iterdir())
    again = tmp_path / "again"
    built = rebase_old(world, add=False, builds_dir=again)
    assert (built.status, built.build_path) == (
        RebaseStatus.BUILT,
        again / build_path.name,
    )
    assert built.new_hash == build_path.name


def test_build_only_refuses_a_taken_build_directory(
    runner: CliRunner, world: SimpleNamespace, cwd: Path
) -> None:
    """An earlier build under the same hash (`xorq build` shares the default) is
    kept, and nothing is built over it."""
    replace_t(world, GROWN)
    new = rebase(runner, world, world.name, "--dry-run").stdout.strip()
    taken = cwd / "builds" / new
    taken.mkdir(parents=True)
    (taken / "edited.txt").write_text("mine")

    result = rebase(runner, world, world.name, "--no-add")
    assert result.exit_code == 1, result.output
    assert f"builds/{new} exists; --no-add does not replace it" in result.stderr
    assert result.stdout == ""
    assert [p.name for p in taken.iterdir()] == ["edited.txt"]
    assert reopen(world).list() == [world.name]


@pytest.mark.parametrize(
    "occupy, message",
    (
        pytest.param(
            lambda builds, new: (builds.mkdir(), (builds / new).symlink_to("gone")),
            "exists; --no-add does not replace it",
            id="dangling-link",
        ),
        pytest.param(
            lambda builds, new: builds.write_text("a file"),
            "is not a directory",
            id="builds-dir-is-a-file",
        ),
    ),
)
def test_build_only_refuses_an_unusable_target_before_building(
    runner: CliRunner,
    world: SimpleNamespace,
    cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    occupy: Callable,
    message: str,
) -> None:
    """Refused where the hash is known, not after the dump."""
    replace_t(world, GROWN)
    new = rebase(runner, world, world.name, "--dry-run").stdout.strip()
    occupy(cwd / "builds", new)
    staged = []
    real_stage = rebase_module.stage_bundle
    monkeypatch.setattr(
        rebase_module,
        "stage_bundle",
        lambda *args: staged.append(args) or real_stage(*args),
    )
    result = rebase(runner, world, world.name, "--no-add")
    assert result.exit_code == 1, result.output
    assert message in result.stderr
    assert staged == []


def test_build_only_interrupted_after_its_claim_can_be_retried(
    world: SimpleNamespace, cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Ctrl-C between the claim and the rename gives the claim back."""
    real_rename = Path.rename

    def interrupted(self: Path, target: Path) -> None:
        raise KeyboardInterrupt

    replace_t(world, GROWN)
    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(KeyboardInterrupt):
        rebase_old(world, add=False)
    monkeypatch.setattr(Path, "rename", real_rename)
    assert list((cwd / "builds").iterdir()) == []
    assert rebase_old(world, add=False).status == RebaseStatus.BUILT


def test_build_only_without_a_claim_publishes_onto_the_absent_target(
    runner: CliRunner,
    world: SimpleNamespace,
    cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows' path, run here: no mkdir claim, the rename itself publishes."""
    monkeypatch.setattr(rebase_module, "CLAIM_BEFORE_RENAME", False)
    replace_t(world, GROWN)
    result = rebase(runner, world, world.name, "--no-add")
    assert result.exit_code == 3, result.output
    build_path = Path(result.stdout.strip())
    assert (build_path / DumpFiles.expr).is_file()
    assert [p.name for p in (cwd / "builds").iterdir()] == [build_path.name]


def test_build_only_that_fails_after_the_dump_leaves_builds_dir_alone(
    runner: CliRunner,
    world: SimpleNamespace,
    cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The build moves into BUILDS_DIR only once complete."""

    def unreadable(catalog_entry: object, build_path: Path) -> None:
        raise RebaseError("bundle unreadable", RebaseFailure.UNREACHABLE)

    replace_t(world, GROWN)
    monkeypatch.setattr(rebase_module, "stage_bundle", unreadable)
    result = rebase(runner, world, world.name, "--no-add")
    assert result.exit_code == 2, result.output
    assert not (cwd / "builds").exists()


def test_build_only_whose_copy_fails_leaves_builds_dir_empty(
    runner: CliRunner,
    world: SimpleNamespace,
    cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A move that copies (across filesystems) and fails partway leaves no
    partial build for a retry to be refused over."""
    real_move = shutil.move

    def partial_move(src: Path, dst: Path) -> object:
        Path(dst).mkdir()
        (Path(dst) / DumpFiles.expr).write_text("half")
        raise OSError(28, "No space left on device")

    replace_t(world, GROWN)
    monkeypatch.setattr(rebase_module.shutil, "move", partial_move)
    result = rebase(runner, world, world.name, "--no-add")
    monkeypatch.setattr(rebase_module.shutil, "move", real_move)
    assert result.exit_code == 1, result.output
    assert list((cwd / "builds").iterdir()) == []
    retried = rebase(runner, world, world.name, "--no-add")
    assert retried.exit_code == 3, retried.output


def test_build_only_refuses_a_claim_written_into_before_the_rename(
    runner: CliRunner,
    world: SimpleNamespace,
    cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A writer that fills the claimed directory first keeps it; this run is
    refused and leaves no stage behind."""
    real_rename = Path.rename

    def written_into(self: Path, target: Path) -> None:
        (Path(target) / "theirs.txt").write_text("first")
        raise OSError(39, "Directory not empty")

    replace_t(world, GROWN)
    monkeypatch.setattr(Path, "rename", written_into)
    result = rebase(runner, world, world.name, "--no-add")
    monkeypatch.setattr(Path, "rename", real_rename)
    assert result.exit_code == 1, result.output
    assert "exists; --no-add does not replace it" in result.stderr
    (claimed,) = (cwd / "builds").iterdir()
    assert [p.name for p in claimed.iterdir()] == ["theirs.txt"]


def test_build_only_whose_final_rename_fails_can_be_retried(
    runner: CliRunner,
    world: SimpleNamespace,
    cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claimed directory is given back, and the error is the rename's own."""

    real_rename = Path.rename

    def full(self: Path, target: Path) -> None:
        raise OSError(28, "No space left on device")

    replace_t(world, GROWN)
    monkeypatch.setattr(Path, "rename", full)
    result = rebase(runner, world, world.name, "--no-add")
    # Not `undo`, which would also leave the `cwd` fixture's directory.
    monkeypatch.setattr(Path, "rename", real_rename)
    assert result.exit_code == 1, result.output
    assert "No space left on device" in result.stderr
    assert list((cwd / "builds").iterdir()) == []
    retried = rebase(runner, world, world.name, "--no-add")
    assert retried.exit_code == 3, retried.output


@pytest.mark.parametrize(
    "theirs",
    (pytest.param(("theirs.txt",), id="built"), pytest.param((), id="just-made")),
)
def test_build_only_refuses_a_directory_taken_while_it_built(
    runner: CliRunner,
    world: SimpleNamespace,
    cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
    theirs: tuple[str, ...],
) -> None:
    """Another --no-add or `xorq build` that gets there first wins, even with
    a directory still empty; this one is refused, not nested inside it nor
    renamed over it."""
    real_move = shutil.move

    def move_then_lose_the_race(src: Path, dst: Path) -> object:
        moved = real_move(src, dst)
        other = cwd / "builds" / Path(src).name
        other.mkdir()
        for name in theirs:
            (other / name).write_text("first")
        return moved

    replace_t(world, GROWN)
    monkeypatch.setattr(rebase_module.shutil, "move", move_then_lose_the_race)
    result = rebase(runner, world, world.name, "--no-add")
    # Not `undo`, which would also leave the `cwd` fixture's directory.
    monkeypatch.setattr(rebase_module.shutil, "move", real_move)
    assert result.exit_code == 1, result.output
    assert "exists; --no-add does not replace it" in result.stderr
    (other,) = (cwd / "builds").iterdir()
    assert tuple(p.name for p in other.iterdir()) == theirs


def test_build_only_without_drift_writes_nothing(
    runner: CliRunner, world: SimpleNamespace, cwd: Path
) -> None:
    result = rebase(runner, world, world.name, "--no-add")
    assert result.exit_code == 0, result.output
    assert result.stdout == f"{world.name}\n"
    assert "no drift" in result.stderr
    assert not any(cwd.iterdir())


@pytest.mark.parametrize(
    "kwargs",
    (
        pytest.param({"add": False, "alias": "v2"}, id="alias"),
        pytest.param({"add": False, "move_aliases": True}, id="move-aliases"),
        pytest.param({"add": False, "dry_run": True}, id="dry-run"),
    ),
)
def test_build_only_refuses_contradicting_arguments(
    world: SimpleNamespace, kwargs: dict
) -> None:
    replace_t(world, GROWN)
    with pytest.raises(ValueError):
        rebase_old(world, **kwargs)


def test_a_rebased_entry_shows_its_ancestor(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    replace_t(world, GROWN)
    new = rebase_old(world).new_entry

    shown = runner.invoke(cli, ["--path", world.catalog_path, "show", new.name])
    assert shown.exit_code == 0, shown.output
    assert f"{'Rebased from:':<15} {world.name}" in shown.stdout.splitlines()
    as_json = runner.invoke(
        cli, ["--path", world.catalog_path, "show", new.name, "--json"]
    )
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.stdout)["rebased_from"] == world.name
    # Recorded in the archive, where a catalog copy reads it from.
    with zipfile.ZipFile(new.catalog_path) as zf:
        (member,) = (m for m in zf.namelist() if m.endswith(DumpFiles.build_metadata))
        assert json.loads(zf.read(member))["rebased_from"] == world.name

    old = runner.invoke(cli, ["--path", world.catalog_path, "show", world.name])
    assert "Rebased from:" not in old.stdout
    old_json = runner.invoke(
        cli, ["--path", world.catalog_path, "show", world.name, "--json"]
    )
    assert "rebased_from" not in json.loads(old_json.stdout)
    assert reopen(world).get_catalog_entry(world.name).rebased_from is None


@pytest.mark.parametrize("copy", ("replay", "clone"))
def test_the_ancestor_survives_a_catalog_copy(
    runner: CliRunner, world: SimpleNamespace, tmp_path: Path, copy: str
) -> None:
    replace_t(world, GROWN)
    new = rebase_old(world).new_entry.name
    target = str(tmp_path / "copied")

    if copy == "replay":
        args = ["--path", world.catalog_path, "replay", target]
    else:
        args = ["clone", world.catalog_path, "--path", target]
    copied = runner.invoke(cli, args)
    assert copied.exit_code == 0, copied.output

    entry = Catalog.from_kwargs(path=target, init=False).get_catalog_entry(new)
    assert entry.rebased_from == world.name
    shown = runner.invoke(cli, ["--path", target, "show", new, "--json"])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.stdout)["rebased_from"] == world.name


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


def refuse_options(*args: str) -> Callable:
    """Options that contradict each other, over a drifted source."""

    def setup(w: SimpleNamespace) -> Setup:
        replace_t(w, GROWN)
        return w.name, args, ()

    return setup


NO_ENTRY_TO_NAME = (
    "--no-add takes no --alias, --move-aliases or --only-alias: there is no "
    "entry to name"
)


def refuse_pinned(w: SimpleNamespace) -> Setup:
    path = w.tmp_path / "t.parquet"
    RECORDED.to_pandas().to_parquet(path, index=False)
    t = deferred_read_parquet(path, xo.connect(), table_name="t")
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=w.tmp_path)
    expr = pin_cache(t.filter(t.a > 1).cache(cache=cache), ensure_materialized=True)
    return w.catalog.add(expr).name, (), ()


def test_an_entry_recording_no_python_minor_needs_no_override_without_drift(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is loaded, so no UDF is unpickled under the wrong minor."""
    monkeypatch.setattr(rebase_module, "recorded_python_minor", lambda _: None)
    result = rebase(runner, world)
    assert result.exit_code == 0, result.output
    assert "no drift" in result.stderr


def refuse_no_python_minor(w: SimpleNamespace) -> Setup:
    replace_t(w, GROWN)
    w.monkeypatch.setattr(rebase_module, "recorded_python_minor", lambda _: None)
    return w.name, (), ()


def refuse_python_minor(w: SimpleNamespace) -> Setup:
    replace_t(w, GROWN)
    w.monkeypatch.setattr(rebase_module, "recorded_python_minor", lambda _: (3, 0))
    return w.name, (), ()


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
        refuse_options("--no-add", "-a", "v2"),
        2,
        NO_ENTRY_TO_NAME,
        id="no-add-alias",
    ),
    pytest.param(
        refuse_options("--no-add", "--only-alias", "live"),
        2,
        NO_ENTRY_TO_NAME,
        id="no-add-only-alias",
    ),
    pytest.param(
        refuse_options("--no-add", "--dry-run"),
        2,
        "--dry-run and --no-add are mutually exclusive",
        id="no-add-dry-run",
    ),
    pytest.param(
        refuse_options("--no-add", "--move-aliases"),
        2,
        NO_ENTRY_TO_NAME,
        id="no-add-move-aliases",
    ),
    pytest.param(
        refuse_options("--builds-dir", "out"),
        2,
        "--builds-dir applies only with --no-add",
        id="builds-dir-without-no-add",
    ),
    pytest.param(
        refuse_options("--no-add", "--builds-dir", ""),
        2,
        "--builds-dir names no directory",
        id="empty-builds-dir",
    ),
    pytest.param(
        refuse_alias_add_failing,
        1,
        "non symlink already exists",
        id="alias-add-failing",
    ),
    pytest.param(refuse_pinned, 1, "xorq catalog unpin {name}", id="pinned"),
    pytest.param(
        refuse_no_python_minor,
        1,
        f"records no Python minor, this is {RUNNING}; pass --ignore-venv-mismatch",
        id="no-python-minor",
    ),
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
    "push_fails, raised, hint",
    (
        pytest.param(
            origin_moved_on,
            "CatalogPushError",
            "if the remote moved on, run `xorq catalog sync`; if it rejected the "
            "entry, fix that first",
            id="rejected",
        ),
        pytest.param(
            origin_gone,
            "GitCommandError",
            "; run `xorq catalog sync`",
            id="unreachable",
        ),
    ),
)
def test_a_failed_push_keeps_the_rebase_and_exits_5(
    runner: CliRunner,
    world: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    push_fails: Callable,
    raised: str,
    hint: str,
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
    assert hint in result.stderr
    assert f"Moved alias 'live' -> {new}" in result.stderr
    # Committed locally, not rolled back.
    catalog = reopen(world)
    assert set(catalog.list()) == {world.name, new}
    assert targets(world, "live", "staging") == (new, new)


def test_a_push_that_fails_a_check_says_to_fix_it_before_syncing(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sync runs the same push, so a check that failed would fail it again."""

    def inconsistent(catalog: Catalog) -> None:
        raise AssertionError("catalog inconsistent")

    replace_t(world, GROWN)
    monkeypatch.setattr(Catalog, "push", inconsistent)
    result = rebase(runner, world)
    monkeypatch.undo()
    assert result.exit_code == RebaseExit.PUSH_FAILED, result.output
    assert (
        "push failed: AssertionError: catalog inconsistent; fix that, then run "
        "`xorq catalog sync`" in result.stderr
    )


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
    assert "undoing it failed too (OSError: rollback failed)" in result.stderr
    assert "the local catalog now holds entry" in result.stderr


def test_a_failed_rollback_names_the_alias_it_could_not_restore(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entry's removal took `live` along; putting it back failed. What is
    left is read from the catalog, not from what the write attempted."""
    replace_t(world, GROWN)
    add = CatalogAlias.add
    calls = []

    def failing_add(self: CatalogAlias) -> None:
        calls.append(self.alias)
        # The second move, then the restore of `live`, fail.
        if len(calls) >= 2:
            raise RuntimeError("alias move failed")
        return add(self)

    monkeypatch.setattr(CatalogAlias, "add", failing_add)
    result = rebase(runner, world, world.name, "--move-aliases")
    monkeypatch.undo()
    assert result.exit_code == 1, result.output
    assert f"alias 'live' on no entry, was on {world.name}" in result.stderr
    assert "holds entry" not in result.stderr


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
    "new_hash",
    "build_path",
    "created",
    "output_changes",
    "failing_op",
    "drift",
    "aliases",
    "warnings",
}
ALIAS_KEYS = ("moved", "added", "already", "unmoved", "left")
FAILURE_STATES = {1: "refused", 2: "unreachable", 4: "conflict"}
# A usage error exits before the entry resolves, so it prints no document.
USAGE_ERRORS = frozenset(
    (
        "both-alias-flags",
        "no-add-alias",
        "no-add-only-alias",
        "no-add-dry-run",
        "no-add-move-aliases",
        "builds-dir-without-no-add",
        "empty-builds-dir",
    )
)
JSON_REFUSALS = tuple(p for p in REFUSALS if p.id not in USAGE_ERRORS)
# The refusals that come after the sweep, so their document carries it.
SWEPT_REFUSALS = frozenset(
    (
        "elsewhere-alias-no-drift",
        "taken-alias",
        "unprobed-db",
        "unmatched-source",
        "unreachable-outranks-changed",
        "unreadable-profile",
        "unreadable-bundle",
        "deleted-db",
        # The Python minor is checked just before loading, after the sweep.
        "python-minor",
        "no-python-minor",
        "corrupt-metadata",
        "unreadable-metadata",
    )
)


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
    # A cataloged entry is the hash the rebase derived.
    if document["new_entry"] is not None:
        assert document["new_hash"] == document["new_entry"]
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
        pytest.param(json_unprobed, "unprobed", None, False, id="unprobed"),
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
    request: pytest.FixtureRequest,
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
        # A refusal after the sweep reports it, as `check-sources` would.
        swept = request.node.callspec.id in SWEPT_REFUSALS
        checked = check_sources_entry(runner, world, old_entry) if swept else None
        assert document["drift"] == checked
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
    assert not [w for w in document["warnings"] if w.startswith("LocalReadPathWarning")]


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
    assert not [w for w in caught if w.category is LocalReadPathWarning]


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
    raise RebaseError("no bundle", RebaseFailure.REFUSED)


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
    "args, state, exit_code",
    (
        pytest.param(("--dry-run",), "preview", 3, id="preview"),
        pytest.param(("--no-add",), "built", 3, id="built"),
    ),
)
def test_json_reports_a_rebase_it_did_not_catalog(
    runner: CliRunner,
    world: SimpleNamespace,
    cwd: Path,
    args: tuple[str, ...],
    state: str,
    exit_code: int,
) -> None:
    """No new entry, but the hash it would have, and where --no-add wrote it."""
    replace_t(world, GROWN)
    result, document = rebase_json(runner, world, world.name, *args)
    assert (document["state"], result.exit_code) == (state, exit_code)
    assert document["new_entry"] is None
    assert document["new_hash"] not in (None, world.name)
    built = document["build_path"]
    assert (built is None) == (state == "preview")
    if built is not None:
        assert Path(built) == cwd / "builds" / document["new_hash"]
    assert document["drift"]["state"] == "changed"
    assert reopen(world).list() == [world.name]


def test_json_preview_reports_what_the_rebase_would_do_to_aliases(
    runner: CliRunner, world: SimpleNamespace, cwd: Path
) -> None:
    """The plan the preview checked, as stderr reports it, not empty lists."""
    replace_t(world, GROWN)
    result, document = rebase_json(
        runner, world, world.name, "--dry-run", "--only-alias", "live", "-a", "v2"
    )
    assert (document["state"], result.exit_code) == ("preview", 3)
    assert document["aliases"] == aliases_document(
        moved=["live"], added=["v2"], left=["staging"]
    )
    assert targets(world, "live", "staging") == (world.name, world.name)
    assert "v2" not in reopen(world).list_aliases()


def test_json_puts_what_resolving_the_entry_warned_in_the_document(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The document's scope starts at resolving the entry, not at the rebase."""
    real_get_catalog_entry = catalog_cli._get_catalog_entry

    def warning_get_catalog_entry(*args: object) -> object:
        warnings.warn("resolving knob", FutureWarning, stacklevel=1)
        return real_get_catalog_entry(*args)

    monkeypatch.setattr(catalog_cli, "_get_catalog_entry", warning_get_catalog_entry)
    _, document = rebase_json(runner, world)
    assert "FutureWarning: resolving knob" in document["warnings"]


def test_json_reports_a_document_that_cannot_be_built_as_an_error(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building the document is handled like the rebase: an error, no traceback."""

    def broken_document(*args: object) -> dict:
        raise RuntimeError("document broke")

    monkeypatch.setattr(rebase_module, "rebase_document", broken_document)
    result = rebase(runner, world, world.name, "--json")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "document broke" in result.stderr
    assert isinstance(result.exception, SystemExit)


def test_json_under_pdb_still_reports_a_refusal(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal is no bug to debug, so `--pdb` leaves the document alone."""
    post_mortem = MagicMock()
    monkeypatch.setattr("xorq.cli.pdb.post_mortem", post_mortem)
    name, args, _ = refuse_unknown_alias(world)
    result = runner.invoke(
        top_cli,
        ["--pdb", "catalog", "--path", world.catalog_path, "rebase", name, *args]
        + ["--json"],
    )
    document = json.loads(result.stdout)
    assert (document["state"], result.exit_code) == ("refused", 1)
    assert result.stderr == ""
    assert not post_mortem.called


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


def test_a_rebase_error_names_its_refusal() -> None:
    """Any other error ends a rebase as `failed`; a refusal says which it is."""
    with pytest.raises(ValueError, match="not failed"):
        RebaseError("refused", RebaseFailure.FAILED)


def test_a_rebase_error_keeps_its_sweep_across_processes(
    world: SimpleNamespace,
) -> None:
    """Every field lives in `args`, so a process pool can pickle it."""
    record = rebase_module.read_entry_record(
        world.catalog.get_catalog_entry(world.name)
    )
    reports = tuple(drift.iter_leaf_reports(record))
    error = RebaseError("refused", RebaseFailure.UNREACHABLE).after_sweep(
        record, reports
    )
    restored = pickle.loads(pickle.dumps(error))
    assert (str(restored), restored.failure, restored.exit_code) == (
        "refused",
        RebaseFailure.UNREACHABLE,
        RebaseExit.UNREACHABLE,
    )
    assert restored.reports == reports


def test_every_rebase_state_names_its_exit_code() -> None:
    """Only a completed rebase exits 0; a new state must name its code."""
    assert {state: state.exit_code for state in (*RebaseStatus, *RebaseFailure)} == {
        RebaseStatus.NOOP: 0,
        RebaseStatus.UNPROBED: 0,
        RebaseStatus.REBASED: 0,
        RebaseStatus.CONFLICT: RebaseExit.CONFLICT,
        RebaseStatus.PUSH_FAILED: RebaseExit.PUSH_FAILED,
        RebaseStatus.PREVIEW: RebaseExit.UNCATALOGED,
        RebaseStatus.BUILT: RebaseExit.UNCATALOGED,
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
