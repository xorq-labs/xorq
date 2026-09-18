"""`xorq catalog check-sources` over a sqlite-backed entry (xorq-labs/xorq#2295).

sqlite because it survives a build as a real `DatabaseTable`: duckdb is a memory
backend, so its tables are materialized into the archive and have nothing to
drift against.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
from attr import evolve
from click.testing import CliRunner

import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.drift import (
    EntryReport,
    Verdict,
    format_unchecked,
    iter_leaf_reports,
    make_profile,
    probe_leaf,
)
from xorq.catalog.enums import LeafKind
from xorq.catalog.inspection import BuildRecord
from xorq.vendor.ibis.backends.profiles import Profile


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})


def recreate(world: SimpleNamespace, name: str, table: pa.Table) -> None:
    """Replace the live table. sqlite has no ALTER COLUMN TYPE, so retype is a
    drop and recreate too."""
    world.con.drop_table("t", force=True)
    world.con.create_table(name, table.to_pandas())


@pytest.fixture
def world(tmp_path: Path, catalog_path: str) -> SimpleNamespace:
    """A sqlite table, and a catalog entry built over it."""
    db_path = tmp_path / "live.sqlite"
    con = SqliteBackend().connect(str(db_path))
    con.create_table("t", RECORDED.to_pandas())
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    t = con.table("t")
    entry = catalog.add(t.filter(t.a > 1), aliases=("live",))
    return SimpleNamespace(
        db_path=db_path,
        con=con,
        catalog=catalog,
        catalog_path=catalog_path,
        name=entry.name,
    )


@pytest.fixture
def add_entry(world: SimpleNamespace):
    """Add a second entry over its own table, on `world`'s database by default."""

    def add(path: Path | None = None, table: str = "u"):
        con = SqliteBackend().connect(str(path or world.db_path))
        con.create_table(table, RECORDED.to_pandas())
        return world.catalog.add(con.table(table))

    return add


@pytest.fixture
def record(world: SimpleNamespace) -> BuildRecord:
    """The parsed build, without probing anything."""
    return BuildRecord.from_catalog_entry(world.catalog.get_catalog_entry(world.name))


def check_sources(runner: CliRunner, world: SimpleNamespace, *names: str):
    return runner.invoke(
        cli, ["--path", world.catalog_path, "check-sources", *(names or (world.name,))]
    )


def test_equal_entry_exits_zero(runner: CliRunner, world: SimpleNamespace) -> None:
    result = check_sources(runner, world)
    assert result.exit_code == 0
    assert "DatabaseTable t: equal" in result.output


def test_an_alias_names_an_entry(runner: CliRunner, world: SimpleNamespace) -> None:
    result = check_sources(runner, world, "live")
    assert result.exit_code == 0
    assert result.output.startswith("live\n")


@pytest.mark.parametrize(
    "live",
    [
        pa.table({"a": pa.array([1], pa.int64()), "b": ["x"], "c": [1.0]}),
        pa.table({"a": pa.array([1], pa.int64())}),
        pa.table({"renamed": pa.array([1], pa.int64()), "b": ["x"]}),
        pa.table({"a": pa.array(["1"], pa.string()), "b": ["x"]}),
    ],
    ids=["added", "dropped", "renamed", "retyped"],
)
def test_column_change_reports_both_schemas(
    runner: CliRunner, world: SimpleNamespace, live: pa.Table
) -> None:
    recreate(world, "t", live)

    result = check_sources(runner, world)
    assert result.exit_code == 3
    assert "DatabaseTable t: changed" in result.output
    assert "recorded: a int64, b string" in result.output
    assert "live:" in result.output


def test_renamed_table_is_missing(runner: CliRunner, world: SimpleNamespace) -> None:
    recreate(world, "t_renamed", RECORDED)

    result = check_sources(runner, world)
    assert result.exit_code == 3
    assert "DatabaseTable t: table-missing" in result.output
    assert "live:     -" in result.output


def test_unreachable_leaf_exits_two_not_one(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """A probe failure must not reach the catalog error handler, which would
    turn every exit code into click's 1."""
    world.db_path.write_bytes(b"not a database")

    result = check_sources(runner, world)
    assert result.exit_code == 2
    assert "DatabaseTable t: unreachable" in result.output


def test_bundled_only_entry_names_its_bundles(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    name = world.catalog.add(xo.memtable({"z": [1]})).name

    result = check_sources(runner, world, name)
    assert result.exit_code == 0
    assert "no external sources (1 memtables)" in result.output


def test_worst_verdict_wins_across_entries(
    runner: CliRunner, world: SimpleNamespace, add_entry, tmp_path: Path
) -> None:
    """One changed entry and one unreachable entry exit 3, not 2."""
    other_path = tmp_path / "other.sqlite"
    other = add_entry(other_path)
    recreate(world, "t", pa.table({"a": pa.array([1], pa.int64())}))
    other_path.write_bytes(b"not a database")

    result = check_sources(runner, world, world.name, other.name)
    assert result.exit_code == 3
    assert "2 entries, 1 drifted" in result.output


def test_reports_stream_per_leaf(record: BuildRecord) -> None:
    """The CLI must be able to print a leaf before the next one is probed."""
    reports = iter_leaf_reports(record)
    assert next(reports).verdict == Verdict.EQUAL


def test_entry_report_is_reusable(world: SimpleNamespace) -> None:
    """The public per-entry API the `rebase` work (XOR-453) consumes."""
    report = EntryReport.from_catalog_entry(world.catalog.get_catalog_entry(world.name))
    assert report.name == world.name
    assert report.exit_code == 0
    assert tuple(r.verdict for r in report.leaf_reports) == (Verdict.EQUAL,)


def test_unhandled_leaf_kind_raises(record: BuildRecord) -> None:
    """A kind with no schema reader must fail loudly, not report unreachable."""
    (leaf,) = record.external_leaves
    read_leaf = evolve(leaf, kind=LeafKind.READ)
    with pytest.raises(ValueError, match="no probe for leaf kind"):
        probe_leaf(read_leaf, record)


def test_leaf_without_a_profile_is_unreachable(record: BuildRecord) -> None:
    """A leaf naming no profile is a record we cannot reach through, not a
    connect against a `None` profile."""
    (leaf,) = record.external_leaves
    report = probe_leaf(evolve(leaf, profile=None), record)
    assert report.verdict is Verdict.UNREACHABLE
    assert "records no profile" in report.error


def test_a_sweep_shares_one_connection_per_profile(
    runner: CliRunner,
    world: SimpleNamespace,
    add_entry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two entries over the same backend cost one connection for the sweep, and
    the sweep closes what it opened."""
    other = add_entry()
    cons, disconnected = [], []
    get_con = Profile.get_con

    def counting_get_con(self, *args, **kwargs):
        con = get_con(self, *args, **kwargs)
        disconnect = con.disconnect

        def counting_disconnect(*a, **kw):
            disconnected.append(con)
            return disconnect(*a, **kw)

        con.disconnect = counting_disconnect
        cons.append(con)
        return con

    monkeypatch.setattr(Profile, "get_con", counting_get_con)

    result = check_sources(runner, world, world.name, other.name)
    assert result.exit_code == 0
    assert len(cons) == 1
    assert len(disconnected) == 1


def test_unchecked_leaves_are_named_beside_the_checked_ones(
    record: BuildRecord,
) -> None:
    """An entry whose other leaves are equal still has to say which external
    leaf nobody probed."""
    (leaf,) = record.external_leaves
    assert format_unchecked(record) is None
    mixed = SimpleNamespace(external_leaves=(leaf, evolve(leaf, kind=LeafKind.READ)))
    assert format_unchecked(mixed) == "  1 external source not checkable (Read)"


def test_an_unchecked_leaf_is_reported_beside_a_checked_one(
    runner: CliRunner, world: SimpleNamespace, tmp_path: Path
) -> None:
    """A mixed entry names the leaf nobody probed and still exits 0 on the one
    that came back equal."""
    csv_path = tmp_path / "side.csv"
    csv_path.write_text("a,c\n1,2\n")
    con = xo.duckdb.connect()
    side = xo.deferred_read_csv(csv_path, con=con)
    t = world.con.table("t").into_backend(con)
    name = world.catalog.add(side.join(t, "a"), relocate_reads=False).name

    result = check_sources(runner, world, name)
    assert result.exit_code == 0
    assert "DatabaseTable t: equal" in result.output
    assert "1 external source not checkable (Read)" in result.output


def test_a_sweep_shares_one_failed_connect_per_profile(
    runner: CliRunner,
    world: SimpleNamespace,
    record: BuildRecord,
    add_entry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead backend behind two leaves costs one connect attempt, and every
    leaf behind it still reports unreachable with the original cause."""
    other = add_entry()
    third = add_entry(tmp_path / "third.sqlite", table="v")
    attempts = []

    def failing_get_con(self, *args, **kwargs):
        attempts.append(self)
        raise RuntimeError("backend is gone")

    monkeypatch.setattr(Profile, "get_con", failing_get_con)

    result = check_sources(runner, world, world.name, other.name, third.name)
    assert result.exit_code == 2
    (leaf,) = record.external_leaves
    hashes = {profile.content_hash for profile in attempts}
    assert len(attempts) == len(hashes) == 2
    assert make_profile(record.get_profile_dict(leaf)).content_hash in hashes
    for table in ("t", "u", "v"):
        assert f"DatabaseTable {table}: unreachable" in result.output
    assert result.output.count("RuntimeError: backend is gone") == 3
