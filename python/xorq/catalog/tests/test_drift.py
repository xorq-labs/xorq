"""`xorq catalog check-sources` over a sqlite-backed entry (xorq-labs/xorq#2295).

sqlite because it survives a build as a real `DatabaseTable`: duckdb is a memory
backend, so its tables are materialized into the archive and have nothing to
drift against.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest
from attr import evolve
from click.testing import CliRunner

import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.catalog import drift
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.drift import (
    EntryReport,
    checkable_leaves,
    format_unchecked,
    get_schema_reader,
    iter_leaf_reports,
    make_profile,
    probe_leaf,
)
from xorq.catalog.enums import Verdict
from xorq.catalog.inspection import BuildRecord
from xorq.vendor.ibis.backends.profiles import Profile


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})


def recreate(world: SimpleNamespace, new: str, table: pa.Table) -> None:
    """Drop `t`, the table the entry was built over, and put `new` in its place.
    sqlite has no ALTER COLUMN TYPE, so retype is a drop and recreate too."""
    world.con.drop_table("t", force=True)
    world.con.create_table(new, table.to_pandas())


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


def test_reports_stream_per_leaf(
    world: SimpleNamespace,
    add_entry: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI must be able to print a leaf before the next one is probed.

    Two leaves, because with one there is no next probe to not have run yet and
    a tuple would pass just as well.
    """
    add_entry()
    t, u = world.con.table("t"), world.con.table("u")
    entry = world.catalog.add(t.join(u, "a"))
    record = BuildRecord.from_catalog_entry(world.catalog.get_catalog_entry(entry.name))
    assert len(checkable_leaves(record)) == 2
    probed = []
    read = drift.get_table_schema

    def counting_read(con, leaf, location):
        probed.append(leaf)
        return read(con, leaf, location)

    monkeypatch.setattr(drift, "get_table_schema", counting_read)

    reports = iter_leaf_reports(record)
    assert next(reports).verdict == Verdict.EQUAL
    assert len(probed) == 1


def test_entry_report_is_reusable(world: SimpleNamespace) -> None:
    """The public per-entry API the `rebase` work (XOR-453) consumes."""
    report = EntryReport.from_catalog_entry(world.catalog.get_catalog_entry(world.name))
    assert report.name == world.name
    assert report.exit_code == 0
    assert tuple(r.verdict for r in report.leaf_reports) == (Verdict.EQUAL,)


def test_unhandled_leaf_kind_raises() -> None:
    """A kind with no schema reader must fail loudly, not report unreachable.

    Asserted on a stand-in: every `LeafKind` now has a reader, and `SourceLeaf`
    validates its kind against that enum, so no leaf can carry an unhandled one.
    """
    with pytest.raises(ValueError, match="no probe for leaf kind"):
        get_schema_reader(SimpleNamespace(kind="Bogus"))


def test_leaf_without_a_profile_is_unreadable(record: BuildRecord) -> None:
    """A leaf naming no profile is a defect in what was read, not a backend we
    failed to reach: nothing was connected to."""
    (leaf,) = record.external_leaves
    with pytest.raises(ValueError, match="records no profile"):
        probe_leaf(evolve(leaf, profile=None), record)

    (report,) = iter_leaf_reports(evolve(record, profiles={}))
    assert report.verdict is Verdict.UNREADABLE
    assert "does not hold" in report.error


@pytest.fixture
def count_cons(monkeypatch: pytest.MonkeyPatch):
    """Start counting connections, and hand back the `(opened, disconnected)`
    lists they land in. Called from the body so that setup a test does first
    does not count."""
    get_con = Profile.get_con

    def install() -> tuple[list, list]:
        opened, disconnected = [], []

        def counting_get_con(self, *args, **kwargs):
            con = get_con(self, *args, **kwargs)
            disconnect = con.disconnect

            def counting_disconnect(*a, **kw):
                disconnected.append(con)
                return disconnect(*a, **kw)

            con.disconnect = counting_disconnect
            opened.append(con)
            return con

        monkeypatch.setattr(Profile, "get_con", counting_get_con)
        return opened, disconnected

    return install


def test_a_sweep_shares_one_connection_per_profile(
    runner: CliRunner,
    world: SimpleNamespace,
    add_entry,
    count_cons,
) -> None:
    """Two entries over the same backend cost one connection for the sweep, and
    the sweep closes what it opened."""
    other = add_entry()
    cons, disconnected = count_cons()

    result = check_sources(runner, world, world.name, other.name)
    assert result.exit_code == 0
    assert len(cons) == 1
    assert len(disconnected) == 1


def test_unchecked_leaves_are_named_beside_the_checked_ones(
    record: BuildRecord,
) -> None:
    """An entry whose other leaves are equal still has to say which external
    leaf nobody probed.

    Stand-in kind for the same reason as ``test_unhandled_leaf_kind_raises``:
    both `LeafKind` members are checkable, so nothing real is left out.
    """
    (leaf,) = record.external_leaves
    assert format_unchecked(record) is None
    mixed = SimpleNamespace(external_leaves=(leaf, SimpleNamespace(kind="Bogus")))
    assert format_unchecked(mixed) == "  1 external source not checkable (Bogus)"


def test_a_read_and_a_table_are_checked_side_by_side(
    runner: CliRunner, world: SimpleNamespace, tmp_path: Path
) -> None:
    """A mixed entry probes both kinds, and exits 0 when both came back equal."""
    csv_path = tmp_path / "side.csv"
    csv_path.write_text("a,c\n1,2\n")
    con = xo.duckdb.connect()
    side = xo.deferred_read_csv(csv_path, con=con)
    t = world.con.table("t").into_backend(con)
    name = world.catalog.add(side.join(t, "a"), relocate_reads=False).name

    result = check_sources(runner, world, name)
    assert result.exit_code == 0
    assert "DatabaseTable t: equal" in result.output
    assert f"Read {csv_path}: equal" in result.output


def test_a_sweep_shares_one_failed_connect_per_profile(
    runner: CliRunner,
    world: SimpleNamespace,
    record: BuildRecord,
    add_entry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two leaves behind one dead backend cost one connect attempt, a third
    behind a second dead backend costs another, and every leaf still reports
    unreachable with the original cause."""
    other = add_entry()
    third = add_entry(tmp_path / "third.sqlite", table="v")
    third_record = BuildRecord.from_catalog_entry(
        world.catalog.get_catalog_entry(third.name)
    )
    attempts = []

    def failing_get_con(self, *args, **kwargs):
        attempts.append(self)
        raise RuntimeError("backend is gone")

    monkeypatch.setattr(Profile, "get_con", failing_get_con)

    result = check_sources(runner, world, world.name, other.name, third.name)
    assert result.exit_code == 2
    hashes = {profile.content_hash for profile in attempts}
    assert len(attempts) == 2
    assert hashes == {
        make_profile(r.get_profile_dict(leaf)).content_hash
        for r in (record, third_record)
        for leaf in checkable_leaves(r)
    }
    for table in ("t", "u", "v"):
        assert f"DatabaseTable {table}: unreachable" in result.output
    assert result.output.count("RuntimeError: backend is gone") == 3


def test_a_namespace_reaches_the_reader_as_ibis_spells_it(
    record: BuildRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database alone stays a bare name; a full pair stays a pair, because
    ibis reads a lone string as a database."""
    (leaf,) = record.external_leaves
    seen = []

    def recording_list_tables(self, *args, database=None, **kwargs):
        seen.append(database)
        return []

    monkeypatch.setattr(SqliteBackend, "list_tables", recording_list_tables)

    for namespace in ((None, "main"), ("cat", "main")):
        report = probe_leaf(evolve(leaf, namespace=namespace), record)
        assert report.verdict is Verdict.TABLE_MISSING
    assert seen == ["main", ("cat", "main")]


def test_a_pair_namespace_is_what_duckdb_takes() -> None:
    """The pair shape the probe passes through is pinned to a real backend's
    contract, not to a stub."""
    con = xo.duckdb.connect()
    try:
        assert con.list_tables(database=("memory", "main")) == []
    finally:
        con.disconnect()


def test_a_pair_a_backend_cannot_express_comes_back_unreachable(
    record: BuildRecord,
) -> None:
    """The pair is not a contract every backend honours: sqlite is reachable as
    recorded, so only the pair breaks it, and it breaks at the read."""
    (leaf,) = record.external_leaves
    assert probe_leaf(leaf, record).verdict is Verdict.EQUAL
    report = probe_leaf(evolve(leaf, namespace=("cat", "main")), record)
    assert report.verdict is Verdict.UNREACHABLE
    assert "OperationalError" in report.error


def test_a_catalog_without_a_database_raises(record: BuildRecord) -> None:
    """A malformed namespace is a property of the leaf, so it must not be
    laundered into an unreachable backend."""
    (leaf,) = record.external_leaves
    with pytest.raises(ValueError, match="catalog 'cat' without a database"):
        probe_leaf(evolve(leaf, namespace=("cat", None)), record)


def test_a_lone_probe_closes_the_connection_it_opened(
    record: BuildRecord, count_cons
) -> None:
    """With no caller-owned cache, `probe_leaf` owns the connection it opened."""
    cons, disconnected = count_cons()

    (leaf,) = record.external_leaves
    assert probe_leaf(leaf, record).verdict is Verdict.EQUAL
    assert len(cons) == 1
    assert len(disconnected) == 1


def test_an_unreadable_archive_is_unreadable_beside_a_healthy_entry(
    runner: CliRunner, world: SimpleNamespace, add_entry: Callable[..., Any]
) -> None:
    """A record that will not parse ranks as unreadable for its own entry and
    leaves the rest of the sweep to report normally."""
    other = add_entry()
    # Unlinked first: under the annex backend the path is a symlink to a
    # read-only object, so writing through it is denied.
    archive = world.catalog.get_catalog_entry(other.name).catalog_path
    archive.unlink()
    archive.write_bytes(b"not a zip")

    result = check_sources(runner, world, world.name, other.name)
    assert result.exit_code == 2
    assert "DatabaseTable t: equal" in result.output
    assert "  unreadable: BadZipFile" in result.output


def test_a_leaf_defect_leaves_the_other_leaves_probed(
    runner: CliRunner,
    world: SimpleNamespace,
    add_entry: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leaf that raises is named as unreadable and stops nothing else: the
    leaves after it are still probed, and the drift among them still wins the
    exit code."""
    add_entry()
    t, u = world.con.table("t"), world.con.table("u")
    entry = world.catalog.add(t.join(u, "a"))
    catalog_entry = world.catalog.get_catalog_entry(entry.name)
    first, second = checkable_leaves(BuildRecord.from_catalog_entry(catalog_entry))
    world.con.drop_table(second.table, force=True)
    world.con.create_table(
        second.table, pa.table({"a": pa.array([1], pa.int64())}).to_pandas()
    )
    table_location = drift.table_location

    def raising_location(leaf):
        if leaf.table == first.table:
            raise ValueError("catalog 'cat' without a database")
        return table_location(leaf)

    monkeypatch.setattr(drift, "table_location", raising_location)

    result = check_sources(runner, world, entry.name)
    assert result.exit_code == 3
    assert f"DatabaseTable {first.name}: unreadable" in result.output
    assert "    ValueError: catalog 'cat' without a database" in result.output
    assert f"DatabaseTable {second.name}: changed" in result.output
