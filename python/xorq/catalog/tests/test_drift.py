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
from click.testing import CliRunner

import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.drift import EntryReport, Verdict, iter_leaf_reports


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
    runner: CliRunner, world: SimpleNamespace, tmp_path: Path
) -> None:
    """One changed entry and one unreachable entry exit 3, not 2."""
    other_path = tmp_path / "other.sqlite"
    other_con = SqliteBackend().connect(str(other_path))
    other_con.create_table("u", RECORDED.to_pandas())
    other = world.catalog.add(other_con.table("u"))
    recreate(world, "t", pa.table({"a": pa.array([1], pa.int64())}))
    other_path.write_bytes(b"not a database")

    result = check_sources(runner, world, world.name, other.name)
    assert result.exit_code == 3
    assert "2 entries, 1 drifted" in result.output


def test_reports_stream_per_leaf(world: SimpleNamespace) -> None:
    """The CLI must be able to print a leaf before the next one is probed."""
    record = EntryReport.from_catalog_entry(
        world.catalog.get_catalog_entry(world.name)
    ).record
    reports = iter_leaf_reports(record)
    assert next(reports).verdict == Verdict.EQUAL


def test_entry_report_is_reusable(world: SimpleNamespace) -> None:
    """The public per-entry API the `rebase` work (XOR-453) consumes."""
    report = EntryReport.from_catalog_entry(world.catalog.get_catalog_entry(world.name))
    assert report.name == world.name
    assert report.exit_code == 0
    assert tuple(r.verdict for r in report.leaf_reports) == (Verdict.EQUAL,)
