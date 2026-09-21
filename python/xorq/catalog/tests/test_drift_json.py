"""`check-sources --json` and the document it emits (xorq-labs/xorq#2298).

One buffered document per sweep, so a consumer parses a complete report or none
at all. The verdicts themselves are covered by `test_drift` and
`test_drift_reads`; what is pinned here is the shape, the roll-ups, and that the
two outputs cannot disagree about an exit code.

sqlite for the same reason as `test_drift`: it survives a build as a real
`DatabaseTable`, where a memory backend's tables are materialized into the
archive and have nothing to drift against.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
from click.testing import CliRunner

import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.catalog import drift
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.drift import record_document, roll_up
from xorq.catalog.enums import Verdict
from xorq.catalog.inspection import BuildRecord


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})

# Every key the document defines, by the level it appears at. The docstring is
# the published shape, so a key added without a line about it fails here rather
# than reaching a consumer undocumented.
DOCUMENTED_KEYS = (
    ("state", "exit_code", "entries"),
    ("leaves", "unchecked", "bundled", "pinned"),
    ("kind", "name", "recorded", "live", "error"),
)


@pytest.fixture
def backend_type() -> str:
    """One content-store backend, not the conftest's three.

    The document is built from a parsed record and never touches the store, so
    the other two parametrizations would buy nothing and cost a full catalog
    each. `test_drift` still sweeps `check-sources` over all three.
    """
    return "git"


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


def recreate(world: SimpleNamespace, new: str, table: pa.Table) -> None:
    """Drop `t`, the table the entry was built over, and put `new` in its place."""
    world.con.drop_table("t", force=True)
    world.con.create_table(new, table.to_pandas())


def check_sources(runner: CliRunner, world: SimpleNamespace, *names: str):
    """One `--json` invocation over `names`, defaulting to the entry itself."""
    return runner.invoke(
        cli,
        [
            "--path",
            world.catalog_path,
            "check-sources",
            *(names or (world.name,)),
            "--json",
        ],
    )


def document(runner: CliRunner, world: SimpleNamespace, *names: str) -> dict:
    """The parsed document, asserting the whole output is that document.

    `json.loads` over the entire stdout is also what pins "nothing else is
    printed": a progress line or a summary would leave trailing text and fail to
    parse.
    """
    result = check_sources(runner, world, *names)
    return json.loads(result.output)


def test_the_root_carries_the_roll_up_beside_the_entries(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    doc = document(runner, world)
    assert tuple(doc) == ("state", "exit_code", "entries")
    assert doc["state"] == Verdict.EQUAL
    assert doc["exit_code"] == 0
    assert tuple(doc["entries"]) == (world.name,)


def test_an_entry_is_keyed_by_the_name_it_was_asked_for(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """An alias is a name a consumer can look back up; the entry name is not."""
    doc = document(runner, world, "live")
    assert tuple(doc["entries"]) == ("live",)


def test_an_equal_leaf_reports_both_schemas_and_no_error(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    (leaf,) = document(runner, world)["entries"][world.name]["leaves"]
    assert leaf == {
        "kind": "DatabaseTable",
        "name": "t",
        "state": Verdict.EQUAL,
        "recorded": {"a": "int64", "b": "string"},
        "live": {"a": "int64", "b": "string"},
    }


def test_a_changed_leaf_reports_the_two_schemas_and_no_delta(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """The delta is set arithmetic on the consumer's side, deliberately unpublished."""
    recreate(world, "t", pa.table({"a": pa.array(["1"], pa.string()), "b": ["x"]}))

    result = check_sources(runner, world)
    doc = json.loads(result.output)
    (leaf,) = doc["entries"][world.name]["leaves"]
    assert result.exit_code == 3
    assert doc["state"] == Verdict.CHANGED
    assert leaf["state"] == Verdict.CHANGED
    assert leaf["recorded"] == {"a": "int64", "b": "string"}
    assert leaf["live"] == {"a": "string", "b": "string"}
    assert "delta" not in result.output


def test_a_missing_table_reports_a_null_live(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    recreate(world, "t_renamed", RECORDED)

    doc = document(runner, world)
    (leaf,) = doc["entries"][world.name]["leaves"]
    assert leaf["state"] == Verdict.TABLE_MISSING
    assert leaf["recorded"] == {"a": "int64", "b": "string"}
    assert leaf["live"] is None
    assert "error" not in leaf


def test_an_unreachable_leaf_reports_a_null_live_and_its_error(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    world.db_path.write_bytes(b"not a database")

    result = check_sources(runner, world)
    doc = json.loads(result.output)
    (leaf,) = doc["entries"][world.name]["leaves"]
    assert result.exit_code == 2
    assert leaf["state"] == Verdict.UNREACHABLE
    assert leaf["live"] is None
    assert leaf["error"]


def test_a_bundled_only_entry_reports_no_leaves_and_its_counts(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    name = world.catalog.add(xo.memtable({"z": [1]})).name

    entry = document(runner, world, name)["entries"][name]
    assert entry["state"] == Verdict.EQUAL
    assert entry["exit_code"] == 0
    assert entry["leaves"] == []
    assert entry["unchecked"] == []
    assert entry["bundled"] == {"memtables": 1}
    assert entry["pinned"] == 0


def test_an_unreadable_entry_carries_its_error_and_no_leaves(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """A record that will not parse is the entry's own defect, not a leaf's."""
    # Unlinked first: under the annex backend the path is a symlink to a
    # read-only object, so writing through it is denied.
    archive = world.catalog.get_catalog_entry(world.name).catalog_path
    archive.unlink()
    archive.write_bytes(b"not a zip")

    result = check_sources(runner, world)
    entry = json.loads(result.output)["entries"][world.name]
    assert result.exit_code == 2
    assert entry["state"] == Verdict.UNREADABLE
    assert entry["exit_code"] == 2
    assert entry["leaves"] == []
    assert entry["error"].startswith("BadZipFile")


def test_the_worst_entry_decides_the_root(
    runner: CliRunner, world: SimpleNamespace, tmp_path: Path
) -> None:
    """A changed entry outranks an unreachable one, at the root as in the exit code."""
    other_path = tmp_path / "other.sqlite"
    other_con = SqliteBackend().connect(str(other_path))
    other_con.create_table("u", RECORDED.to_pandas())
    other = world.catalog.add(other_con.table("u")).name
    recreate(world, "t", pa.table({"a": pa.array([1], pa.int64())}))
    other_path.write_bytes(b"not a database")

    doc = document(runner, world, world.name, other)
    assert doc["state"] == Verdict.CHANGED
    assert doc["exit_code"] == 3
    assert doc["entries"][world.name]["state"] == Verdict.CHANGED
    assert doc["entries"][other]["state"] == Verdict.UNREACHABLE


@pytest.mark.parametrize(
    "break_it",
    [
        lambda world: None,
        lambda world: recreate(world, "t", pa.table({"a": pa.array([1], pa.int64())})),
        lambda world: recreate(world, "t_renamed", RECORDED),
        lambda world: world.db_path.write_bytes(b"not a database"),
    ],
    ids=["equal", "changed", "table-missing", "unreachable"],
)
def test_the_exit_code_matches_the_human_run(
    runner: CliRunner, world: SimpleNamespace, break_it
) -> None:
    """Two renderings of one sweep, so the flag must not change what `$?` says."""
    break_it(world)
    human = runner.invoke(
        cli, ["--path", world.catalog_path, "check-sources", world.name]
    )

    result = check_sources(runner, world)
    assert result.exit_code == human.exit_code
    assert json.loads(result.output)["exit_code"] == human.exit_code


def test_nothing_is_printed_before_the_sweep_finishes(
    runner: CliRunner,
    world: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sweep cut off part way prints nothing, not half a document.

    `KeyboardInterrupt` because it is the one interruption the probe does not
    catch, and it is also the real case: a sweep the user gives up on must not
    leave a truncated document behind for a script to parse.
    """
    other_con = SqliteBackend().connect(str(tmp_path / "other.sqlite"))
    other_con.create_table("u", RECORDED.to_pandas())
    other = world.catalog.add(other_con.table("u")).name
    read = drift.get_table_schema
    probed = []

    def interrupting_read(con, leaf, location):
        probed.append(leaf)
        if len(probed) == 2:
            raise KeyboardInterrupt
        return read(con, leaf, location)

    monkeypatch.setattr(drift, "get_table_schema", interrupting_read)

    result = check_sources(runner, world, world.name, other)
    assert len(probed) == 2
    # click turns the interrupt into its own abort, so the exception is gone by
    # the time the runner sees it. What matters is what reached stdout: no
    # document at all, rather than the first entry's leaves and a truncated tail.
    assert "entries" not in result.output


def test_an_unchecked_leaf_is_named_rather_than_dropped() -> None:
    """An entry that exits 0 while a source went unprobed has to say which one.

    A read whose method has no registered inference is replayed against its
    connection, and on sqlite that replay ingests, so the probe refuses it.
    """
    record = BuildRecord(
        {
            "definitions": {
                "dtypes": {},
                "nodes": {
                    "@read_0": {
                        "op": "Read",
                        "name": "src",
                        "method_name": "read_json",
                        "profile": "p0",
                        "read_kwargs": [
                            ["hash_path", "/data/src.json"],
                            ["table_name", "src"],
                        ],
                        "schema_ref": "schema_0",
                    }
                },
                "schemas": {
                    "schema_0": {
                        "a": {"op": "DataType", "type": "Int64", "nullable": True}
                    }
                },
            },
            "expression": {"node_ref": "@read_0"},
        },
        {"p0": {"con_name": "sqlite"}},
    )

    entry = record_document(record)
    assert entry["state"] == Verdict.EQUAL
    assert entry["exit_code"] == 0
    assert entry["leaves"] == []
    assert entry["unchecked"] == [{"kind": "Read", "name": "/data/src.json"}]


@pytest.mark.parametrize(
    "verdicts, expected",
    [
        ((), Verdict.EQUAL),
        ((Verdict.EQUAL, Verdict.UNREACHABLE), Verdict.UNREACHABLE),
        ((Verdict.CHANGED, Verdict.UNREACHABLE), Verdict.CHANGED),
        ((Verdict.TABLE_MISSING, Verdict.CHANGED), Verdict.TABLE_MISSING),
        ((Verdict.UNREADABLE, Verdict.UNREACHABLE), Verdict.UNREADABLE),
    ],
    ids=["empty", "worst", "worst-across-codes", "tie-3", "tie-2"],
)
def test_the_roll_up_names_a_state_the_exit_code_cannot(
    verdicts: tuple[Verdict, ...], expected: Verdict
) -> None:
    """Two verdicts share each non-zero code, so the state is rolled up over the
    verdicts themselves and a tie keeps the first."""
    assert roll_up(verdicts) is expected


def test_every_document_key_is_documented() -> None:
    """The module docstring is the published shape of the document."""
    for level in DOCUMENTED_KEYS:
        for key in level:
            assert f'"{key}"' in drift.__doc__
