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
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.caching import ParquetCache
from xorq.catalog import drift
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.drift import drift_document, record_document, roll_up
from xorq.catalog.enums import Verdict
from xorq.catalog.inspection import BuildRecord
from xorq.catalog.zip_utils import write_zip
from xorq.ibis_yaml.enums import DumpFiles
from xorq.vendor.ibis.backends.profiles import Profile


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})

UNEXTRACTABLE_EXPR = """\
definitions:
  dtypes: {}
  schemas: {}
  nodes:
    "@databasetable_0":
      op: DatabaseTable
      profile: p0
expression:
  node_ref: "@databasetable_0"
"""


@pytest.fixture
def backend_type() -> str:
    """One content-store backend, not the conftest's three.

    What this module pins is the document, which is built from the parsed
    record. The one place a sweep does touch the store is the `fetch` inside
    `BuildRecord.from_catalog_entry`, and `test_drift` already sweeps
    `check-sources` over all three backends; running the document tests over
    them too would cost a full catalog each and pin nothing new.
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


def parse(result) -> dict:
    """The document a run printed, asserting the whole of stdout is that document.

    `json.loads` over the entire stream is also what pins "nothing else is
    printed": a progress line or a summary would leave trailing text and fail to
    parse. `result.stdout`, not `result.output`, which since click 8.2
    interleaves stderr -- a document written to the wrong stream is not
    pipeable, and the parse is what has to notice.
    """
    return json.loads(result.stdout)


def document(runner: CliRunner, world: SimpleNamespace, *names: str) -> dict:
    """The parsed document of one `--json` run over `names`."""
    return parse(check_sources(runner, world, *names))


def test_the_root_carries_the_roll_up_beside_the_entries(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    doc = document(runner, world)
    assert tuple(doc) == ("state", "exit_code", "unchecked_count", "entries")
    assert doc["state"] == Verdict.EQUAL
    assert doc["exit_code"] == 0
    assert doc["unchecked_count"] == 0
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
    doc = parse(result)
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
    doc = parse(result)
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
    # `equal`, not the null an entry that compared nothing gets: having no
    # source outside the archive is a reason this one cannot drift, not a
    # reason the question went unanswered.
    assert entry["state"] == Verdict.EQUAL
    assert entry["exit_code"] == 0
    assert entry["leaves"] == []
    assert entry["unchecked"] == []
    assert entry["bundled"] == {"memtables": 1}
    assert entry["pinned"] == 0


@pytest.mark.parametrize(
    "relocate_reads, bundled, pinned",
    [(False, {}, 1), (True, {"reads": 1}, 0)],
    ids=["pinned", "pinned-and-bundled"],
)
def test_a_pinned_leaf_is_counted_in_one_column_only(
    runner: CliRunner,
    world: SimpleNamespace,
    tmp_path: Path,
    relocate_reads: bool,
    bundled: dict,
    pinned: int,
) -> None:
    """The pin's frozen read of its cache artifact, bundled and not.

    One leaf either way, so the two columns must add to one: counting it in
    both would report a source the entry does not have. Which column it lands
    in is `relocate_reads`, the flag that decides whether the artifact's bytes
    were copied into the archive.
    """
    src = tmp_path / "src.parquet"
    pq.write_table(RECORDED, src)
    read = xo.deferred_read_parquet(src, xo.connect(), table_name="src")
    cache = ParquetCache.from_kwargs(source=xo.connect(), base_path=tmp_path / "cache")
    expr = read.filter(read.a > 1).cache(cache=cache).ls.pin(ensure_materialized=True)
    name = world.catalog.add(expr, relocate_reads=relocate_reads).name

    entry = document(runner, world, name)["entries"][name]
    assert entry["state"] == Verdict.EQUAL
    assert entry["bundled"] == bundled
    assert entry["pinned"] == pinned

    human = runner.invoke(cli, ["--path", world.catalog_path, "check-sources", name])
    detail = ", ".join(
        [
            *(f"{count} {label}" for label, count in bundled.items()),
            *([f"{pinned} pinned"] if pinned else []),
        ]
    )
    assert f"no external sources ({detail})" in human.output


def test_an_unreadable_entry_carries_its_error_and_no_leaves(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """A record that will not parse is the entry's own defect, not a leaf's.

    It enumerates nothing either, so the external sources it really has go
    unnamed and ``unchecked_count`` adds nothing for them. What reports the
    entry is its `unreadable` state and the exit code the root takes from it --
    never a green root, so that zero cannot be read as "everything was probed".
    """
    # Unlinked first: this module pins `git`, but under the annex backend the
    # path is a symlink to a read-only object and writing through it is denied,
    # so it is the one pattern that works on every backend.
    archive = world.catalog.get_catalog_entry(world.name).catalog_path
    archive.unlink()
    archive.write_bytes(b"not a zip")

    result = check_sources(runner, world)
    doc = parse(result)
    entry = doc["entries"][world.name]
    assert result.exit_code == 2
    assert entry["state"] == Verdict.UNREADABLE
    assert entry["exit_code"] == 2
    assert entry["leaves"] == []
    assert entry["error"].startswith("BadZipFile")
    assert entry["unchecked"] == []
    assert doc["state"] == Verdict.UNREADABLE
    assert doc["exit_code"] == 2
    assert doc["unchecked_count"] == 0


def test_a_record_whose_leaves_will_not_extract_is_still_one_entry(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """The other half of `unreadable`: an archive that opens but will not parse.

    Every other unreadable case breaks the zip, which fails inside
    `BuildRecord.from_catalog_entry`. This one leaves both members readable and
    breaks leaf extraction, which is a `cached_property`: unforced it would
    first run in `iter_leaf_reports`'s `for` header, outside the per-leaf
    handler, and escape `record_document` and `drift_document` alike -- so the
    sweep would print no document at all, for this entry or any other. That
    forcing is the line `read_record` exists to own.
    """
    archive = world.catalog.get_catalog_entry(world.name).catalog_path
    with zipfile.ZipFile(archive) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    (expr_member,) = (n for n in members if n.endswith(str(DumpFiles.expr)))
    members[expr_member] = UNEXTRACTABLE_EXPR.encode()
    # Unlinked first, for the same reason as the corrupt-archive test above.
    archive.unlink()
    write_zip(archive, members)

    result = check_sources(runner, world)
    entry = parse(result)["entries"][world.name]
    assert result.exit_code == 2
    assert entry["state"] == Verdict.UNREADABLE
    assert entry["leaves"] == []
    assert entry["error"]


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


def test_the_root_state_does_not_depend_on_the_order_of_the_names(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """One `changed` entry and one `table-missing` entry both exit 3, so which
    of the two the root reports is a tie -- and a tie broken by arrival order
    would publish a different state for the same catalog depending on which name
    was typed first."""
    world.con.create_table("u", RECORDED.to_pandas())
    other = world.catalog.add(world.con.table("u")).name
    recreate(world, "t", pa.table({"a": pa.array([1], pa.int64())}))
    world.con.drop_table("u", force=True)

    forwards = document(runner, world, world.name, other)
    backwards = document(runner, world, other, world.name)
    assert forwards["entries"][world.name]["state"] == Verdict.CHANGED
    assert forwards["entries"][other]["state"] == Verdict.TABLE_MISSING
    assert forwards["state"] == backwards["state"] == Verdict.TABLE_MISSING
    assert forwards["exit_code"] == backwards["exit_code"] == 3


def test_a_sweep_that_compared_nothing_reaches_no_verdict(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The root says it too: an entry carrying no verdict is not rolled up, so a
    sweep where none did must not report the `equal` of leaves it never had."""
    monkeypatch.setattr(drift, "read_record", lambda catalog_entry: unchecked_record())

    doc = drift_document(((world.name, world.catalog.get_catalog_entry(world.name)),))
    assert doc["state"] is None
    assert doc["exit_code"] == 0
    assert doc["entries"][world.name]["state"] is None


def test_a_mixed_sweep_keeps_the_verdict_it_reached_and_counts_what_it_did_not(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One entry that compared nothing beside one that came back `equal`.

    The root keeps the `equal`: null is for a sweep that reached no verdict at
    all, and propagating it here would let one unprobed source erase what the
    other entry established. The count is what stops that `equal` from reading
    as "every source was compared", which no entry-level key can say at the
    root.
    """
    world.con.create_table("u", RECORDED.to_pandas())
    other = world.catalog.add(world.con.table("u")).name
    read = drift.read_record

    def read_one_unchecked(catalog_entry):
        if catalog_entry.name == other:
            return unchecked_record()
        return read(catalog_entry)

    monkeypatch.setattr(drift, "read_record", read_one_unchecked)

    doc = drift_document(
        (name, world.catalog.get_catalog_entry(name)) for name in (world.name, other)
    )
    assert doc["entries"][world.name]["state"] == Verdict.EQUAL
    assert doc["entries"][other]["state"] is None
    assert doc["state"] == Verdict.EQUAL
    assert doc["exit_code"] == 0
    assert doc["unchecked_count"] == 1


def test_an_alias_and_its_name_count_their_shared_unprobed_source_twice(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One entry, two keys, and the count added off the keys.

    The document owes a key per requested name, so the entry behind an alias is
    carried twice, and ``unchecked_count`` -- added off the ``unchecked`` lists
    the document already carries -- counts its one unprobed source once per
    key. The count is what a consumer gates on, not what it sizes a fix by: it
    is zero when everything was probed and non-zero when something was not.
    """
    monkeypatch.setattr(drift, "read_record", lambda catalog_entry: unchecked_record())

    doc = document(runner, world, world.name, "live")
    assert tuple(doc["entries"]) == (world.name, "live")
    assert [len(entry["unchecked"]) for entry in doc["entries"].values()] == [1, 1]
    assert doc["unchecked_count"] == 2


def test_a_repeated_name_is_swept_once(
    runner: CliRunner, world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keying by name already collapses a repeat, so probing it twice buys nothing.

    It also costs: the second sweep's verdict would replace the first, so a
    source that changed between the two probes could drop out of the document
    the exit code was owed for.
    """
    read = drift.read_record
    reads = []

    def counting_read(catalog_entry):
        reads.append(catalog_entry)
        return read(catalog_entry)

    monkeypatch.setattr(drift, "read_record", counting_read)

    doc = document(runner, world, world.name, world.name)
    assert len(reads) == 1
    assert tuple(doc["entries"]) == (world.name,)


def test_a_repeated_name_is_swept_once_by_either_rendering(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """The one case where the two renderings could disagree about a repeat.

    Both sweep the same names or neither's exit code is the other's: a source
    that changed between a second probe the human path pays and the JSON path
    does not would answer 0 in one rendering and 3 in the other.
    """
    human = runner.invoke(
        cli,
        ["--path", world.catalog_path, "check-sources", world.name, world.name],
    )
    result = check_sources(runner, world, world.name, world.name)

    assert human.output.count(world.name) == 1
    assert "1 entries, 0 drifted" in human.output
    assert result.exit_code == human.exit_code
    assert tuple(parse(result)["entries"]) == (world.name,)


def test_a_sweep_owns_a_connection_cache_when_it_is_given_none(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller that passes no cache still pays one dial per profile.

    Threading its ``None`` down would hand every entry a private cache, which is
    the per-entry timeout on a dead backend that the cache exists to remove. The
    CLI happens to pass one, so nothing else here would notice.
    """
    world.con.create_table("u", RECORDED.to_pandas())
    other = world.catalog.add(world.con.table("u")).name
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

    # Installed in the body, so the entry the test adds above is not counted.
    monkeypatch.setattr(Profile, "get_con", counting_get_con)

    doc = drift_document(
        (name, world.catalog.get_catalog_entry(name)) for name in (world.name, other)
    )
    assert doc["exit_code"] == 0
    assert len(cons) == 1
    assert len(disconnected) == 1


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
    assert parse(result)["exit_code"] == human.exit_code


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
    assert "entries" not in result.stdout


def unchecked_record() -> BuildRecord:
    """A record whose one external source this command refuses to probe.

    A read whose method has no registered inference is replayed against its
    connection, and on sqlite that replay ingests, so `is_checkable` refuses it.
    """
    return BuildRecord(
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


def test_an_unchecked_leaf_is_named_rather_than_dropped() -> None:
    """An entry that exits 0 while a source went unprobed has to say which one."""
    _, entry = record_document(unchecked_record())
    assert entry["exit_code"] == 0
    assert entry["leaves"] == []
    assert entry["unchecked"] == [{"kind": "Read", "name": "/data/src.json"}]


def test_an_entry_that_compared_nothing_reaches_no_verdict() -> None:
    """`equal` over zero comparisons is the strongest claim on the weakest
    evidence, and a consumer gating on it would go green having checked nothing."""
    verdict, entry = record_document(unchecked_record())
    assert verdict is None
    assert entry["state"] is None


@pytest.mark.parametrize(
    "verdicts, expected",
    [
        ((), Verdict.EQUAL),
        ((Verdict.EQUAL, Verdict.UNREACHABLE), Verdict.UNREACHABLE),
        ((Verdict.CHANGED, Verdict.UNREACHABLE), Verdict.CHANGED),
        ((Verdict.TABLE_MISSING, Verdict.CHANGED), Verdict.TABLE_MISSING),
        ((Verdict.CHANGED, Verdict.TABLE_MISSING), Verdict.TABLE_MISSING),
        ((Verdict.UNREADABLE, Verdict.UNREACHABLE), Verdict.UNREADABLE),
        ((Verdict.UNREACHABLE, Verdict.UNREADABLE), Verdict.UNREADABLE),
    ],
    ids=[
        "empty",
        "worst",
        "worst-across-codes",
        "tie-3",
        "tie-3-reversed",
        "tie-2",
        "tie-2-reversed",
    ],
)
def test_the_roll_up_names_a_state_the_exit_code_cannot(
    verdicts: tuple[Verdict, ...], expected: Verdict
) -> None:
    """Two verdicts share each non-zero code, so the state is rolled up over the
    verdicts themselves -- and on a total order, so each tie answers the same
    whichever way round it arrives."""
    assert roll_up(verdicts) is expected


def test_severity_refines_the_exit_code() -> None:
    """A total order over the verdicts that does not reorder their codes.

    The root reports `state.exit_code` for the whole sweep, so the worst verdict
    and the worst exit code have to be the same leaf. A member declared out of
    place would break that here rather than in a consumer's report.
    """
    by_severity = sorted(Verdict, key=lambda verdict: verdict.severity)
    assert len({verdict.severity for verdict in Verdict}) == len(tuple(Verdict))
    assert [verdict.exit_code for verdict in by_severity] == sorted(
        verdict.exit_code for verdict in Verdict
    )
    assert roll_up(Verdict).exit_code == max(v.exit_code for v in Verdict)


def document_keys(doc: dict) -> dict[str, set[str]]:
    """Every key `doc` defines, by the level it sits at, entry names excluded.

    By level rather than pooled, so a key that moves between levels -- `pinned`
    migrating off the entry and onto a leaf, say -- is a change to the
    published shape here rather than a union that still adds up.
    """
    keys = {"root": set(doc), "entry": set(), "leaf": set()}
    for entry in doc["entries"].values():
        keys["entry"] |= set(entry)
        for leaf in (*entry["leaves"], *entry["unchecked"]):
            keys["leaf"] |= set(leaf)
    return keys


def merge_keys(*keyings: dict[str, set[str]]) -> dict[str, set[str]]:
    """The levels of several documents' keys, unioned level by level.

    The levels are read off the keyings themselves, so a level added to
    `document_keys` is carried through here rather than silently dropped out of
    the comparison.
    """
    return {
        level: set().union(*(keys[level] for keys in keyings)) for level in keyings[0]
    }


def levels_carrying(keys: dict[str, set[str]], name: str) -> set[str]:
    """The levels `name` sits at, for a key that sits at different levels in
    the documents being compared."""
    return {level for level, names in keys.items() if name in names}


def documented_document() -> dict:
    """The example document the module docstring publishes, parsed.

    The docstring's example is the published shape, so it is read as the
    document it claims to be rather than grepped for quoted words: a substring
    test answers yes to a key named anywhere in the prose, and cannot notice a
    key the example still carries that the sweep has stopped emitting.
    """
    match = re.search(r"\n(    \{\n.*?\n    \})\n", drift.__doc__, re.DOTALL)
    assert match is not None, (
        "no example document in the module docstring: expected a block opening "
        "on a line of `    {` and closing on a line of `    }`"
    )
    block = match.group(1)
    # Comments annotate the example and are not part of the document. No `#`
    # appears inside its strings, and one added later fails loudly here rather
    # than silently dropping a key from the comparison.
    return json.loads(re.sub(r"#.*", "", block))


@pytest.mark.skipif(drift.__doc__ is None, reason="docstrings stripped under -OO")
def test_every_document_key_is_documented(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    """The module docstring is the published shape of the document.

    Both directions, over the same reader: a key the sweep emits and the
    docstring does not is undocumented API, and a key the docstring shows and
    the sweep no longer emits is a consumer reading for something that will
    never arrive. Read off documents the sweep really emits rather than a list
    kept beside them, which can only restate what someone already wrote down.
    Three runs, because `error` appears on a leaf and on an entry only when
    each has one.

    Level by level, because a key that moved between levels -- off the entry
    and onto its leaves, say -- keeps a pooled union intact while breaking
    every consumer reading it where it used to sit. `error` is the one key the
    two documents place differently: the example carries it on a leaf and the
    third run carries it on an entry as well, so it is held out of the
    level-by-level comparison and its levels are stated for each side instead,
    rather than demanding an example unreadable entry the docstring documents
    in prose. The example is held to carrying `error` on a leaf and to no level
    the sweep does not emit it at, so it still fails on a move -- `error`
    leaving the leaf, or reaching the root -- while leaving room for an example
    unreadable entry to be added later.
    """
    keys = document_keys(document(runner, world))
    world.db_path.write_bytes(b"not a database")
    keys = merge_keys(keys, document_keys(document(runner, world)))
    archive = world.catalog.get_catalog_entry(world.name).catalog_path
    archive.unlink()
    archive.write_bytes(b"not a zip")
    keys = merge_keys(keys, document_keys(document(runner, world)))
    documented = document_keys(documented_document())

    assert {"state", "exit_code", "entries"} <= keys["root"]
    assert "leaves" in keys["entry"]
    assert levels_carrying(keys, "error") == {"entry", "leaf"}
    assert "leaf" in levels_carrying(documented, "error")
    assert levels_carrying(documented, "error") <= levels_carrying(keys, "error")
    assert {level: names - {"error"} for level, names in keys.items()} == {
        level: names - {"error"} for level, names in documented.items()
    }
