"""`check-sources` over `Read` leaves (xorq-labs/xorq#2296).

`relocate_reads` defaults to True and relocation copies the bytes into the
archive, so the *default* build of a local read is bundled and exempt. What is
left to check is a read built with relocation off, and a read whose path is a
remote URI, which relocation refuses and leaves untouched.

A csv or parquet read is answered by the inference that recorded it, so it is
probed wherever it is bound -- the file is read, the backend is not. A read with
no registered inference is replayed against its connection instead, and there
only a memory backend is probed: everywhere else the backend's own ``read_*``
ingests, so replaying one would write to the user's database rather than read
from it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from attr import evolve
from click.testing import CliRunner

import xorq.api as xo
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.drift import (
    checkable_leaves,
    close_cons,
    get_read_schema,
    get_schema_reader,
    is_checkable,
    iter_leaf_reports,
    path_resolves,
    read_call,
)
from xorq.catalog.enums import Verdict
from xorq.catalog.inspection import BuildRecord, iter_source_leaves


RECORDED = pa.table({"a": pa.array([1, 2], pa.int64()), "b": ["x", "y"]})


def read_doc(hash_path: str, extra: tuple = ()) -> dict:
    """A one-`Read` document, for the shapes a build cannot easily produce."""
    return {
        "definitions": {
            "dtypes": {},
            "nodes": {
                "@read_0": {
                    "op": "Read",
                    "name": "src",
                    "method_name": "read_parquet",
                    "profile": "p0",
                    "read_kwargs": [
                        ["hash_path", hash_path],
                        ["table_name", "src"],
                        *extra,
                    ],
                    "schema_ref": "schema_0",
                }
            },
            "schemas": {
                "schema_0": {"a": {"op": "DataType", "type": "Int64", "nullable": True}}
            },
        },
        "expression": {"node_ref": "@read_0"},
    }


@pytest.fixture
def world(tmp_path: Path, catalog_path: str) -> SimpleNamespace:
    """One parquet file, built into a catalog twice: relocation off, then on."""
    src = tmp_path / "src.parquet"
    pq.write_table(RECORDED, src)
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    read = xo.deferred_read_parquet(src, xo.connect(), table_name="src")
    expr = read.filter(read.a > 1)
    return SimpleNamespace(
        src=src,
        catalog=catalog,
        catalog_path=catalog_path,
        external=catalog.add(expr, relocate_reads=False).name,
        bundled=catalog.add(expr).name,
    )


def check_sources(runner: CliRunner, catalog_path: str, name: str):
    """One `check-sources` invocation, against the catalog `catalog_path` names."""
    return runner.invoke(cli, ["--path", catalog_path, "check-sources", name])


def test_an_unrelocated_read_is_checked_by_path(
    runner: CliRunner, world: SimpleNamespace
) -> None:
    result = check_sources(runner, world.catalog_path, world.external)
    assert result.exit_code == 0
    assert f"Read {world.src}: equal" in result.output


def test_a_moved_file_is_missing(runner: CliRunner, world: SimpleNamespace) -> None:
    world.src.rename(world.src.with_name("moved.parquet"))

    result = check_sources(runner, world.catalog_path, world.external)
    assert result.exit_code == 3
    assert f"Read {world.src}: table-missing" in result.output
    assert "live:     -" in result.output


def test_an_added_column_is_changed(runner: CliRunner, world: SimpleNamespace) -> None:
    pq.write_table(RECORDED.append_column("c", pa.array([1.0, 2.0])), world.src)

    result = check_sources(runner, world.catalog_path, world.external)
    assert result.exit_code == 3
    assert "changed" in result.output
    assert "live:     a int64, b string, c float64" in result.output


def test_a_relocated_read_is_exempt(runner: CliRunner, world: SimpleNamespace) -> None:
    """Its bytes are in the archive, so the source file no longer speaks for it."""
    pq.write_table(RECORDED.append_column("c", pa.array([1.0, 2.0])), world.src)

    result = check_sources(runner, world.catalog_path, world.bundled)
    assert result.exit_code == 0
    assert "no external sources (1 reads)" in result.output


def test_a_remote_uri_is_checkable() -> None:
    """Relocation refuses remote schemes, so the real URI survives the build."""
    record = BuildRecord(read_doc("s3://bucket/src.parquet"), {})
    (leaf,) = checkable_leaves(record)
    assert leaf.name == "s3://bucket/src.parquet"


def test_the_relocation_keys_are_not_replayed() -> None:
    """`READ_EXCLUDE_KEYS`, not a second spelling of the same list."""
    (leaf,) = iter_source_leaves(
        read_doc(
            "/data/src.parquet",
            extra=(["read_path", "reads/abc.parquet"], ["relocatable", True]),
        )
    )
    assert read_call(leaf) == (("/data/src.parquet",), {"table_name": "src"})


def test_probing_a_parquet_read_dials_nothing(world: SimpleNamespace) -> None:
    """The inference reads the file, so the recorded backend is never opened.

    Asserted on the sweep's own cache: nothing was dialled, so nothing was
    registered anywhere either, and a profile that can no longer be opened
    cannot turn an answerable file into `unreachable`.
    """
    record = BuildRecord.from_catalog_entry(
        world.catalog.get_catalog_entry(world.external)
    )
    con_cache: dict = {}
    try:
        (report,) = iter_leaf_reports(record, con_cache)
        assert report.verdict == Verdict.EQUAL
        assert con_cache == {}
    finally:
        close_cons(con_cache)


def test_the_replay_arm_drops_the_destination_and_the_record(tmp_path: Path) -> None:
    """What a read with no registered inference is asked, exactly.

    The recorded ``table_name`` is a destination in the live catalog, and taking
    it would shadow whatever already holds it for every later leaf of the sweep,
    starting with `get_table_schema`'s `list_tables` check. The recorded schema
    is an instruction, and replaying it would derive `live` from `recorded`.
    """
    calls = []

    class Con:
        def read_json(self, *args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(schema=lambda: xo.schema({"a": "int64"}))

    src = tmp_path / "src.json"
    src.write_text("{}\n")
    (leaf,) = iter_source_leaves(
        read_doc(str(src), extra=(["schema", {"a": "int64"}],))
    )
    schema = get_read_schema(Con(), evolve(leaf, method_name="read_json"), None)

    assert schema == xo.schema({"a": "int64"})
    assert calls == [((str(src),), {})]


def test_a_read_on_an_ingesting_backend_is_checked_without_writing(
    runner: CliRunner, tmp_path: Path, catalog_path: str
) -> None:
    """sqlite's `read_parquet` ingests, but the probe never runs it.

    The recorded schema came from datafusion's inference, so the probe re-reads
    the file with datafusion and the sqlite connection has no part in the
    answer -- including the `mode="replace"` riding along in the recorded
    kwargs, which a replay would have used to drop whatever now holds the
    recorded table name. The leaf is checked and the database is left exactly
    as it was.
    """
    src = tmp_path / "src.parquet"
    pq.write_table(RECORDED, src)
    db_path = tmp_path / "live.sqlite"
    con = xo.sqlite.connect(str(db_path))
    read = xo.deferred_read_parquet(src, con, table_name="precious")
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    name = catalog.add(read.filter(read.a > 1), relocate_reads=False).name
    con.create_table("precious", pa.table({"keep": ["me"]}))
    before = con.table("precious").to_pyarrow()

    result = check_sources(runner, catalog_path, name)
    assert result.exit_code == 0
    assert f"Read {src}: equal" in result.output
    assert xo.sqlite.connect(str(db_path)).table("precious").to_pyarrow() == before


def test_a_read_with_no_inference_on_an_ingesting_backend_is_unchecked() -> None:
    """The replay arm is what the memory-backend test still guards.

    A method `get_read_inference` does not answer for is replayed against the
    connection it recorded, and on sqlite that replay ingests.
    """
    record = BuildRecord(read_doc("/data/src.parquet"), {"p0": {"con_name": "sqlite"}})
    (leaf,) = record.external_leaves
    assert is_checkable(leaf, record)
    assert not is_checkable(evolve(leaf, method_name="read_json"), record)


def test_a_parquet_read_outlives_its_profile(
    runner: CliRunner, tmp_path: Path, catalog_path: str
) -> None:
    """The file answers, so an unopenable backend is not evidence about it.

    The recorded duckdb database is gone, which `connect` reports as a
    `FileNotFoundError` rather than creating. Opening it before dispatching
    would rank an intact source `unreachable` on a question the file settles.
    """
    src = tmp_path / "src.parquet"
    pq.write_table(RECORDED, src)
    db_path = tmp_path / "live.ddb"
    con = xo.duckdb.connect(str(db_path))
    read = xo.deferred_read_parquet(src, con, table_name="src")
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    name = catalog.add(read.filter(read.a > 1), relocate_reads=False).name
    con.disconnect()
    db_path.unlink()

    result = check_sources(runner, catalog_path, name)
    assert result.exit_code == 0
    assert f"Read {src}: equal" in result.output


def test_a_glob_read_is_checked_by_path(
    runner: CliRunner, tmp_path: Path, catalog_path: str
) -> None:
    """A glob is a path the read resolves, not a file that is not there."""
    parts = tmp_path / "parts"
    parts.mkdir()
    for part in ("a", "b"):
        pq.write_table(RECORDED, parts / f"{part}.parquet")
    pattern = str(parts / "*.parquet")
    con = xo.connect()
    read = xo.deferred_read_parquet(pattern, con, table_name="parts")
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    name = catalog.add(read.filter(read.a > 1), relocate_reads=False).name

    result = check_sources(runner, catalog_path, name)
    assert result.exit_code == 0
    assert f"Read {pattern}: equal" in result.output


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("s3://b/x.parquet", id="s3"),
        pytest.param("az://c/x.parquet", id="az"),
        pytest.param("abfs://c/x.parquet", id="abfs"),
        pytest.param("hf://d/x.parquet", id="hf"),
    ],
)
def test_a_uri_is_left_to_the_read(path: str) -> None:
    """Absence is not evidence a URI scheme this version has not heard of.

    `REMOTE_SCHEMES` names what relocation refuses, not what a filesystem can
    answer for; resolving through the read's own helper keeps a new scheme from
    being reported as a missing source.
    """
    assert path_resolves(path)


def test_a_node_naming_a_non_read_method_raises() -> None:
    """The archive names the method the probe dispatches on, so it is checked."""
    (leaf,) = iter_source_leaves(read_doc("/data/src.parquet"))
    with pytest.raises(ValueError, match="names no read method"):
        get_schema_reader(evolve(leaf, method_name="drop_table"))


def test_a_read_without_a_path_raises() -> None:
    """A record defect, resolved before a connection is opened.

    `from_node_def` tolerates the shape; there is nothing to replay for it, and
    reaching the probe would rank it an unreachable backend.
    """
    (leaf,) = iter_source_leaves(read_doc("/data/src.parquet"))
    with pytest.raises(ValueError, match="records no read path"):
        get_schema_reader(evolve(leaf, read_kwargs=(("table_name", "src"),)))


@pytest.mark.parametrize(
    ("con_name", "key"),
    [
        pytest.param("duckdb", "columns", id="duckdb"),
        pytest.param(None, "schema", id="datafusion"),
    ],
)
def test_the_recorded_schema_is_not_replayed(
    runner: CliRunner, tmp_path: Path, catalog_path: str, con_name: str | None, key: str
) -> None:
    """`deferred_read_csv` records the schema it read with, under one of two
    spellings. Replaying it makes the read answer with what it was told, so
    `live` would come from `recorded` and drift could only ever be `equal`.
    """
    csv_path = tmp_path / "src.csv"
    csv_path.write_text("a,b\n1,x\n2,y\n")
    con = xo.duckdb.connect() if con_name == "duckdb" else xo.connect()
    read = xo.deferred_read_csv(csv_path, con, table_name="src")
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    name = catalog.add(read.filter(read.a > 1), relocate_reads=False).name
    (leaf,) = BuildRecord.from_catalog_entry(
        catalog.get_catalog_entry(name)
    ).external_leaves
    assert key in dict(leaf.read_kwargs)

    # Unchanged: the serialized schema is a plain mapping, and handing one back
    # to the read raises -- which would rank an intact source `unreachable`.
    assert check_sources(runner, catalog_path, name).exit_code == 0

    csv_path.write_text("a,b,c\n1,x,9\n2,y,8\n")
    result = check_sources(runner, catalog_path, name)
    assert result.exit_code == 3
    assert "live:     a int64, b string, c int64" in result.output


def test_a_date_column_is_not_drift(
    runner: CliRunner, tmp_path: Path, catalog_path: str
) -> None:
    """`recorded` is pandas' inference, so `live` has to be pandas' too.

    duckdb and datafusion read `2024-01-01` as a date, and an all-empty column
    as text, where pandas leaves the first a string and the second a float.
    Probing with the backend's own read would compare two inference engines and
    report `changed` for a file nobody touched.
    """
    csv_path = tmp_path / "src.csv"
    csv_path.write_text("d,b,e\n2024-01-01,x,\n2024-01-02,y,\n")
    con = xo.duckdb.connect()
    read = xo.deferred_read_csv(csv_path, con, table_name="src")
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    name = catalog.add(read.filter(read.b == "x"), relocate_reads=False).name
    (leaf,) = BuildRecord.from_catalog_entry(
        catalog.get_catalog_entry(name)
    ).external_leaves
    assert leaf.recorded == xo.schema({"d": "string", "b": "string", "e": "float64"})
    # The premise: the two engines really do disagree about this file.
    assert con.read_csv(csv_path).schema() != leaf.recorded

    result = check_sources(runner, catalog_path, name)
    assert result.exit_code == 0
    assert "equal" in result.output


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_a_declared_schema_is_compared_against_inference(
    runner: CliRunner, tmp_path: Path, catalog_path: str, suffix: str
) -> None:
    """The known limitation, pinned rather than discovered.

    A `schema=` handed to either deferred read is recorded exactly the way an
    inferred one is, and nothing in the archive says which it was. The probe
    reports the file's own inference against it, so an override that disagreed
    with inference at build time reads as `changed` on an untouched file.
    """
    declared = xo.schema({"a": "string", "b": "string"})
    src = tmp_path / f"src.{suffix}"
    if suffix == "csv":
        src.write_text("a,b\n1,x\n2,y\n")
        read = xo.deferred_read_csv(
            src, xo.connect(), table_name="src", schema=declared
        )
    else:
        pq.write_table(RECORDED, src)
        read = xo.deferred_read_parquet(
            src, xo.connect(), table_name="src", schema=declared
        )
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    name = catalog.add(read.filter(read.b == "x"), relocate_reads=False).name

    result = check_sources(runner, catalog_path, name)
    assert result.exit_code == 3
    assert "recorded: a string, b string" in result.output
    assert "live:     a int64, b string" in result.output


def test_a_float16_parquet_column_is_not_drift(
    runner: CliRunner, tmp_path: Path, catalog_path: str
) -> None:
    """`recorded` is datafusion's inference for parquet, whatever `con` is.

    `deferred_read_parquet` derives the node schema from a datafusion session
    no matter which backend the read is bound to, so the probe has to re-read
    with datafusion too. duckdb maps a parquet `float16` to `float32` and drops
    the millisecond timestamp precision, so probing with the bound backend
    would report `changed` for a file nobody touched.
    """
    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table(
            {
                "f": pa.array([1.0, 2.0], pa.float16()),
                "t": pa.array([0, 1], pa.timestamp("ms")),
            }
        ),
        src,
    )
    con = xo.duckdb.connect()
    read = xo.deferred_read_parquet(src, con, table_name="src")
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    name = catalog.add(read.filter(read.f > 1), relocate_reads=False).name
    (leaf,) = BuildRecord.from_catalog_entry(
        catalog.get_catalog_entry(name)
    ).external_leaves
    assert leaf.recorded == xo.schema({"f": "float16", "t": "timestamp(3)"})
    # The premise: the two engines really do disagree about this file.
    assert con.read_parquet(src).schema() != leaf.recorded

    result = check_sources(runner, catalog_path, name)
    assert result.exit_code == 0
    assert "equal" in result.output
