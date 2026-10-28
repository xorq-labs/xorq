import datetime
import threading
from operator import methodcaller

import pandas as pd
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.caching import (
    ParquetCache,
    ParquetSnapshotCache,
    SourceCache,
    SourceSnapshotCache,
)
from xorq.common.utils.dasher import tokenize
from xorq.common.utils.defer_utils import infer_csv_schema_pandas
from xorq.config import options
from xorq.tests.util import assert_frame_equal


def test_can_connect(sqlite_con):
    assert sqlite_con is not None
    assert sqlite_con.list_tables() is not None


def test_read_record_batch_reader(sqlite_con, astronauts_parquet_path):
    reader = pq.read_table(astronauts_parquet_path).to_reader()
    expr = sqlite_con.read_record_batches(reader).filter(xo._.number == 104)
    assert not expr.execute().empty


def test_can_into_backend(sqlite_con, astronauts_parquet_path):
    astronauts_table = pq.read_table(astronauts_parquet_path)
    reader = astronauts_table.to_reader()
    astronauts = sqlite_con.read_record_batches(reader, table_name="astronauts")
    ddb = xo.duckdb.connect()

    expr = astronauts.join(
        ddb.read_in_memory(
            astronauts_table.to_pandas(), table_name="astronauts"
        ).into_backend(sqlite_con, name="ddb_astronauts"),
        "id",
    ).filter(xo._.number == 104)

    assert not expr.execute().empty


def test_read_parquet(sqlite_con, astronauts_parquet_path):
    t = sqlite_con.read_parquet(astronauts_parquet_path)
    assert not t.execute().empty


def test_can_cache(sqlite_con, astronauts_parquet_path):
    ddb = xo.duckdb.connect()
    astronauts = ddb.read_parquet(astronauts_parquet_path)

    expr = (
        astronauts.cache(SourceCache.from_kwargs(source=sqlite_con))
        .filter(xo._.number == 104)
        .select(xo._.id, xo._.number, xo._.nationwide_number, xo._.name)
    )

    actual = expr.execute()
    expected = (
        astronauts.filter(xo._.number == 104)
        .select(xo._.id, xo._.number, xo._.nationwide_number, xo._.name)
        .execute()
    )

    assert_frame_equal(expected, actual)


def test_can_be_cached(sqlite_con, astronauts_parquet_path):
    astronauts = sqlite_con.read_parquet(astronauts_parquet_path)
    expr = (
        astronauts.cache(SourceCache.from_kwargs(source=xo.duckdb.connect()))
        .filter(xo._.number == 104)
        .select(xo._.id, xo._.number, xo._.nationwide_number, xo._.name)
    )

    actual = expr.execute()
    expected = (
        astronauts.filter(xo._.number == 104)
        .select(xo._.id, xo._.number, xo._.nationwide_number, xo._.name)
        .execute()
    )

    assert_frame_equal(expected, actual)


@pytest.mark.parametrize("collect", ["to_pyarrow", "to_pyarrow_batches", "execute"])
def test_can_collect(sqlite_con, astronauts_parquet_path, collect):
    astronauts = sqlite_con.read_parquet(astronauts_parquet_path)
    expr = (
        astronauts.filter(xo._.number == 104)
        .select(xo._.id, xo._.number, xo._.nationwide_number, xo._.name)
        .mutate(add_1=xo._.number + 1, clean_name=xo._.name.strip())
    )
    assert methodcaller(collect)(expr) is not None


def test_can_outo_backend_and_tokenize(sqlite_con, astronauts_parquet_path):
    ddb = xo.duckdb.connect()

    astronauts = sqlite_con.read_parquet(astronauts_parquet_path)
    expr = (
        astronauts.filter(xo._.number == 104)
        .select(xo._.id, xo._.number, xo._.nationwide_number, xo._.name)
        .mutate(add_1=xo._.number + 1, clean_name=xo._.name.strip())
        .into_backend(ddb, name="ddb_astronauts")
    )

    assert tokenize(expr) is not None
    assert not expr.execute().empty


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_can_deferred_read(sqlite_con, file_format, request):
    read = methodcaller(
        f"deferred_read_{file_format}",
        request.getfixturevalue(f"astronauts_{file_format}_path"),
        con=sqlite_con,
    )(xo)
    assert not read.execute().empty


def test_deferred_read_csv_keeps_schema(astronauts_csv_path, tmp_path):
    con = xo.sqlite.connect(tmp_path / "warehouse.sqlite")
    schema = xo.schema(
        {**infer_csv_schema_pandas(astronauts_csv_path), "id": "float64"}
    )
    xo.deferred_read_csv(
        astronauts_csv_path, con=con, schema=schema, table_name="astronauts"
    ).execute()
    assert con.table("astronauts").schema() == schema


@pytest.mark.parametrize(
    ("database", "kwargs"),
    [
        (None, {}),
        ("file::memory:", {"uri": True}),
        ("file:x?mode=memory", {"uri": True}),
    ],
)
def test_is_in_memory(database, kwargs):
    assert xo.sqlite.connect(database, **kwargs).is_in_memory()


def test_is_in_memory_false_for_file_under_memory_dir(tmp_path):
    path = tmp_path / "memory" / "warehouse.sqlite"
    path.parent.mkdir()
    assert not xo.sqlite.connect(path).is_in_memory()


def test_is_in_memory_from_another_thread():
    # expression hashing asks is_in_memory, and may run off the connecting
    # thread; sqlite3 objects refuse use from any other thread
    con = xo.sqlite.connect()
    result = []
    thread = threading.Thread(target=lambda: result.append(con.is_in_memory()))
    thread.start()
    thread.join()
    assert result == [True]


@pytest.mark.parametrize(
    "read",
    [
        lambda path, con: con.read_csv(path, "astronauts"),
        lambda path, con: xo.deferred_read_csv(path, con=con),
    ],
    ids=["direct", "deferred"],
)
def test_read_csv_on_swapped_default_backend(astronauts_csv_path, monkeypatch, read):
    # the default backend is user-settable; duckdb's read_csv rejects schema=
    # and deferred_read_csv always records one
    monkeypatch.setattr(options, "default_backend", xo.duckdb.connect())
    assert not read(astronauts_csv_path, xo.sqlite.connect()).execute().empty


@pytest.mark.parametrize(
    "read",
    [
        lambda path, con, schema: con.read_csv(path, "padded", schema=schema),
        lambda path, con, schema: xo.deferred_read_csv(
            path, con=con, schema=schema, table_name="padded"
        ),
    ],
    ids=["direct", "deferred"],
)
@pytest.mark.parametrize("in_memory", [True, False], ids=["memory", "file"])
def test_read_csv_schema_on_pandas_default_backend(
    tmp_path, monkeypatch, read, in_memory
):
    # with pandas as the default backend a supplied schema used to be dropped
    # on the way to the reader: ids 001/002 were stored as integers 1/2
    monkeypatch.setattr(options, "default_backend", xo.pandas.connect())
    path = tmp_path / "zero_padded.csv"
    path.write_text("id,region\n001,north\n002,south\n")
    schema = xo.schema({"id": "string", "region": "string"})
    con = xo.sqlite.connect(None if in_memory else tmp_path / "warehouse.sqlite")
    assert read(path, con, schema).execute()["id"].tolist() == ["001", "002"]
    assert con.table("padded").schema() == schema


def test_read_csv_usecols_on_pandas_default_backend(tmp_path):
    # column selection belongs to the reader, not to the ingest
    path = tmp_path / "wide.csv"
    path.write_text("a,b,c\n1,x,7\n")
    schema = xo.schema({"a": "int64", "b": "string"})
    with options({"default_backend": xo.pandas.connect()}):
        con = xo.sqlite.connect()
        direct = con.read_csv(path, "direct", schema=schema, usecols=("a", "b"))
        deferred = xo.deferred_read_csv(
            path, con=con, table_name="deferred", usecols=("a", "b")
        )
        assert direct.execute().columns.tolist() == ["a", "b"]
        assert deferred.schema() == schema
        assert deferred.execute().columns.tolist() == ["a", "b"]


def test_read_csv_header_on_duckdb_default_backend(tmp_path):
    # duckdb's header is a bool and must reach it as one
    path = tmp_path / "numbers.csv"
    path.write_text("n\n2\n10\n")
    with options({"default_backend": xo.duckdb.connect()}):
        t = xo.sqlite.connect().read_csv(path, "numbers", header=True)
    assert t.execute()["n"].tolist() == [2, 10]


def test_read_csv_missing_ints_on_pandas_default_backend(tmp_path, monkeypatch):
    # pandas reads a nullable int schema with its Int64 dtype, which comes
    # back through the arrow metadata as pd.NA; sqlite3 binds only None
    monkeypatch.setattr(options, "default_backend", xo.pandas.connect())
    path = tmp_path / "missing.csv"
    path.write_text("k,n\na,1\nb,\nc,3\n")
    schema = xo.schema({"k": "string", "n": "int64"})
    t = xo.sqlite.connect().read_csv(path, "missing", schema=schema)
    assert t.count().execute() == 3
    assert t.n.sum().execute() == 4
    assert t.filter(t.n.isnull()).k.execute().tolist() == ["b"]


def test_sqlite_snapshot(con_snapshot):
    con_snapshot(xo.sqlite.connect())


def test_cross_source_snapshot(con_cross_source_snapshot):
    con_cross_source_snapshot(xo.sqlite.connect())


@pytest.mark.parametrize(
    "cls",
    [ParquetSnapshotCache, ParquetCache, SourceSnapshotCache, SourceCache],
)
def test_cache_find_backend(cls, con_cache_find_backend):
    con_cache_find_backend(cls, xo.sqlite.connect())


def test_can_train_test_split(sqlite_con, astronauts_parquet_path):
    t = sqlite_con.read_parquet(astronauts_parquet_path)
    (train, test) = xo.train_test_splits(t, 0.2)
    assert train.execute() is not None
    assert test.execute() is not None


def test_time_values_execute(sqlite_con):
    # sqlite hands time back as text; it used to be cast to a duration, which
    # arrow cannot turn into time64, and a time literal came back as Timedelta
    t = sqlite_con.create_table(
        "clocks", pd.DataFrame({"ts": pd.to_datetime(["2024-01-01 12:30:00"])})
    )
    assert t.ts.time().execute().tolist() == [datetime.time(12, 30)]
    assert sqlite_con.execute(xo.literal(datetime.time(12, 30))) == datetime.time(
        12, 30
    )
