from operator import methodcaller

import dask
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.caching import (
    ParquetCache,
    ParquetSnapshotCache,
    SourceCache,
    SourceSnapshotCache,
)
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
        astronauts.cache(SourceCache(sqlite_con))
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
        astronauts.cache(SourceCache(xo.duckdb.connect()))
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

    assert dask.base.tokenize(expr) is not None
    assert not expr.execute().empty


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_can_deferred_read(sqlite_con, file_format, request):
    read = methodcaller(
        f"deferred_read_{file_format}",
        request.getfixturevalue(f"astronauts_{file_format}_path"),
        con=sqlite_con,
    )(xo)
    assert not read.execute().empty


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
