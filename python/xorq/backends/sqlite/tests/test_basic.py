import pyarrow.parquet as pq

import xorq.api as xo
from xorq.caching import SourceStorage
from xorq.tests.util import assert_frame_equal


def test_can_connect(sqlite_con):
    assert sqlite_con is not None
    assert sqlite_con.list_tables() is not None


def test_read_record_batch_reader(sqlite_con, parquet_dir):
    reader = pq.read_table(parquet_dir / "astronauts.parquet").to_reader()
    expr = sqlite_con.read_record_batches(reader).filter(xo._.number == 104)
    assert not expr.execute().empty


def test_can_into_backend(sqlite_con, parquet_dir):
    astronauts_table = pq.read_table(parquet_dir / "astronauts.parquet")
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


def test_read_parquet(sqlite_con, parquet_dir):
    t = sqlite_con.read_parquet(parquet_dir / "astronauts.parquet")
    assert not t.execute().empty


def test_can_cache(sqlite_con, parquet_dir):
    ddb = xo.duckdb.connect()
    astronauts = ddb.read_parquet(parquet_dir / "astronauts.parquet")

    expr = (
        astronauts.cache(SourceStorage(sqlite_con))
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


def test_can_be_cached(sqlite_con, parquet_dir):
    astronauts = sqlite_con.read_parquet(parquet_dir / "astronauts.parquet")
    expr = (
        astronauts.cache(SourceStorage(xo.duckdb.connect()))
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
