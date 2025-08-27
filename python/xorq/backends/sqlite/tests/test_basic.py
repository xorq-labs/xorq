import pyarrow.parquet as pq

import xorq.api as xo


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


def test_can_cache(sqlite_con):
    pass
