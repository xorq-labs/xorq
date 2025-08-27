import pyarrow.parquet as pq

import xorq.api as xo


def test_can_connect(sqlite_con):
    assert sqlite_con is not None
    assert sqlite_con.list_tables() is not None


def test_read_record_batch_reader(sqlite_con, parquet_dir):
    reader = pq.read_table(parquet_dir / "astronauts.parquet").to_reader()
    expr = sqlite_con.read_record_batches(reader).filter(xo._.number == 104)
    assert not expr.execute().empty
