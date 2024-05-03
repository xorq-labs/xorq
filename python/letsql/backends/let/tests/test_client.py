import pyarrow as pa

import letsql as ls
from letsql.backends.let.tests.conftest import assert_frame_equal


def test_register_record_batch_reader(alltypes_df):
    con = ls.connect()
    t = con.register(alltypes_df, "alltypes")
    record_batch_reader = t.to_pyarrow_batches()
    t2 = con.register(record_batch_reader, "t2")
    actual = t2.execute()

    assert isinstance(record_batch_reader, pa.RecordBatchReader)
    assert_frame_equal(actual, alltypes_df)


def test_register_expr(alltypes_df):
    con = ls.connect()
    t = con.register(alltypes_df, "alltypes")
    t2 = con.register(t, "alltypes2")
    actual = t2.execute()
    assert_frame_equal(actual, alltypes_df)


def test_register_record_batch_reader_with_filter(alltypes, alltypes_df):
    con = ls.connect()
    t = con.register(alltypes_df, "alltypes")
    record_batch_reader = t.to_pyarrow_batches()
    t2 = con.register(record_batch_reader, "t2")
    cols_a = [ca for ca in alltypes.columns.copy() if ca != "timestamp_col"]
    expr = t2.filter((t2.id >= 5200) & (t2.id <= 5210))[cols_a]
    expr.execute()
