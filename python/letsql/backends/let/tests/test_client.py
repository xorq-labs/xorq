import ibis
import pyarrow as pa

import letsql as ls
from letsql.tests.util import (
    assert_frame_equal,
)


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


def test_execute_nonnull():
    schema = pa.schema((pa.field("x", pa.int64(), nullable=False),))
    tab = pa.table({"x": [1, 2, 3]}, schema=schema)
    con = ls.connect()
    t = con.register(tab, "my_table")
    t.execute()


def test_register_record_batch_reader_with_filter(alltypes, alltypes_df):
    con = ls.connect()
    t = con.register(alltypes_df, "alltypes")
    record_batch_reader = t.to_pyarrow_batches()
    t2 = con.register(record_batch_reader, "t2")
    cols_a = [ca for ca in alltypes.columns.copy() if ca != "timestamp_col"]
    expr = t2.filter((t2.id >= 5200) & (t2.id <= 5210))[cols_a]
    expr.execute()


def test_create_table(con):
    import pandas as pd

    con.create_table("UPPERCASE", schema=ibis.schema({"A": "int"}))
    con.create_table("name", pd.DataFrame({"a": [1]}))


def test_register_table_with_uppercase(con):
    db_con = ibis.duckdb.connect()
    db_t = db_con.create_table("lowercase", schema=ibis.schema({"A": "int"}))

    uppercase_table_name = "UPPERCASE"
    t = con.register(db_t, uppercase_table_name)
    assert uppercase_table_name in con.list_tables()
    assert t.execute() is not None


def test_register_table_with_uppercase_multiple_times(con):
    db_con = ibis.duckdb.connect()
    db_t = db_con.create_table("lowercase", schema=ibis.schema({"A": "int"}))

    uppercase_table_name = "UPPERCASE"
    con.register(db_t, uppercase_table_name)

    expected_schema = ibis.schema({"B": "int"})
    db_t = db_con.create_table("lowercase_2", schema=expected_schema)
    t = con.register(db_t, uppercase_table_name)

    assert uppercase_table_name in con.list_tables()
    assert t.execute() is not None
    assert t.schema() == expected_schema
