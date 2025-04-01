import pyarrow as pa
import pytest
from pytest import param

import xorq as xo
from xorq.tests.util import (
    assert_frame_equal,
)
from xorq.vendor import ibis


def test_register_record_batch_reader(alltypes_df):
    con = xo.connect()
    t = con.register(alltypes_df, "alltypes")
    record_batch_reader = t.to_pyarrow_batches()
    t2 = con.register(record_batch_reader, "t2")
    actual = xo.execute(t2)

    assert isinstance(record_batch_reader, pa.RecordBatchReader)
    assert_frame_equal(actual, alltypes_df)


def test_register_expr(alltypes_df):
    con = xo.connect()
    t = con.register(alltypes_df, "alltypes")
    t2 = con.register(t, "alltypes2")
    actual = xo.execute(t2)
    assert_frame_equal(actual, alltypes_df)


def test_execute_nonnull():
    schema = pa.schema((pa.field("x", pa.int64(), nullable=False),))
    tab = pa.table({"x": [1, 2, 3]}, schema=schema)
    con = xo.connect()
    t = con.register(tab, "my_table")
    xo.execute(t)


def test_register_record_batch_reader_with_filter(alltypes, alltypes_df):
    con = xo.connect()
    t = con.register(alltypes_df, "alltypes")
    record_batch_reader = t.to_pyarrow_batches()
    t2 = con.register(record_batch_reader, "t2")
    cols_a = [ca for ca in alltypes.columns.copy() if ca != "timestamp_col"]
    expr = t2.filter((t2.id >= 5200) & (t2.id <= 5210))[cols_a]
    xo.execute(expr)


def test_create_table(con):
    import pandas as pd

    con.create_table("UPPERCASE", schema=ibis.schema({"A": "int"}))
    con.create_table("name", pd.DataFrame({"a": [1]}))


def test_register_table_with_uppercase(ls_con):
    db_con = xo.duckdb.connect()
    db_t = db_con.create_table("lowercase", schema=ibis.schema({"A": "int"}))

    uppercase_table_name = "UPPERCASE"
    t = ls_con.register(db_t, uppercase_table_name)
    assert uppercase_table_name in ls_con.list_tables()
    assert xo.execute(t) is not None


def test_register_table_with_uppercase_multiple_times(ls_con):
    db_con = xo.duckdb.connect()
    db_t = db_con.create_table("lowercase", schema=ibis.schema({"A": "int"}))

    uppercase_table_name = "UPPERCASE"
    ls_con.register(db_t, uppercase_table_name)

    expected_schema = ibis.schema({"B": "int"})
    db_t = db_con.create_table("lowercase_2", schema=expected_schema)
    t = ls_con.register(db_t, uppercase_table_name)

    assert uppercase_table_name in ls_con.list_tables()
    assert xo.execute(t) is not None
    assert t.schema() == expected_schema


@pytest.mark.parametrize(
    "keys",
    [
        param(tuple(), id="empty"),
        param((xo.asc("yearID"),), id="one-column"),
        param((xo.asc("yearID"), xo.desc("stint")), id="two-columns"),
    ],
)
def test_register_record_batch_reader_sorted(batting_df, keys):
    con = xo.connect()
    t = con.register(batting_df, "batting")

    if keys:
        t = t.order_by(*keys)

    record_batch_reader = t.select("yearID", "stint").to_pyarrow_batches()
    t = con.register(
        record_batch_reader,
        "t2",
        ordering=[key.resolve(t) for key in keys],
    )
    expr = t.group_by("yearID").agg(max_tiny=t["stint"].max())

    physical_plan = xo.get_plans(expr)["physical_plan"]
    ordering_string = "ordering_mode=Sorted"
    if keys:
        assert ordering_string in physical_plan
    else:
        assert ordering_string not in physical_plan
