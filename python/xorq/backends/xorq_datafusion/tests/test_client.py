import pandas as pd
import pyarrow as pa
import pytest
from pytest import param

import xorq.api as xo
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
    con.create_table("UPPERCASE", schema=ibis.schema({"A": "int"}))
    con.create_table("name", pd.DataFrame({"a": [1]}))


@pytest.mark.parametrize(
    "kwargs",
    [
        param({"schema": ibis.schema({"a": "!int64", "b": "string"})}, id="schema"),
        param(
            {"obj": pa.table({"a": pa.array([], pa.int64())})},
            id="empty-pyarrow",
        ),
    ],
)
def test_create_empty_table(kwargs):
    con = xo.connect()
    expected = kwargs.get("schema") or ibis.schema(kwargs["obj"].schema)
    t = con.create_table("t", **kwargs)

    assert t.schema() == expected
    assert t.count().execute() == 0


@pytest.mark.parametrize(
    "dtype",
    ["timestamp('UTC', 6)", "int8", "struct<a: int64>", "!int64", "largestring"],
)
def test_create_table_from_schema_keeps_types(dtype):
    con = xo.connect()
    schema = ibis.schema({"c": dtype})

    assert con.create_table("t", schema=schema).schema() == schema


def test_create_table_from_schema_adds_only_that_table():
    con = xo.connect()
    before = set(con.list_tables())
    con.create_table("t", schema=ibis.schema({"a": "int64"}))

    assert set(con.list_tables()) - before == {"t"}


def test_create_table_from_schema_overwrite():
    con = xo.connect()
    con.create_table("t", schema=ibis.schema({"a": "int64"}))
    t = con.create_table("t", schema=ibis.schema({"b": "string"}), overwrite=True)

    assert t.schema() == ibis.schema({"b": "string"})
    with pytest.raises(ValueError, match="already exists"):
        con.create_table("t", schema=ibis.schema({"b": "string"}))


def test_create_table_from_schema_failed_overwrite_keeps_table():
    con = xo.connect()
    con.create_table("t", pa.table({"a": [1]}))

    with pytest.raises(ValueError, match="precision"):
        con.create_table(
            "t", schema=ibis.schema({"c": "decimal(77, 2)"}), overwrite=True
        )
    assert con.table("t").to_pyarrow() == pa.table({"a": [1]})


@pytest.mark.parametrize(
    "keys",
    [
        param((), id="empty"),
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


def test_insert_into_view_names_the_view():
    con = xo.connect()
    data = pa.table({"a": [1]})
    con.create_view("v", con.create_table("t", data))

    con.insert("t", data)
    with pytest.raises(ValueError, match="cannot insert into 'v': it is a view"):
        con.insert("v", data)
    with pytest.raises(ValueError, match="Table not found: missing"):
        con.insert("missing", data)
    # a namespaced target is not the default-schema view: keep its own error
    with pytest.raises(Exception) as excinfo:
        con.insert("v", data, schema="other")
    assert "is a view" not in str(excinfo.value)

    assert con.table("t").count().execute() == 2


def test_drop_table_with_qualified_database():
    con = xo.connect()
    con.create_table("tp", pa.table({"a": [1]}))

    con.drop_table("tp", database=("datafusion", "public"))

    assert "tp" not in con.list_tables()


@pytest.mark.parametrize(
    "kwargs",
    [
        param({"obj": xo.memtable({"a": [1]})}, id="obj"),
        param({"schema": ibis.schema({"a": "int64"})}, id="schema"),
    ],
)
def test_create_table_temp(kwargs):
    con = xo.connect()
    t = con.create_table("t", temp=True, **kwargs)

    assert t.schema() == ibis.schema({"a": "int64"})
    assert "t" in con.list_tables()


@pytest.mark.parametrize(
    "kwargs",
    [
        param({"obj": pa.table({"a": [1]})}, id="obj"),
        param({"schema": ibis.schema({"a": "int64"})}, id="schema"),
    ],
)
@pytest.mark.parametrize(
    "database",
    [
        param("foo", id="name"),
        param("datafusion.foo", id="dotted"),
        param(("datafusion", "foo"), id="tuple"),
    ],
)
def test_create_table_in_database(kwargs, database):
    con = xo.connect()
    con.create_database("foo")

    t = con.create_table("tt", database=database, **kwargs)

    assert t.schema() == ibis.schema({"a": "int64"})
    assert con.get_schema("tt", database="foo") == t.schema()
    assert "tt" not in con.list_tables()


def test_mixed_case_database_round_trips():
    con = xo.connect()
    con.create_database("Foo")
    assert "Foo" in con.list_databases()
    con.create_table("tt", obj=pa.table({"a": [1]}), database="Foo")

    assert xo.execute(con.table("tt", database="Foo")).to_dict("list") == {"a": [1]}

    con.drop_table("tt", database="Foo")
    con.drop_database("Foo")
    assert "Foo" not in con.list_databases()
