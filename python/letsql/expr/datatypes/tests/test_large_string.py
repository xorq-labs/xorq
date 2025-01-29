import ibis
import pandas as pd
from ibis.expr.datatypes import Int64, String

import letsql as ls
from letsql.expr.datatypes import LargeString
from letsql.tests.util import assert_frame_equal, assert_series_equal


def test_can_create_table(utf8_data):
    con = ls.connect()
    schema = ibis.schema([("name", LargeString), ("age", Int64)])
    t = con.create_table("names", utf8_data, schema=schema)
    assert t.schema() == schema
    assert "names" in con.list_tables()

    actual = ls.execute(t)  # can be executed
    assert_frame_equal(utf8_data, actual)


def test_can_read_table(utf8_data):
    con = ls.connect()

    data = """
    create table utf8_data(str string, val bigint) as values
      ('A', 1),
      ('B', 2),
      ('A', 2),
      ('A', 4),
      ('C', 1),
      ('A', 1);
    """

    con.con.sql(data)

    large_data = """
    create table largeutf8_data as
    select arrow_cast(str, 'LargeUtf8') as str, val
    from utf8_data;
    """

    con.con.sql(large_data)

    t = con.table("largeutf8_data")
    schema = ibis.schema([("str", LargeString), ("val", Int64)])
    assert t.schema() == schema

    expr = t.select(t.str.name("name"), t.val.name("age"))
    actual = ls.execute(expr)

    assert_frame_equal(actual, utf8_data)


def test_can_read_write_parquet(utf8_data, tmp_path):
    con = ls.connect()
    schema = ibis.schema([("name", LargeString), ("age", Int64)])
    t = con.create_table("names", utf8_data, schema=schema)

    names_parquet_path = tmp_path / "names.parquet"
    ls.to_parquet(t, names_parquet_path)

    # need to specify the schema when reading parquet because DataFusion transforms into StringView
    t = ls.read_parquet(names_parquet_path, "names_parquet", schema=schema.to_pyarrow())
    assert t.schema() == schema

    actual = ls.execute(t)

    assert_frame_equal(utf8_data, actual)


def test_can_execute_test_ops(utf8_data):
    con = ls.connect()
    schema = ibis.schema([("name", LargeString), ("age", Int64)])
    t = con.create_table("names", utf8_data, schema=schema)
    assert t.schema() == schema

    expr = t.name.startswith("A").name("prefix")
    actual = con.execute(expr)
    expected = utf8_data["name"].str.startswith("A")

    assert_series_equal(actual, expected, check_names=False)


def test_cast(utf8_data):
    con = ls.connect()
    t = con.register(utf8_data, "t")
    expr = t.mutate(a=t.name.cast(LargeString))
    actual = ls.execute(expr)

    assert expr.schema() == ibis.schema(
        [("name", String), ("age", Int64), ("a", LargeString)]
    )
    assert isinstance(actual, pd.DataFrame)
