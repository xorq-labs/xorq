import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.caching import SourceCache
from xorq.expr.datatypes import Binary, Int64, LargeString, String
from xorq.tests.util import assert_frame_equal, assert_series_equal
from xorq.vendor import ibis


def test_can_create_table(utf8_data):
    con = xo.connect()
    schema = ibis.schema([("name", LargeString), ("age", Int64)])
    t = con.create_table("names", utf8_data, schema=schema)
    assert t.schema() == schema
    assert "names" in con.list_tables()

    actual = xo.execute(t)  # can be executed
    assert_frame_equal(utf8_data, actual)


def test_can_read_table(utf8_data):
    con = xo.connect()

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
    actual = xo.execute(expr)

    assert_frame_equal(actual, utf8_data)


def test_can_read_write_parquet(utf8_data, tmp_path):
    con = xo.connect()
    schema = ibis.schema([("name", LargeString), ("age", Int64)])
    t = con.create_table("names", utf8_data, schema=schema)

    names_parquet_path = tmp_path / "names.parquet"
    xo.to_parquet(t, names_parquet_path)

    # need to specify the schema when reading parquet because DataFusion transforms into StringView
    t = con.read_parquet(
        names_parquet_path, "names_parquet", schema=schema.to_pyarrow()
    )
    assert t.schema() == schema

    actual = xo.execute(t)

    assert_frame_equal(utf8_data, actual)


def test_can_execute_test_ops(utf8_data):
    con = xo.connect()
    schema = ibis.schema([("name", LargeString), ("age", Int64)])
    t = con.create_table("names", utf8_data, schema=schema)
    assert t.schema() == schema

    expr = t.name.startswith("A").name("prefix")
    actual = con.execute(expr)
    expected = utf8_data["name"].str.startswith("A")

    assert_series_equal(actual, expected, check_names=False)


def test_cast(utf8_data):
    con = xo.connect()
    t = con.register(utf8_data, "t")
    expr = t.mutate(a=t.name.cast(LargeString))
    actual = xo.execute(expr)

    assert expr.schema() == ibis.schema(
        [("name", String), ("age", Int64), ("a", LargeString)]
    )
    assert isinstance(actual, pd.DataFrame)


def test_deferred_read_parquet_keeps_large_string(tmp_path):
    path = tmp_path / "test_largestring.parquet"
    expr = xo.memtable({"id": [1, 2, 3], "text_col": ["A", "B", "C"]}).mutate(
        text_col=xo._.text_col.cast(LargeString)
    )
    expr.to_parquet(path)

    read_back = xo.deferred_read_parquet(path=path, con=xo.connect())

    assert read_back.schema() == expr.schema()
    assert read_back.to_pyarrow().schema == expr.to_pyarrow().schema


@pytest.mark.parametrize(
    ("arrow_type", "value", "expected"),
    (
        (pa.string(), "a", String),
        (pa.large_string(), "a", LargeString),
        (pa.binary(), b"a", Binary),
        (pa.large_binary(), b"a", Binary),
    ),
)
def test_deferred_read_parquet_infers_stored_type(
    tmp_path, arrow_type, value, expected
):
    path = tmp_path / "typed.parquet"
    pq.write_table(pa.table({"col": pa.array([value], arrow_type)}), path)

    read_back = xo.deferred_read_parquet(path=path, con=xo.connect())

    assert read_back.schema()["col"] == expected()


@pytest.fixture
def large_string_parquet(tmp_path):
    path = tmp_path / "large_string.parquet"
    table = pa.table({"id": [1, 2], "s": pa.array(["a", "b"], pa.large_string())})
    pq.write_table(table, path)
    return path


READ_CONNECTS = pytest.mark.parametrize(
    "connect",
    (lambda: xo.connect(), lambda: xo.duckdb.connect(), lambda: xo.sqlite.connect()),
    ids=("xorq", "duckdb", "sqlite"),
)


@READ_CONNECTS
def test_deferred_read_parquet_unions_with_large_string_table(
    large_string_parquet, connect
):
    read = xo.deferred_read_parquet(path=large_string_parquet, con=connect())
    other = xo.memtable({"id": [3], "s": ["c"]}).mutate(s=xo._.s.cast(LargeString))

    assert sorted(read.union(other).execute().s) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "connect",
    (lambda: xo.connect(), lambda: xo.sqlite.connect()),
    ids=("xorq", "sqlite"),
)
def test_deferred_read_parquet_streams_recorded_schema(large_string_parquet, connect):
    read = xo.deferred_read_parquet(path=large_string_parquet, con=connect())

    assert read.to_pyarrow_batches().schema == read.schema().to_pyarrow()


@READ_CONNECTS
def test_deferred_read_parquet_large_string_caches_into_duckdb(
    large_string_parquet, connect
):
    read = xo.deferred_read_parquet(path=large_string_parquet, con=connect())
    cached = read.cache(SourceCache.from_kwargs(source=xo.duckdb.connect()))

    assert sorted(cached.execute().s) == ["a", "b"]


def test_create_table_from_schema_only():
    con = xo.connect()
    schema = ibis.schema({"c": LargeString()})
    t = con.create_table("e", schema=schema)
    con.insert("e", pa.table({"c": pa.array(["x"], pa.large_string())}))

    assert t.schema() == schema
    assert con.table("e").to_pyarrow() == pa.table(
        {"c": pa.array(["x"], pa.large_string())}
    )
