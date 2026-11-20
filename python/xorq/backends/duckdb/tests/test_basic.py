import random

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.caching import (
    ParquetCache,
    ParquetSnapshotCache,
    SourceCache,
    SourceSnapshotCache,
)
from xorq.common.exceptions import AmbiguousUDFError
from xorq.common.utils.dasher import tokenize
from xorq.conftest import array_types_df
from xorq.ibis_yaml.compiler import build_expr, load_expr
from xorq.tests.util import assert_frame_equal, check_eq
from xorq.vendor import ibis


KEY_PREFIX = xo.config.options.cache.key_prefix


def test_duckdb_cache_parquet(con, parquet_dir, tmp_path):
    parquet_path = parquet_dir / "astronauts.parquet"
    expr = (
        xo.duckdb.connect()
        .read_parquet(parquet_path)[lambda t: t.number > 22]
        .cache(cache=ParquetCache.from_kwargs(source=con, relative_path=tmp_path))
    )
    expr.execute()


def test_duckdb_cache_csv(con, csv_dir, tmp_path):
    csv_path = csv_dir / "astronauts.csv"
    expr = (
        xo.duckdb.connect()
        .read_csv(csv_path)[lambda t: t.number > 22]
        .cache(cache=ParquetCache.from_kwargs(source=con, relative_path=tmp_path))
    )
    expr.execute()


def test_duckdb_cache_arrow(con, tmp_path):
    name = "astronauts"
    expr = (
        xo.duckdb.connect()
        .create_table(name, con.table(name).to_pyarrow())[lambda t: t.number > 22]
        .cache(cache=ParquetCache.from_kwargs(source=con, relative_path=tmp_path))
    )
    expr.execute()


def test_duckdb_snapshot(con_snapshot):
    con_snapshot(xo.duckdb.connect())


def test_cross_source_snapshot(con_cross_source_snapshot):
    con_cross_source_snapshot(xo.duckdb.connect())


@pytest.mark.parametrize(
    "cls",
    [ParquetSnapshotCache, ParquetCache, SourceSnapshotCache, SourceCache],
)
def test_cache_find_backend(cls, con_cache_find_backend):
    con_cache_find_backend(cls, xo.duckdb.connect())


def test_register_table_with_uppercase(con):
    db_con = xo.duckdb.connect()
    db_t = db_con.create_table("lowercase", schema=ibis.schema({"A": "int"}))

    uppercase_table_name = "UPPERCASE"
    t = con.register(db_t, uppercase_table_name)
    assert uppercase_table_name in con.list_tables()
    assert xo.execute(t) is not None


def test_register_table_with_uppercase_multiple_times(con):
    db_con = xo.duckdb.connect()
    db_t = db_con.create_table("lowercase", schema=ibis.schema({"A": "int"}))

    uppercase_table_name = "UPPERCASE"
    con.register(db_t, uppercase_table_name)

    expected_schema = ibis.schema({"B": "int"})
    db_t = db_con.create_table("lowercase_2", schema=expected_schema)
    t = con.register(db_t, uppercase_table_name)

    assert uppercase_table_name in con.list_tables()
    assert xo.execute(t) is not None
    assert t.schema() == expected_schema


def test_sql_execution(con, duckdb_con, awards_players, batting):
    def make_right(t):
        return t[t.lgID == "NL"].drop("yearID", "lgID")

    ddb_players = con.register(
        duckdb_con.table("ddb_players"), table_name="ddb_players"
    )

    left = batting[batting.yearID == 2015]
    right_df = make_right(awards_players).execute()
    left_df = xo.execute(left)
    predicate = ["playerID"]
    result_order = ["playerID", "yearID", "lgID", "stint"]

    expr = left.join(
        make_right(ddb_players),
        predicate,
        how="inner",
    )

    result = (
        expr.execute()
        .fillna(np.nan)[left.columns]
        .sort_values(result_order)
        .reset_index(drop=True)
    )

    expected = (
        check_eq(
            left_df,
            right_df,
            how="inner",
            on=predicate,
            suffixes=("_x", "_y"),
        )[left.columns]
        .sort_values(result_order)
        .reset_index(drop=True)
    )

    assert_frame_equal(result, expected, check_like=True)

    query = xo.to_sql(expr)
    sql_result = (
        con.sql(query)
        .execute()
        .fillna(np.nan)[left.columns]
        .sort_values(result_order)
        .reset_index(drop=True)
    )
    assert_frame_equal(sql_result, expected, check_like=True)


def test_register_arbitrary_expression(con, duckdb_con):
    batting_table_name = "batting"
    t = duckdb_con.table(batting_table_name)

    expr = t.filter(t.playerID == "allisar01")[
        ["playerID", "yearID", "stint", "teamID", "lgID"]
    ]
    expected = expr.execute()

    ddb_batting_table_name = f"{duckdb_con.name}_{batting_table_name}"
    table = con.register(expr, table_name=ddb_batting_table_name)
    result = table.execute()

    assert result is not None
    assert_frame_equal(result, expected, check_like=True)


def test_array_filter_cached(con, duckdb_con):
    t = duckdb_con.create_table("array_types", array_types_df)
    uncached = t.mutate(filtered=t.x.filter(xo._ > 1)).into_backend(
        con, name="array_types"
    )
    expr = uncached.cache(SourceCache.from_kwargs(source=con))

    assert tokenize(uncached)
    assert not expr.execute().empty


def test_pyarrow_udf_load_executes(tmp_path):
    @xo.udf.scalar.pyarrow
    def plus_one(arr: dt.float64) -> dt.float64:
        return pc.add(arr, 1.0)

    t = xo.duckdb.connect().create_table("t", {"x": [1.5, 2.5]})
    expr = t.mutate(y=plus_one(t.x))
    actual = load_expr(build_expr(expr, builds_dir=tmp_path)).execute()

    assert actual.to_dict("records") == [{"x": 1.5, "y": 2.5}, {"x": 2.5, "y": 3.5}]


# expected == 1 relies on duckdb folding a side-effect-free python UDF over a
# literal into one call, an optimizer behaviour rather than an API guarantee
@pytest.mark.parametrize(
    ("volatility", "expected"), [("volatile", 3), ("immutable", 1), ("stable", 1)]
)
def test_pyarrow_udf_volatility_controls_folding(volatility, expected):
    @xo.udf.scalar.pyarrow(volatility=volatility)
    def noise(arr: dt.float64) -> dt.float64:
        return pa.array([random.random() for _ in range(len(arr))])

    t = xo.duckdb.connect().create_table("t", {"x": [1.0, 2.0, 3.0]})
    actual = t.mutate(r=noise(xo.literal(1.0, "float64"))).execute()

    assert actual.r.nunique() == expected


def test_pyarrow_udf_load_keeps_volatility(tmp_path):
    @xo.udf.scalar.pyarrow(volatility="volatile")
    def noise(arr: dt.float64) -> dt.float64:
        return pa.array([random.random() for _ in range(len(arr))])

    t = xo.duckdb.connect().create_table("t", {"x": [1.0, 2.0, 3.0]})
    expr = t.mutate(r=noise(xo.literal(1.0, "float64")))
    loaded = load_expr(build_expr(expr, builds_dir=tmp_path))

    assert loaded.execute().r.nunique() == expr.execute().r.nunique() == 3


@pytest.mark.parametrize("first", ["live", "loaded"])
def test_pyarrow_udf_executes_twice_on_one_connection(tmp_path, first):
    @xo.udf.scalar.pyarrow
    def plus_one(arr: dt.float64) -> dt.float64:
        return pc.add(arr, 1.0)

    t = xo.duckdb.connect().create_table("t", {"x": [1.5, 2.5]})
    expr = t.mutate(y=plus_one(t.x))
    if first == "loaded":
        expr = load_expr(build_expr(expr, builds_dir=tmp_path))
    first_rows = expr.execute()
    u = expr._find_backend().create_table("u", {"x": [1.5, 2.5]})
    live_rows = u.mutate(y=plus_one(u.x)).execute()

    assert first_rows.y.tolist() == live_rows.y.tolist() == [2.5, 3.5]


def test_builtin_udf_keeps_connection_function():
    con = xo.duckdb.connect()
    con.con.create_function("twice", lambda x: x * 2, ["DOUBLE"], "DOUBLE")

    @xo.udf.scalar.builtin
    def twice(x: dt.float64) -> dt.float64: ...

    t = con.create_table("t", {"x": [1.0, 2.0]})

    assert t.mutate(y=twice(t.x)).execute().y.tolist() == [2.0, 4.0]


def test_distinct_udfs_sharing_a_name_raise():
    def make(offset):
        @xo.udf.scalar.pyarrow
        def add_offset(arr: dt.float64) -> dt.float64:
            return pc.add(arr, offset)

        return add_offset

    one, hundred = make(1.0), make(100.0)
    t = xo.duckdb.connect().create_table("t", {"x": [1.0]})

    with pytest.raises(AmbiguousUDFError, match="add_offset"):
        t.mutate(p=one(t.x), q=hundred(t.x)).execute()


@pytest.mark.parametrize(
    "column",
    [
        pa.array(["a", None], pa.large_string()),
        pa.array([b"a", None], pa.large_binary()),
        pa.array([["a"], None], pa.large_list(pa.large_string())),
        pa.array([b"a", None], pa.binary_view()),
    ],
    ids=["large_string", "large_binary", "large_list", "binary_view"],
)
def test_create_table_from_arrow_large_types(duckdb_con, column):
    t = duckdb_con.create_table("large_types", pa.table({"c": column}), overwrite=True)

    assert t.to_pyarrow().column("c").to_pylist() == column.to_pylist()
