import dask
import numpy as np
import pytest

import xorq.api as xo
from xorq.caching import (
    ParquetCache,
    ParquetSnapshotCache,
    SourceCache,
    SourceSnapshotCache,
)
from xorq.conftest import array_types_df
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

    expr = con.register(left, "batting").join(
        make_right(con.register(ddb_players, "players")),
        predicate,
        how="inner",
    )
    query = xo.to_sql(expr)

    result = (
        con.sql(query)
        .execute()
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

    assert dask.base.tokenize(uncached)
    assert not expr.execute().empty
