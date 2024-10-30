from operator import methodcaller

import ibis
import pandas as pd
import pytest
import pyarrow as pa
from ibis import _

import letsql as ls
from letsql.common.caching import SourceStorage, ParquetCacheStorage
from letsql.expr.relations import into_backend, RemoteTableReplacer

expected_tables = (
    "array_types",
    "astronauts",
    "awards_players",
    "awards_players_special_types",
    "batting",
    "diamonds",
    "functional_alltypes",
    "geo",
    "geography_columns",
    "geometry_columns",
    "json_t",
    "map",
    "spatial_ref_sys",
    "topk",
    "tzone",
)


@pytest.fixture(scope="session")
def pg():
    conn = ls.postgres.connect_env()
    yield conn
    remove_unexpected_tables(conn)


@pytest.fixture(scope="session")
def trino_table():
    return (
        ibis.trino.connect(database="tpch", schema="sf1")
        .table("orders")
        .cast({"orderdate": "date"})
    )


def make_merged(expr):
    agg = expr.group_by(["custkey", "orderdate"]).agg(
        _.totalprice.sum().name("totalprice")
    )
    w = ibis.window(group_by="custkey", order_by="orderdate")
    windowed = (
        agg.mutate(_.totalprice.cumsum().over(w).name("curev"))
        .mutate(_.curev.lag(1).over(w).name("curev@t-1"))
        .select(["custkey", "orderdate", "curev", "curev@t-1"])
    )
    merged = expr.asof_join(
        windowed,
        on="orderdate",
        predicates="custkey",
    ).select(
        [expr[c] for c in expr.columns]
        + [windowed[c] for c in windowed.columns if c not in expr.columns]
    )
    return merged


def remove_unexpected_tables(dirty):
    # drop tables
    for table in dirty.list_tables():
        if table not in expected_tables:
            dirty.drop_table(table, force=True)

    # drop view
    for table in dirty.list_tables():
        if table not in expected_tables:
            dirty.drop_view(table, force=True)

    if sorted(dirty.list_tables()) != sorted(expected_tables):
        raise ValueError


def test_multiple_record_batches(pg):
    con = ls.connect()

    table = pg.table("batting")
    left = con.register(table.to_pyarrow_batches(), "batting_0")
    right = con.register(table.to_pyarrow_batches(), "batting_1")

    expr = (
        left.join(right, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
        .cache(SourceStorage(source=con))
    )

    res = expr.execute()
    assert isinstance(res, pd.DataFrame)
    assert 0 < len(res) <= 15


@pytest.mark.parametrize("method", ["to_pyarrow", "to_pyarrow_batches", "execute"])
def test_into_backend_simple(pg, method):
    con = ls.connect()
    expr = into_backend(pg.table("batting"), con, "ls_batting")
    res = methodcaller(method)(expr)

    if isinstance(res, pa.RecordBatchReader):
        res = next(res)

    assert len(res) > 0


@pytest.mark.parametrize("method", ["to_pyarrow", "to_pyarrow_batches", "execute"])
def test_into_backend_complex(pg, method):
    con = ls.connect()

    t = into_backend(pg.table("batting"), con, "ls_batting")

    expr = (
        t.join(t, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
        .cache(SourceStorage(source=con))
    )

    assert ls.to_sql(expr).count("ls_batting") == 2
    res = methodcaller(method)(expr)

    if isinstance(res, pa.RecordBatchReader):
        res = next(res)

    assert 0 < len(res) <= 15


def test_into_backend_cache(pg, tmp_path):
    con = ls.connect()
    ddb_con = ls.duckdb.connect()

    t = into_backend(pg.table("batting"), con, "ls_batting")

    expr = (
        t.join(t, "playerID")
        .limit(15)
        .cache(SourceStorage(source=con))
        .pipe(into_backend, ddb_con)
        .select(player_id="playerID", year_id="yearID_right")
        .cache(ParquetCacheStorage(source=ddb_con, path=tmp_path))
    )

    res = ls.execute(expr)
    assert 0 < len(res) <= 15


def test_into_backend_duckdb(pg):
    ddb = ibis.duckdb.connect()
    t = into_backend(pg.table("batting"), ddb, "ls_batting")
    expr = (
        t.join(t, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
    )

    replacer = RemoteTableReplacer()
    expr = expr.op().replace(replacer).to_expr()
    query = ibis.to_sql(expr, dialect="duckdb")

    res = ddb.con.sql(query).df()

    assert query.count("ls_batting") == 2
    assert 0 < len(res) <= 15
    assert len(replacer.created) == 3


def test_into_backend_duckdb_expr(pg):
    ddb = ibis.duckdb.connect()
    t = into_backend(pg.table("batting"), ddb, "ls_batting")
    expr = t.join(t, "playerID").limit(15).select(_.playerID * 2)

    replacer = RemoteTableReplacer()
    expr = expr.op().replace(replacer).to_expr()
    query = ibis.to_sql(expr, dialect="duckdb")

    res = ddb.con.sql(query).df()

    assert query.count("ls_batting") == 2
    assert 0 < len(res) <= 15
    assert len(replacer.created) == 3


def test_into_backend_duckdb_trino(trino_table):
    db_con = ibis.duckdb.connect()
    expr = trino_table.head(10_000).pipe(into_backend, db_con).pipe(make_merged)

    replacer = RemoteTableReplacer()
    expr = expr.op().replace(replacer).to_expr()
    query = ibis.to_sql(expr, dialect="duckdb")

    df = db_con.con.sql(query).df()  # to bypass execute hotfix

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert len(replacer.created) == 3


def test_multiple_into_backend_duckdb_letsql(trino_table):
    db_con = ls.duckdb.connect()
    ls_con = ls.connect()

    expr = (
        trino_table.head(10_000)
        .pipe(into_backend, db_con)
        .pipe(make_merged)
        .pipe(into_backend, ls_con)[lambda t: t.orderstatus == "F"]
    )

    replacer = RemoteTableReplacer()
    expr = expr.op().replace(replacer).to_expr()
    df = ls.execute(expr)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert len(replacer.created) == 5


def test_into_backend_duckdb_trino_cached(trino_table, tmp_path):
    db_con = ls.duckdb.connect()
    expr = (
        trino_table.head(10_000)
        .pipe(into_backend, db_con)
        .pipe(make_merged)
        .cache(ParquetCacheStorage(path=tmp_path))
    )
    df = ls.execute(expr)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
