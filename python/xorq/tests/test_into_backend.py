from __future__ import annotations

import itertools
import operator
import random
import warnings
from operator import methodcaller
from pathlib import PosixPath

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pytest import param

import xorq.api as xo
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.types as ir
from xorq.caching import ParquetCache, SourceCache
from xorq.common.exceptions import XorqError
from xorq.expr.remote_table_exec import REMOTE_PASS, RemoteTableScope
from xorq.expr.transform import TransformCtx, apply_pass
from xorq.loader import load_backend
from xorq.tests.util import assert_frame_equal, check_eq
from xorq.vendor import ibis
from xorq.vendor.ibis import _
from xorq.vendor.ibis.backends import BaseBackend
from xorq.vendor.ibis.expr.types.relations import CACHED_NODE_NAME_PLACEHOLDER


KEY_PREFIX = xo.config.options.cache.key_prefix


@pytest.fixture(scope="function")
def ls_con():
    return xo.connect()


@pytest.fixture
def pg_alltypes(pg):
    return pg.table("functional_alltypes")


@pytest.fixture
def pg_alltypes_df(pg_alltypes):
    return pg_alltypes.execute()


@pytest.fixture(scope="session")
def trino_table():
    return (
        xo.trino.connect(database="tpch", schema="sf1")
        .table("orders")
        .cast({"orderdate": "date"})
    )


@pytest.fixture(scope="function")
def duckdb_con(csv_dir):
    con = xo.duckdb.connect()
    con.read_csv(csv_dir / "awards_players.csv", table_name="ddb_players")
    con.read_csv(csv_dir / "batting.csv", table_name="batting")
    return con


@pytest.fixture(scope="function")
def pg_batting(pg):
    return pg.table("batting")


@pytest.fixture(scope="session")
def parquet_batting(parquet_dir):
    return parquet_dir / "batting.parquet"


@pytest.fixture(scope="session")
def ls_batting(parquet_batting):
    return xo.connect().read_parquet(parquet_batting)


@pytest.fixture(scope="function")
def ddb_batting(duckdb_con):
    return duckdb_con.create_table(
        "db-batting",
        duckdb_con.table("batting").to_pyarrow(),
    )


@pytest.fixture(scope="function")
def pd_batting(parquet_batting):
    return xo.deferred_read_parquet(
        parquet_batting,
        xo.pandas.connect(),
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


@pytest.mark.postgres
def test_multiple_record_batches(pg):
    con = xo.connect()

    table = pg.table("batting")
    left = con.register(table.to_pyarrow_batches(), "batting_0")
    right = con.register(table.to_pyarrow_batches(), "batting_1")

    expr = (
        left.join(right, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
        .cache(SourceCache.from_kwargs(source=con))
    )

    res = expr.execute()
    assert isinstance(res, pd.DataFrame)
    assert 0 < len(res) <= 15


@pytest.mark.postgres
@pytest.mark.parametrize("method", [xo.to_pyarrow, xo.to_pyarrow_batches, xo.execute])
def test_into_backend_simple(pg, method):
    con = xo.connect()
    expr = pg.table("batting").into_backend(con, "ls_batting")
    res = method(expr)

    if isinstance(res, pa.RecordBatchReader):
        res = next(res)

    assert len(res) > 0


@pytest.mark.postgres
@pytest.mark.parametrize("method", ["to_pyarrow", "to_pyarrow_batches", "execute"])
def test_into_backend_complex(pg, method):
    con = xo.connect()

    t = pg.table("batting").into_backend(con, "ls_batting")

    expr = (
        t.join(t, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
        .cache(SourceCache.from_kwargs(source=con))
    )

    assert xo.to_sql(expr).count(CACHED_NODE_NAME_PLACEHOLDER) > 0
    res = methodcaller(method)(expr)

    if isinstance(res, pa.RecordBatchReader):
        res = next(res)

    assert 0 < len(res) <= 15


@pytest.mark.postgres
def test_double_into_backend_batches(pg):
    con = xo.connect()
    ddb_con = xo.duckdb.connect()

    t = pg.table("batting").into_backend(con, "ls_batting")

    expr = (
        t.join(t, "playerID")
        .limit(15)
        .into_backend(ddb_con)
        .select(player_id="playerID", year_id="yearID_right")
        .cache(SourceCache.from_kwargs(source=con))
    )

    res = expr.to_pyarrow_batches()
    res = next(res)

    assert len(res) == 15


@pytest.mark.benchmark
def test_into_backend_cache(pg, tmp_path):
    con = xo.connect()
    ddb_con = xo.duckdb.connect()

    t = pg.table("batting").into_backend(con, "ls_batting")

    expr = (
        t.join(t, "playerID")
        .limit(15)
        .cache(SourceCache.from_kwargs(source=con))
        .into_backend(ddb_con)
        .select(player_id="playerID", year_id="yearID_right")
        .cache(ParquetCache.from_kwargs(source=ddb_con, relative_path=tmp_path))
    )

    res = expr.execute()
    assert 0 < len(res) <= 15


@pytest.mark.postgres
def test_into_backend_duckdb(pg):
    ddb = xo.duckdb.connect()
    t = pg.table("batting").into_backend(ddb, "ls_batting")
    expr = (
        t.join(t, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
    )

    scope = RemoteTableScope()
    expr = apply_pass(REMOTE_PASS, expr, TransformCtx(scope=scope))
    with scope:
        query = ibis.to_sql(expr, dialect="duckdb")
        res = ddb.con.sql(query).df()

    assert scope.table_count == 1
    registered_name = scope.table_names[0]
    assert query.count(registered_name) == 2
    assert 0 < len(res) <= 15


@pytest.mark.postgres
def test_into_backend_duckdb_expr(pg: BaseBackend) -> None:
    ddb = xo.duckdb.connect()
    t = pg.table("batting").into_backend(ddb, "ls_batting")
    expr = t.join(t, "playerID").limit(15).select(_.playerID * 2)

    scope = RemoteTableScope()
    expr = apply_pass(REMOTE_PASS, expr, TransformCtx(scope=scope))
    with scope:
        query = ibis.to_sql(expr, dialect="duckdb")
        res = ddb.con.sql(query).df()

    assert scope.table_count == 1
    registered_name = scope.table_names[0]
    assert query.count(registered_name) == 2
    assert 0 < len(res) <= 15


@pytest.mark.trino
def test_into_backend_duckdb_trino(trino_table):
    db_con = xo.duckdb.connect()
    expr = trino_table.head(10_000).into_backend(db_con).pipe(make_merged)

    scope = RemoteTableScope()
    expr = apply_pass(REMOTE_PASS, expr, TransformCtx(scope=scope))
    with scope:
        query = ibis.to_sql(expr, dialect="duckdb")
        df = db_con.con.sql(query).df()  # to bypass execute hotfix

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert scope.table_count == 1


@pytest.mark.trino
def test_multiple_into_backend_duckdb_xorq(trino_table: ir.Table) -> None:
    db_con = xo.duckdb.connect()
    ls_con = xo.connect()

    expr = (
        trino_table.head(10_000)
        .into_backend(db_con)
        .pipe(make_merged)
        .into_backend(ls_con)[lambda t: t.orderstatus == "F"]
    )

    scope = RemoteTableScope()
    expr = apply_pass(REMOTE_PASS, expr, TransformCtx(scope=scope))
    with scope:
        df = expr.execute()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert scope.table_count == 1


@pytest.mark.trino
@pytest.mark.benchmark
def test_into_backend_duckdb_trino_cached(trino_table, tmp_path):
    db_con = xo.duckdb.connect()
    expr = (
        trino_table.head(10_000)
        .into_backend(db_con)
        .pipe(make_merged)
        .cache(ParquetCache.from_kwargs(relative_path=tmp_path))
    )
    df = expr.execute()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.trino
def test_into_backend_to_pyarrow_batches(trino_table):
    db_con = xo.duckdb.connect()
    df = (
        trino_table.head(10_000)
        .into_backend(db_con)
        .pipe(make_merged)
        .to_pyarrow_batches()
        .read_pandas()
    )
    assert not df.empty


@pytest.mark.postgres
def test_to_pyarrow_batches_simple(pg):
    con = xo.duckdb.connect()

    t = pg.table("batting").into_backend(con, "ls_batting")

    expr = (
        t.join(t, "playerID")
        .limit(15)
        .select(player_id="playerID", year_id="yearID_right")
    )

    df = expr.to_pyarrow_batches().read_pandas()
    assert not df.empty


@pytest.mark.postgres
def test_join(ls_con, pg_alltypes, pg_alltypes_df):
    first_10 = pg_alltypes_df.head(10)
    in_memory = ls_con.create_table("in_memory", first_10).into_backend(
        pg_alltypes.op().source
    )
    expr = pg_alltypes.join(in_memory, predicates=[pg_alltypes.id == in_memory.id])
    actual = expr.execute().sort_values("id")
    expected = pd.merge(
        pg_alltypes_df, first_10, how="inner", on="id", suffixes=("", "_right")
    ).sort_values("id")

    assert_frame_equal(actual, expected)


@pytest.fixture
def union_subsets(pg_alltypes, pg_alltypes_df):
    cols_a, cols_b, cols_c = (pg_alltypes.columns.copy() for _ in range(3))

    random.seed(89)
    random.shuffle(cols_a)
    random.shuffle(cols_b)
    random.shuffle(cols_c)
    assert cols_a != cols_b != cols_c

    cols_a = [ca for ca in cols_a if ca != "timestamp_col"]
    cols_b = [cb for cb in cols_b if cb != "timestamp_col"]
    cols_c = [cc for cc in cols_c if cc != "timestamp_col"]

    a = pg_alltypes.filter((pg_alltypes.id >= 5200) & (pg_alltypes.id <= 5210))[cols_a]
    b = pg_alltypes.filter((pg_alltypes.id >= 5205) & (pg_alltypes.id <= 5215))[cols_b]
    c = pg_alltypes.filter((pg_alltypes.id >= 5213) & (pg_alltypes.id <= 5220))[cols_c]

    da = pg_alltypes_df[(pg_alltypes_df.id >= 5200) & (pg_alltypes_df.id <= 5210)][
        cols_a
    ]
    db = pg_alltypes_df[(pg_alltypes_df.id >= 5205) & (pg_alltypes_df.id <= 5215)][
        cols_b
    ]
    dc = pg_alltypes_df[(pg_alltypes_df.id >= 5213) & (pg_alltypes_df.id <= 5220)][
        cols_c
    ]

    return (a, b, c), (da, db, dc)


@pytest.mark.postgres
@pytest.mark.parametrize("distinct", [False, True], ids=["all", "distinct"])
def test_union(ls_con, union_subsets, distinct):
    (a, _, _), (da, db, dc) = union_subsets

    b = ls_con.create_table("b", db)
    expr = ibis.union(a.into_backend(ls_con), b, distinct=distinct).order_by("id")
    result = expr.execute()

    expected = pd.concat([da, db], axis=0).sort_values("id").reset_index(drop=True)

    if distinct:
        expected = expected.drop_duplicates("id")

    assert_frame_equal(result, expected)


@pytest.mark.postgres
def test_union_mixed_distinct(ls_con, union_subsets):
    (a, _, _), (da, db, dc) = union_subsets

    a = a.into_backend(ls_con)
    b = ls_con.create_table("b", db)
    c = ls_con.create_table("c", dc)

    expr = a.union(b, distinct=True).union(c, distinct=False).order_by("id")
    result = expr.execute()
    expected = pd.concat(
        [pd.concat([da, db], axis=0).drop_duplicates("id"), dc], axis=0
    ).sort_values("id")

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "left_filter",
    [
        lambda t: t.yearID == 2015,
        lambda t: t.yearID != 2015,
        lambda t: t.yearID >= 2015,
        lambda t: t.yearID >= 2014.5,
        lambda t: t.yearID <= 2015,
        lambda t: t.yearID > 2015,
        lambda t: t.yearID < 2015,
        lambda t: ~(t.yearID < 2015),
        lambda t: t.yearID.notnull(),
        lambda t: t.yearID.isnull(),
        lambda t: t.playerID == "wilsobo02",
        lambda t: t.yearID.isin([2015, 2014]),
        lambda t: t.yearID.notin([2015, 2014]),
        lambda t: t.yearID.between(2013, 2016),
    ],
)
def test_join_non_trivial_filters(ls_batting, duckdb_con, left_filter):
    awards_players = duckdb_con.table("ddb_players")
    batting = ls_batting.into_backend(duckdb_con)

    left = batting[left_filter]
    right = awards_players[awards_players.lgID == "NL"].drop("yearID", "lgID")
    right_df = right.execute()
    left_df = left.execute()
    predicate = "playerID"
    result_order = ["playerID", "yearID", "lgID", "stint"]

    expr = left.join(right, predicate, how="inner")
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


@pytest.mark.parametrize(
    ("predicate", "pandas_value"),
    [
        # True
        param(True, True, id="true"),
        param(ibis.literal(True), True, id="true-literal"),
        param([True], True, id="true-list"),
        param([ibis.literal(True)], True, id="true-literal-list"),
        # only True
        param([True, True], True, id="true-true-list"),
        param(
            [ibis.literal(True), ibis.literal(True)], True, id="true-true-literal-list"
        ),
        param([True, ibis.literal(True)], True, id="true-true-const-expr-list"),
        param([ibis.literal(True), True], True, id="true-true-expr-const-list"),
        # False
        param(False, False, id="false"),
        param(ibis.literal(False), False, id="false-literal"),
        param([False], False, id="false-list"),
        param([ibis.literal(False)], False, id="false-literal-list"),
        # only False
        param([False, False], False, id="false-false-list"),
        param(
            [ibis.literal(False), ibis.literal(False)],
            False,
            id="false-false-literal-list",
        ),
        param([False, ibis.literal(False)], False, id="false-false-const-expr-list"),
        param([ibis.literal(False), False], False, id="false-false-expr-const-list"),
    ],
)
@pytest.mark.parametrize(
    "how",
    [
        "inner",
        "left",
        "right",
        "outer",
    ],
)
def test_join_with_trivial_predicate(
    duckdb_con, awards_players, predicate, how, pandas_value
):
    ddb_players = duckdb_con.table("ddb_players")

    n = 5

    base = awards_players.into_backend(duckdb_con).limit(n)
    ddb_base = ddb_players.limit(n)

    left = base.select(left_key="playerID")
    right = ddb_base.select(right_key="playerID")

    left_df = pd.DataFrame({"key": [True] * n})
    right_df = pd.DataFrame({"key": [pandas_value] * n})

    expected = pd.merge(left_df, right_df, on="key", how=how)

    expr = left.join(right, predicate, how=how)
    result = expr.execute()

    assert len(result) == len(expected)


@pytest.mark.postgres
@pytest.mark.parametrize(
    "backend_name",
    [
        "",
        "duckdb",
    ],
)
def test_multiple_pipes(pg, backend_name):
    """This test address the issue reported on bug #69
    link: https://github.com/letsql/letsql/issues/69

    NOTE
    The previous tests didn't catch it because the con object registered the table batting.
    In this test (and the rest) ls_con is a clean (no tables) letsql connection
    """

    new_con = load_backend(backend_name).connect() if backend_name else xo.connect()
    table_name = "batting"
    pg_t = pg.table(table_name)[lambda t: t.yearID == 2015]
    db_t = new_con.create_table(f"db-{table_name}", pg_t.to_pyarrow())[
        lambda t: t.yearID == 2014
    ]

    expr = pg_t.join(
        db_t.into_backend(pg),
        "playerID",
    )

    assert expr.execute() is not None


@pytest.mark.postgres
@pytest.mark.parametrize(
    "function",
    ["to_pyarrow", "execute", "to_pyarrow_batches"],
)
@pytest.mark.parametrize("remote", [True, False])
def test_duckdb_datafusion_roundtrip(ls_con, pg, duckdb_con, function, remote):
    source = pg if remote else ls_con
    cache = SourceCache.from_kwargs(source=source)

    table_name = "batting"
    pg_t = pg.table(table_name)[lambda t: t.yearID == 2015].cache(cache)

    db_t = duckdb_con.create_table(f"ls-{table_name}", xo.to_pyarrow(pg_t))[
        lambda t: t.yearID == 2014
    ]

    expr = pg_t.join(
        db_t.into_backend(pg if remote else ls_con),
        "playerID",
    )

    assert methodcaller(function)(expr) is not None
    assert any(table_name.startswith(KEY_PREFIX) for table_name in source.list_tables())


@pytest.mark.postgres
@pytest.mark.parametrize(
    "tables",
    [
        param(pair, id="-".join(pair))
        for pair in itertools.combinations_with_replacement(
            ["batting", "pg_batting", "parquet_batting", "ls_batting", "ddb_batting"],
            r=2,
        )
    ],
)
def test_execution_expr_multiple_tables(ls_con, tables, request, mocker):
    left, right = map(request.getfixturevalue, tables)

    left_t = (left if not isinstance(left, PosixPath) else ls_con.read_parquet(left))[
        lambda t: t.yearID == 2015
    ]
    left_backend = left_t._find_backend(use_default=False)

    right_t = (
        right if not isinstance(right, PosixPath) else ls_con.read_parquet(right)
    )[lambda t: t.yearID == 2014]
    right_backend = right_t._find_backend(use_default=False)

    # FIXME there seems to be an issue when doing into_backend from duckdb into duckdb
    expr = left_t.join(
        right_t.into_backend(left_backend)
        if right_backend is not left_backend
        else right_t,
        "playerID",
    )

    native_backend = (
        left_backend is right_backend and left_backend.name != "xorq_datafusion"
    )
    spy = mocker.spy(left_backend, "to_pyarrow_batches") if native_backend else None

    assert expr.execute() is not None
    assert getattr(spy, "call_count", 0) == int(native_backend)


@pytest.mark.postgres
@pytest.mark.parametrize(
    "tables",
    [
        param(
            pair,
            id="-".join(pair),
        )
        for pair in itertools.permutations(
            ["pg_batting", "ls_batting", "pd_batting", "ddb_batting"],
            r=2,
        )
    ],
)
def test_execution_expr_multiple_tables_cached(ls_con, tables, request):
    table_name = "batting"
    left, right = map(request.getfixturevalue, tables)
    (left_source, right_source) = (expr.op().source for expr in (left, right))
    (left_cache, right_cache) = (
        SourceCache.from_kwargs(source=source)
        if source.name != "duckdb"
        else ParquetCache.from_kwargs(source=ls_con)
        for source in (left_source, right_source)
    )
    (left_filter, right_filter) = (
        operator.eq(xo._.yearID, year) for year in (2015, 2014)
    )

    left_t = left.into_backend(ls_con, name=f"left-{table_name}").filter(left_filter)
    right_t = right.into_backend(ls_con, name=f"right-{table_name}").filter(
        right_filter
    )
    joined = (
        left_t.cache(left_cache)
        .join(
            right_t.cache(right_cache).into_backend(left_cache.storage.source),
            "playerID",
        )
        .cache(left_cache)
    )

    actual = joined.execute()
    expected = left_t.join(
        right_t.into_backend(left_t._find_backend()),
        "playerID",
    ).execute()
    assert not actual.empty
    columns = list(actual.columns)
    assert_frame_equal(
        actual.sort_values(columns, ignore_index=True),
        expected.sort_values(columns, ignore_index=True),
    )


@pytest.mark.postgres
def test_no_registration_same_table_name(ls_con, pg_batting):
    ddb_con = xo.duckdb.connect()
    table = pg_batting[["playerID", "yearID"]].to_pyarrow()
    ddb_batting = ddb_con.create_table("batting", table)
    ls_batting = ls_con.register(table, "batting")

    expr = ddb_batting.join(
        ls_batting.into_backend(ddb_con),
        "playerID",
    )

    assert expr.execute() is not None


@pytest.mark.postgres
@pytest.mark.parametrize("backend_name", ["", "duckdb"])
def test_multi_engine_cache(pg, ls_con, ls_batting, tmp_path, backend_name):
    other_con = load_backend(backend_name).connect() if backend_name else xo.connect()
    table_name = "batting"
    pg_t = pg.table(table_name)[lambda t: t.yearID > 2014]
    db_t = other_con.create_table(f"db-{table_name}", ls_batting.to_pyarrow())[
        lambda t: t.stint == 1
    ].into_backend(pg)

    expr = pg_t.join(
        db_t,
        db_t.columns,
    ).cache(
        cache=ParquetCache.from_kwargs(
            source=ls_con,
            relative_path=tmp_path,
        )
    )

    assert expr.execute() is not None


def test_multi_backend(parquet_dir):
    batting = xo.deferred_read_parquet(
        parquet_dir.joinpath("batting.parquet"), con=xo.connect()
    )
    awards_players = xo.deferred_read_parquet(
        parquet_dir.joinpath("awards_players.parquet"), con=xo.connect()
    )
    on = set(batting.columns).intersection(awards_players.columns)
    joined = batting.select(on).join(awards_players.select(on), predicates=on)
    with pytest.raises(XorqError, match="Multiple backends"):
        joined.execute()


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(lambda: xo.connect(), id="xorq_datafusion"),
        # resolved inside the test: the library job collects this module
        # from the bare wheel, where the duckdb extra is absent
        pytest.param(lambda: xo.duckdb.connect(), id="duckdb"),
    ],
)
def test_into_backend_window_used_twice(target):
    # GH #2415: the window is hoisted into a CTE that the cross join references
    # twice; datafusion inlines CTEs and reads the into_backend stream twice,
    # duckdb materialises the CTE and reads it once
    df = pd.DataFrame({"k": [1, 2], "v": [10.0, 20.0]})
    t = xo.pandas.connect().create_table("t", df).into_backend(target())
    t = t.mutate(r=xo.rank().over(order_by="v"))

    a = t.filter(t.k == 1).select(a=t.r)
    b = t.filter(t.k == 2).select(b=t.r)
    actual = a.cross_join(b).execute()

    assert_frame_equal(actual, pd.DataFrame({"a": [0], "b": [1]}), check_dtype=False)


def test_into_backend_self_join_used_twice_datafusion():
    # the CTE holds a self join (two scans) and is referenced twice: four reads
    df = pd.DataFrame({"k": [1, 2], "v": [10.0, 20.0]})
    t = xo.pandas.connect().create_table("t", df).into_backend(xo.connect())
    t = t.join(t.view(), "k").select("k", "v")

    a = t.filter(t.k == 1).select(a=t.v)
    b = t.filter(t.k == 2).select(b=t.v)
    actual = a.cross_join(b).execute()

    assert_frame_equal(
        actual, pd.DataFrame({"a": [10.0], "b": [20.0]}), check_dtype=False
    )


def test_into_backend_select_used_twice_datafusion():
    # control: no CTE, the stream appears in both join inputs and is counted
    df = pd.DataFrame({"k": [1, 2], "v": [10.0, 20.0]})
    t = xo.pandas.connect().create_table("t", df).into_backend(xo.connect())
    t = t.select("k", "v")

    a = t.filter(t.k == 1).select(a=t.v)
    b = t.filter(t.k == 2).select(b=t.v)
    actual = a.cross_join(b).execute()

    assert_frame_equal(
        actual, pd.DataFrame({"a": [10.0], "b": [20.0]}), check_dtype=False
    )


def test_into_backend_projects_extra_physical_column():
    """GH #2055: a remote stream carrying a column the logical schema omits is projected away, neither leaked nor warned about."""
    source = xo.duckdb.connect()
    source.create_table("phys", pa.table({"a": [1, 2, 3], "row_number": [0, 1, 2]}))
    logical = ops.DatabaseTable(
        name="phys", schema=ibis.schema({"a": "int64"}), source=source
    ).to_expr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = logical.into_backend(xo.connect()).execute()
    assert not [w for w in caught if "extra columns" in str(w.message)]
    assert_frame_equal(
        actual.sort_values("a", ignore_index=True),
        pd.DataFrame({"a": [1, 2, 3]}),
        check_dtype=False,
    )
