import random
from pathlib import Path

import ibis
import numpy as np
import pandas as pd
import pytest
from pytest import param

from letsql.tests.util import (
    assert_frame_equal,
)


def _pandas_semi_join(left, right, on, **_):
    assert len(on) == 1, str(on)
    inner = pd.merge(left, right, how="inner", on=on)
    filt = left.loc[:, on[0]].isin(inner.loc[:, on[0]])
    return left.loc[filt, :]


def _pandas_anti_join(left, right, on, **_):
    inner = pd.merge(left, right, how="left", indicator=True, on=on)
    return inner[inner["_merge"] == "left_only"]


IMPLS = {
    "semi": _pandas_semi_join,
    "anti": _pandas_anti_join,
}


def check_eq(left, right, how, **kwargs):
    impl = IMPLS.get(how, pd.merge)
    return impl(left, right, how=how, **kwargs)


@pytest.fixture
def union_subsets(alltypes, alltypes_df):
    cols_a, cols_b, cols_c = (alltypes.columns.copy() for _ in range(3))

    random.seed(89)
    random.shuffle(cols_a)
    random.shuffle(cols_b)
    random.shuffle(cols_c)
    assert cols_a != cols_b != cols_c

    cols_a = [ca for ca in cols_a if ca != "timestamp_col"]
    cols_b = [cb for cb in cols_b if cb != "timestamp_col"]
    cols_c = [cc for cc in cols_c if cc != "timestamp_col"]

    a = alltypes.filter((alltypes.id >= 5200) & (alltypes.id <= 5210))[cols_a]
    b = alltypes.filter((alltypes.id >= 5205) & (alltypes.id <= 5215))[cols_b]
    c = alltypes.filter((alltypes.id >= 5213) & (alltypes.id <= 5220))[cols_c]

    da = alltypes_df[(alltypes_df.id >= 5200) & (alltypes_df.id <= 5210)][cols_a]
    db = alltypes_df[(alltypes_df.id >= 5205) & (alltypes_df.id <= 5215)][cols_b]
    dc = alltypes_df[(alltypes_df.id >= 5213) & (alltypes_df.id <= 5220)][cols_c]

    return (a, b, c), (da, db, dc)


@pytest.fixture(scope="session")
def csv_dir():
    root = Path(__file__).absolute().parents[5]
    data_dir = root / "ci" / "ibis-testing-data" / "csv"
    return data_dir


@pytest.fixture(scope="session")
def parquet_dir():
    root = Path(__file__).absolute().parents[5]
    data_dir = root / "ci" / "ibis-testing-data" / "parquet"
    return data_dir


@pytest.fixture(scope="session")
def dirty_duckdb_con(csv_dir):
    con = ibis.duckdb.connect()
    con.read_csv(csv_dir / "awards_players.csv", table_name="ddb_players")
    con.read_csv(csv_dir / "batting.csv", table_name="batting")
    return con


@pytest.fixture(scope="function")
def duckdb_con(dirty_duckdb_con):
    expected_tables = ("ddb_players", "batting")
    for table in dirty_duckdb_con.list_tables():
        if table not in expected_tables:
            dirty_duckdb_con.drop_view(table)
    yield dirty_duckdb_con


def test_join(con, alltypes, alltypes_df):
    first_10 = alltypes_df.head(10)
    in_memory = con.register(first_10, table_name="in_memory")
    expr = alltypes.join(in_memory, predicates=[alltypes.id == in_memory.id])
    expected = pd.merge(
        alltypes_df, first_10, how="inner", on="id", suffixes=("", "_right")
    ).sort_values("id")
    actual = expr.execute().sort_values("id")

    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_filtering_join(con, batting, awards_players, how):
    left = batting[batting.yearID == 2015]
    right = awards_players[awards_players.lgID == "NL"].drop("yearID", "lgID")

    left_df = left.execute()
    right_df = right.execute()
    predicate = ["playerID"]
    result_order = ["playerID", "yearID", "lgID", "stint"]

    right = con.register(right_df, table_name="right")

    expr = left.join(right, predicate, how=how)
    result = (
        expr.execute()
        .fillna(np.nan)
        .sort_values(result_order)[left.columns]
        .reset_index(drop=True)
    )

    expected = check_eq(
        left_df,
        right_df,
        how=how,
        on=predicate,
        suffixes=("", "_y"),
    ).sort_values(result_order)[list(left.columns)]

    assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize("distinct", [False, True], ids=["all", "distinct"])
def test_union(con, union_subsets, distinct):
    (a, _, c), (da, db, dc) = union_subsets

    b = con.register(db, table_name="b")
    expr = ibis.union(a, b, distinct=distinct).order_by("id")
    result = expr.execute()

    expected = pd.concat([da, db], axis=0).sort_values("id").reset_index(drop=True)

    if distinct:
        expected = expected.drop_duplicates("id")

    assert_frame_equal(result, expected)


def test_union_mixed_distinct(con, union_subsets):
    (a, _, _), (da, db, dc) = union_subsets

    b = con.register(db, table_name="b")
    c = con.register(dc, table_name="c")

    expr = a.union(b, distinct=True).union(c, distinct=False).order_by("id")
    result = expr.execute()
    expected = pd.concat(
        [pd.concat([da, db], axis=0).drop_duplicates("id"), dc], axis=0
    ).sort_values("id")

    assert_frame_equal(result, expected)


def test_register_already_existing_table(con, batting):
    own_batting = con.register(batting, "own_batting")
    own_batting.execute()


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
def test_join_non_trivial_filters(ls_con, pg, duckdb_con, left_filter):
    awards_players = ls_con.register(duckdb_con.table("ddb_players"), "ddb_players")
    batting = ls_con.register(pg.table("batting"), table_name="pg-batting")

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
    con, duckdb_con, awards_players, predicate, how, pandas_value
):
    ddb_players = con.register(
        duckdb_con.table("ddb_players"), table_name="ddb_players"
    )

    n = 5

    base = awards_players.limit(n)
    ddb_base = ddb_players.limit(n)

    left = base.select(left_key="playerID")
    right = ddb_base.select(right_key="playerID")

    left_df = pd.DataFrame({"key": [True] * n})
    right_df = pd.DataFrame({"key": [pandas_value] * n})

    expected = pd.merge(left_df, right_df, on="key", how=how)

    expr = left.join(right, predicate, how=how)
    result = expr.to_pandas()

    assert len(result) == len(expected)


def test_sql_execution(con, duckdb_con, awards_players, batting):
    ddb_players = con.register(
        duckdb_con.table("ddb_players"), table_name="ddb_players"
    )

    left = batting[batting.yearID == 2015]
    right = awards_players[awards_players.lgID == "NL"].drop("yearID", "lgID")
    right_df = right.execute()
    left_df = left.execute()
    predicate = ["playerID"]
    result_order = ["playerID", "yearID", "lgID", "stint"]

    right = ddb_players[ddb_players.lgID == "NL"].drop("yearID", "lgID")
    expr = left.join(right, predicate, how="inner")
    query = ibis.to_sql(expr, dialect="datafusion")

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


def test_multiple_execution_letsql_register_table(con, csv_dir):
    table_name = "csv_players"
    t = con.register(csv_dir / "awards_players.csv", table_name=table_name)
    con.register(t, table_name=f"{con.name}_{table_name}")

    assert (first := t.execute()) is not None
    assert (second := t.execute()) is not None
    assert_frame_equal(first, second)


@pytest.mark.parametrize(
    "other_con",
    [
        ibis.datafusion.connect(),
        ibis.duckdb.connect(),
        ibis.postgres.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="postgres",
            database="ibis_testing",
        ),
    ],
)
def test_expr_over_same_table_multiple_times(con, parquet_dir, other_con):
    batting_path = parquet_dir.joinpath("batting.parquet")
    table_name = "batting"
    col = "playerID"

    batting_name = f"{con.name}_{table_name}"
    batting = con.register(batting_path, table_name=batting_name)

    if other_con.name == "postgres":
        t = other_con.table(table_name)
    else:
        t = other_con.register(batting_path, table_name=table_name)

    ls_table_name = f"{other_con.name}_{table_name}"
    con.register(t, ls_table_name)
    other = con.table(ls_table_name)

    expr = batting[[col]].distinct().join(other[[col]].distinct(), col)

    assert (first := expr.execute()) is not None
    assert (second := expr.execute()) is not None
    assert_frame_equal(first.sort_values(col), second.sort_values(col))


@pytest.mark.parametrize(
    "get_expr",
    [
        lambda t: t,
        lambda t: t.group_by("playerID").agg(t.stint.max().name("n-stints")),
    ],
)
def test_register_with_different_name(ls_con, duckdb_con, get_expr):
    table_name = "batting"
    letsql_table_name = f"{duckdb_con.name}_{table_name}"

    t = duckdb_con.table(table_name)
    ls_con.register(t, table_name=letsql_table_name)

    table = ls_con.table(letsql_table_name)
    expr = get_expr(table)

    assert table_name != letsql_table_name
    assert expr.execute() is not None


def test_register_with_different_name_more_than_one_table_expr(con, duckdb_con):
    batting_table_name = "batting"

    t = duckdb_con.table(batting_table_name)
    ddb_batting_table_name = f"{duckdb_con.name}_{batting_table_name}"
    con.register(t, table_name=ddb_batting_table_name)

    players_table_name = "ddb_players"
    ddb_players_table_name = f"{duckdb_con.name}_{players_table_name}"
    con.register(
        duckdb_con.table(players_table_name), table_name=ddb_players_table_name
    )

    batting_table = con.table(ddb_batting_table_name)
    awards_players_table = con.table(ddb_players_table_name)

    left = batting_table[batting_table.yearID == 2015]
    right = awards_players_table[awards_players_table.lgID == "NL"].drop(
        "yearID", "lgID"
    )

    left_df = left.execute()
    right_df = right.execute()
    predicate = ["playerID"]
    result_order = ["playerID", "yearID", "lgID", "stint"]

    expr = left.join(right, predicate, how="inner")
    result = (
        expr.execute()
        .fillna(np.nan)
        .sort_values(result_order)[left.columns]
        .reset_index(drop=True)
    )

    expected = check_eq(
        left_df,
        right_df,
        how="inner",
        on=predicate,
        suffixes=("", "_y"),
    ).sort_values(result_order)[list(left.columns)]

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


def test_register_arbitrary_expression_multiple_tables(con, duckdb_con):
    batting_table_name = "batting"
    batting_table = duckdb_con.table(batting_table_name)

    players_table_name = "ddb_players"
    awards_players_table = duckdb_con.table(players_table_name)

    left = batting_table[batting_table.yearID == 2015]
    right = awards_players_table[awards_players_table.lgID == "NL"].drop(
        "yearID", "lgID"
    )

    left_df = left.execute()
    right_df = right.execute()
    predicate = ["playerID"]
    result_order = ["playerID", "yearID", "lgID", "stint"]

    expr = left.join(right, predicate, how="inner")
    table = con.register(expr, table_name="expr_table")

    result = (
        table.execute()
        .fillna(np.nan)
        .sort_values(result_order)[left.columns]
        .reset_index(drop=True)
    )

    expected = check_eq(
        left_df,
        right_df,
        how="inner",
        on=predicate,
        suffixes=("", "_y"),
    ).sort_values(result_order)[list(left.columns)]

    assert_frame_equal(result, expected, check_like=True)


def test_multiple_pipes(ls_con, pg):
    """This test address the issue reported on bug #69
    link: https://github.com/letsql/letsql/issues/69

    NOTE
    The previous tests didn't catch it because the con object registered the table batting.
    In this test (and the rest) ls_con is a clean (no tables) letsql connection
    """

    duckdb_con = ibis.duckdb.connect()
    table_name = "batting"
    pg_t = pg.table(table_name)[lambda t: t.yearID == 2015].pipe(
        ls_con.register, f"pg-{table_name}"
    )
    db_t = duckdb_con.register(pg_t.to_pyarrow(), f"{table_name}")[
        lambda t: t.yearID == 2014
    ].pipe(ls_con.register, f"db-{table_name}")

    expr = pg_t.join(
        db_t,
        "playerID",
    )

    assert expr.execute() is not None
