import numpy as np
import pandas as pd
import pytest

import xorq as xo
from xorq.tests.util import (
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


@pytest.fixture(scope="session")
def dirty_duckdb_con(csv_dir):
    con = xo.duckdb.connect()
    con.read_csv(csv_dir / "awards_players.csv", table_name="ddb_players")
    con.read_csv(csv_dir / "batting.csv", table_name="batting")
    return con


@pytest.fixture(scope="function")
def duckdb_con(dirty_duckdb_con):
    from duckdb import CatalogException

    expected_tables = ("ddb_players", "batting")
    for table in dirty_duckdb_con.list_tables():
        if table not in expected_tables:
            try:
                dirty_duckdb_con.drop_view(table, force=True)
            except CatalogException:
                dirty_duckdb_con.drop_table(table, force=True)
    yield dirty_duckdb_con


@pytest.fixture(scope="function")
def ddb_batting(duckdb_con):
    return duckdb_con.create_table(
        "db-batting",
        duckdb_con.table("batting").to_pyarrow(),
    )


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_filtering_join(batting, awards_players, how):
    left = batting[batting.yearID == 2015]
    right = awards_players[awards_players.lgID == "NL"].drop("yearID", "lgID")

    left_df = left.execute()
    right_df = right.execute()
    predicate = ["playerID"]
    result_order = ["playerID", "yearID", "lgID", "stint"]

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


def test_sql_execution(ls_con, duckdb_con, awards_players, batting):
    def make_right(t):
        return t[t.lgID == "NL"].drop("yearID", "lgID")

    ddb_players = ls_con.register(
        duckdb_con.table("ddb_players"), table_name="ddb_players"
    )

    left = batting[batting.yearID == 2015]
    right_df = make_right(awards_players).execute()
    left_df = xo.execute(left)
    predicate = ["playerID"]
    result_order = ["playerID", "yearID", "lgID", "stint"]

    expr = ls_con.register(left, "batting").join(
        make_right(ls_con.register(ddb_players, "players")),
        predicate,
        how="inner",
    )
    query = xo.to_sql(expr)

    result = (
        ls_con.sql(query)
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


def test_multiple_execution_letsql_register_table(ls_con, csv_dir):
    table_name = "csv_players"
    t = ls_con.read_csv(csv_dir / "awards_players.csv", table_name=table_name)
    ls_con.register(t, table_name=f"{ls_con.name}_{table_name}")

    assert (first := t.execute()) is not None
    assert (second := t.execute()) is not None
    assert_frame_equal(first, second)


def test_register_arbitrary_expression(ls_con, duckdb_con):
    batting_table_name = "batting"
    t = duckdb_con.table(batting_table_name)

    expr = t.filter(t.playerID == "allisar01")[
        ["playerID", "yearID", "stint", "teamID", "lgID"]
    ]
    expected = expr.execute()

    ddb_batting_table_name = f"{duckdb_con.name}_{batting_table_name}"
    table = ls_con.register(expr, table_name=ddb_batting_table_name)
    result = table.execute()

    assert result is not None
    assert_frame_equal(result, expected, check_like=True)
