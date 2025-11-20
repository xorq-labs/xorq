import numpy as np
import pytest

from xorq.tests.util import (
    assert_frame_equal,
    check_eq,
)


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


def test_multiple_execution_letsql_register_table(ls_con, csv_dir):
    table_name = "csv_players"
    t = ls_con.read_csv(csv_dir / "awards_players.csv", table_name=table_name)
    ls_con.register(t, table_name=f"{ls_con.name}_{table_name}")

    assert (first := t.execute()) is not None
    assert (second := t.execute()) is not None
    assert_frame_equal(first, second)
