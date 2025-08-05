from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest import param

import xorq as xo
import xorq.vendor.ibis.expr.schema as sch
from xorq.tests.util import assert_frame_equal


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


@pytest.mark.parametrize(
    "how",
    [
        "inner",
        "left",
        "right",
        "outer",
    ],
)
def test_mutating_join(batting, awards_players, how):
    left = batting[batting.yearID == 2015]
    right = awards_players[awards_players.lgID == "NL"].drop("yearID", "lgID")

    left_df = left.execute()
    right_df = right.execute()
    predicate = ["playerID"]
    result_order = ["playerID", "yearID", "lgID", "stint"]

    expr = left.join(right, predicate, how=how)
    if how == "inner":
        result = (
            expr.execute()
            .fillna(np.nan)[left.columns]
            .sort_values(result_order)
            .reset_index(drop=True)
        )
    else:
        result = (
            expr.execute()
            .fillna(np.nan)
            .assign(
                playerID=lambda df: df.playerID.where(
                    df.playerID.notnull(),
                    df.playerID_right,
                )
            )
            .drop(["playerID_right"], axis=1)[left.columns]
            .sort_values(result_order)
            .reset_index(drop=True)
        )

    expected = (
        check_eq(
            left_df,
            right_df,
            how=how,
            on=predicate,
            suffixes=("_x", "_y"),
        )[left.columns]
        .sort_values(result_order)
        .reset_index(drop=True)
    )

    assert_frame_equal(result, expected, check_like=True)


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


def test_join_then_filter_no_column_overlap(awards_players, batting):
    left = batting[batting.yearID == 2015]
    year = left.yearID.name("year")
    left = left[year, "RBI"]
    right = awards_players[awards_players.lgID == "NL"]

    expr = left.join(right, left.year == right.yearID)
    filters = [expr.RBI == 9]
    q = expr.filter(filters)
    assert not q.execute().empty


def test_mutate_then_join_no_column_overlap(batting, awards_players):
    left = batting.mutate(year=batting.yearID).filter(lambda t: t.year == 2015)
    left = left["year", "RBI"]
    right = awards_players
    expr = left.join(right, left.year == right.yearID)
    assert not expr.limit(5).execute().empty


@pytest.mark.parametrize(
    "func",
    [
        param(lambda left, right: left.semi_join(right, "year"), id="method"),
        param(
            lambda left, right: left.join(right, "year", how="left_semi"),
            id="how_left_semi",
        ),
        param(lambda left, right: left.join(right, "year", how="semi"), id="how_semi"),
    ],
)
def test_semi_join_topk(batting, awards_players, func):
    batting = batting.mutate(year=batting.yearID)
    left = func(batting, batting.year.topk(5)).select("year", "RBI")
    expr = left.join(awards_players, left.year == awards_players.yearID)
    assert not expr.limit(5).execute().empty


def test_join_with_pandas(batting, awards_players):
    batting_filt = batting[lambda t: t.yearID < 1900]
    awards_players_filt = awards_players[lambda t: t.yearID < 1900].execute()
    assert isinstance(awards_players_filt, pd.DataFrame)
    expr = batting_filt.join(awards_players_filt, "yearID")
    df = expr.execute()
    assert df.yearID.nunique() == 7


def test_join_with_pandas_non_null_typed_columns(batting, awards_players):
    batting_filt = batting[lambda t: t.yearID < 1900][["yearID"]]
    awards_players_filt = awards_players[lambda t: t.yearID < 1900][
        ["yearID"]
    ].execute()

    # ensure that none of the columns of either table have type null
    batting_schema = batting_filt.schema()
    assert len(batting_schema) == 1
    assert batting_schema["yearID"].is_integer()

    assert sch.infer(awards_players_filt) == sch.Schema(dict(yearID="int"))
    assert isinstance(awards_players_filt, pd.DataFrame)
    expr = batting_filt.join(awards_players_filt, "yearID")
    df = expr.execute()
    assert df.yearID.nunique() == 7


@pytest.mark.parametrize(
    ("predicate", "pandas_value"),
    [
        # Trues
        param(True, True, id="true"),
        param(xo.literal(True), True, id="true-literal"),
        param([True], True, id="true-list"),
        param([xo.literal(True)], True, id="true-literal-list"),
        # only trues
        param([True, True], True, id="true-true-list"),
        param(
            [xo.literal(True), xo.literal(True)],
            True,
            id="true-true-literal-list",
        ),
        param([True, xo.literal(True)], True, id="true-true-const-expr-list"),
        param([xo.literal(True), True], True, id="true-true-expr-const-list"),
        # Falses
        param(False, False, id="false"),
        param(xo.literal(False), False, id="false-literal"),
        param([False], False, id="false-list"),
        param([xo.literal(False)], False, id="false-literal-list"),
        # only falses
        param([False, False], False, id="false-false-list"),
        param(
            [xo.literal(False), xo.literal(False)],
            False,
            id="false-false-literal-list",
        ),
        param([False, xo.literal(False)], False, id="false-false-const-expr-list"),
        param([xo.literal(False), False], False, id="false-false-expr-const-list"),
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
def test_join_with_trivial_predicate(awards_players, predicate, how, pandas_value):
    n = 5

    base = awards_players.limit(n)

    left = base.select(left_key="playerID")
    right = base.select(right_key="playerID")

    left_df = pd.DataFrame({"key": [True] * n})
    right_df = pd.DataFrame({"key": [pandas_value] * n})

    expected = pd.merge(left_df, right_df, on="key", how=how)

    expr = left.join(right, predicate, how=how)
    result = expr.to_pandas()

    assert len(result) == len(expected)
