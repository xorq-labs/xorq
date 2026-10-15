from __future__ import annotations

import pandas as pd
import pytest

import xorq.api as xo
from xorq.tests.util import assert_frame_equal


def test_where_multiple_conditions(alltypes, alltypes_df):
    expr = alltypes.filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    result = expr.execute()

    expected = alltypes_df[
        (alltypes_df["float_col"] > 0)
        & (alltypes_df["smallint_col"] == 9)
        & (alltypes_df["int_col"] < alltypes_df["float_col"] * 2)
    ]

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "connect",
    [
        pytest.param(xo.connect, id="xorq_datafusion"),
        pytest.param(xo.sqlite.connect, id="sqlite"),
        pytest.param(xo.pandas.connect, id="pandas"),
    ],
)
@pytest.mark.parametrize(
    "labels",
    [pytest.param(["b", "a"], id="agree"), pytest.param(["a", "b"], id="conflict")],
)
def test_order_by_then_select_alias_shadowing_sort_column(connect, labels):
    # https://github.com/xorq-labs/xorq/issues/2407
    # the "conflict" labels sort the other way round from cohort_month, so
    # ordering by the output alias instead of the source column would fail
    df = pd.DataFrame({"cohort_month": [2, 1], "cohort_label": labels})
    t = connect().create_table("t", df)
    expr = t.order_by("cohort_month").select(cohort_month=t.cohort_label)
    assert expr.execute()["cohort_month"].tolist() == labels[::-1]


def test_order_by_expressions_then_select_alias_shadowing_sort_column():
    con = xo.connect()
    t = con.create_table(
        "t",
        pd.DataFrame(
            {"cohort_month": [1, 2, 3], "cohort_label": ["a", "b", "c"], "k": [1, 1, 2]}
        ),
    )
    other = con.create_table("other", pd.DataFrame({"k": [1, 2], "x": [2, 1]}))
    # a correlated scalar subquery and an expression over the shadowed column;
    # the rows with k=1 are inserted against the second key's order
    x = other.filter(other.k == t.k).x.max().as_scalar()
    expr = t.order_by([x, -t.cohort_month]).select(cohort_month=t.cohort_label)
    assert expr.execute()["cohort_month"].tolist() == ["c", "b", "a"]
