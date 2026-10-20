from __future__ import annotations

import pandas as pd
import pytest

import xorq.api as xo
import xorq.common.exceptions as com
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


@pytest.mark.parametrize(
    ("make_expr", "expected"),
    [
        pytest.param(
            lambda t: t.order_by([t.m, t.k.sum().over()]).select(x=t.l),
            ["a", "b", "b", "c"],
            id="window_sort_key",
        ),
        pytest.param(
            lambda t: t.distinct().order_by("m").select(x=t.l),
            ["a", "b", "c"],
            id="distinct_parent",
        ),
    ],
)
def test_order_by_then_select_keeps_sort_in_subquery(make_expr, expected):
    # the projection is not merged into the sorted select (window function in
    # the keys, distinct parent), and DataFusion drops a subquery ORDER BY;
    # the rows are inserted against the sorted order
    df = pd.DataFrame({"m": [2, 2, 1, 3], "k": [1, 1, 1, 2], "l": ["b", "b", "a", "c"]})
    t = xo.connect().create_table("t", df)
    assert make_expr(t).execute()["x"].tolist() == expected


def test_order_by_then_select_correlated_subquery_over_sorted_select():
    # the scalar subquery reads the sorted select itself, not the base table
    con = xo.connect()
    t = con.create_table("t", pd.DataFrame({"m": [2, 1, 3], "k": [2, 1, 2]}))
    other = con.create_table("other", pd.DataFrame({"k": [1, 2], "x": [2, 1]}))
    s = t.order_by([t.m, t.k.sum().over()])
    expr = s.select(x=other.filter(other.k == s.k).x.max().as_scalar())
    assert expr.execute()["x"].tolist() == [2, 1, 1]


def test_order_by_then_select_scalar_subquery_over_sorted_select():
    # the scalar subquery reads the sorted select as its own input, so it
    # must keep that select's one-column schema and its ORDER BY
    t = xo.connect().create_table("t", pd.DataFrame({"m": [2, 2, 1, 3]}))
    s = t.distinct().order_by("m")
    expr = s.select(m=s.m, first=s.limit(1).as_scalar())
    result = expr.execute()
    assert result["m"].tolist() == [1, 2, 3]
    assert result["first"].tolist() == [1, 1, 1]


def test_order_by_then_filter_in_subquery_over_sorted_select():
    # the IN subquery reads the sorted select directly, with no relation in
    # between, while its needle reads the outer row
    t = xo.connect().create_table("t", pd.DataFrame({"m": [2, 2, 1, 3]}))
    s = t.distinct().order_by("m")
    expr = s.filter(s.m.isin(s.m))
    assert expr.execute()["m"].tolist() == [1, 2, 3]


def test_distinct_after_select_dropping_sort_column_is_refused():
    # the sort column is dropped by the projection, so the order of the
    # de-duplicated rows is undefined: refuse before the engine does
    t = xo.connect().create_table("t", pd.DataFrame({"a": [2, 1], "b": ["x", "y"]}))
    expr = t.order_by("a").select(a=t.b).distinct()
    with pytest.raises(
        com.UnsupportedOperationError, match="DISTINCT.*ORDER BY.*does not select"
    ):
        expr.execute()
