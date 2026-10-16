import pandas as pd
import pytest

import xorq.api as xo


@pytest.mark.parametrize(
    "labels",
    [pytest.param(["b", "a"], id="agree"), pytest.param(["a", "b"], id="conflict")],
)
def test_order_by_then_select_alias_shadowing_sort_column(labels):
    # https://github.com/xorq-labs/xorq/issues/2407, on the vendored ibis compiler
    df = pd.DataFrame({"cohort_month": [2, 1], "cohort_label": labels})
    t = xo.datafusion.connect().create_table("t", df)
    expr = t.order_by("cohort_month").select(cohort_month=t.cohort_label)
    assert expr.execute()["cohort_month"].tolist() == labels[::-1]


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
    # DataFusion drops a subquery ORDER BY, on the vendored ibis compiler;
    # the rows are inserted against the sorted order
    df = pd.DataFrame({"m": [2, 2, 1, 3], "k": [1, 1, 1, 2], "l": ["b", "b", "a", "c"]})
    t = xo.datafusion.connect().create_table("t", df)
    assert make_expr(t).execute()["x"].tolist() == expected


def test_order_by_then_select_correlated_subquery_over_sorted_select():
    con = xo.datafusion.connect()
    t = con.create_table("t", pd.DataFrame({"m": [2, 1, 3], "k": [2, 1, 2]}))
    other = con.create_table("other", pd.DataFrame({"k": [1, 2], "x": [2, 1]}))
    s = t.order_by([t.m, t.k.sum().over()])
    expr = s.select(x=other.filter(other.k == s.k).x.max().as_scalar())
    assert expr.execute()["x"].tolist() == [2, 1, 1]


def test_order_by_then_select_scalar_subquery_over_sorted_select():
    # the scalar subquery reads the sorted select as its own input, so it
    # must keep that select's one-column schema and its ORDER BY
    t = xo.datafusion.connect().create_table("t", pd.DataFrame({"m": [2, 2, 1, 3]}))
    s = t.distinct().order_by("m")
    expr = s.select(m=s.m, first=s.limit(1).as_scalar())
    result = expr.execute()
    assert result["m"].tolist() == [1, 2, 3]
    assert result["first"].tolist() == [1, 1, 1]


def test_order_by_then_filter_in_subquery_over_sorted_select():
    t = xo.datafusion.connect().create_table("t", pd.DataFrame({"m": [2, 2, 1, 3]}))
    s = t.distinct().order_by("m")
    expr = s.filter(s.m.isin(s.m))
    assert expr.execute()["m"].tolist() == [1, 2, 3]
