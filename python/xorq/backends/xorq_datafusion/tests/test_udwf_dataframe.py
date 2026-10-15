import numpy as np
import pytest

import xorq.api as xo
from xorq.backends.xorq_datafusion.tests.udwf_evaluators import (
    ExponentialSmoothBounded,
    ExponentialSmoothDefault,
    ExponentialSmoothFrame,
    ExponentialSmoothRank,
    SmoothTwoColumn,
)
from xorq.expr.udf import pyarrow_udwf
from xorq.vendor import ibis
from xorq.vendor.ibis import _


FLOAT_A = ibis.schema({"a": float})
FLOAT = ibis.dtype(float)

exp_smooth = pyarrow_udwf(
    ExponentialSmoothDefault.evaluate_all,
    schema=FLOAT_A,
    return_type=FLOAT,
    name="exp_smooth",
    alpha=0.9,
)

smooth_two_row = pyarrow_udwf(
    ExponentialSmoothBounded.evaluate,
    schema=FLOAT_A,
    return_type=FLOAT,
    name="smooth_two_row",
    supports_bounded_execution=True,
    get_range=ExponentialSmoothBounded.get_range,
    alpha=0.9,
)

smooth_rank = pyarrow_udwf(
    ExponentialSmoothRank.evaluate_all_with_rank,
    schema=FLOAT_A,
    return_type=FLOAT,
    name="smooth_rank",
    alpha=0.9,
    include_rank=True,
)

smooth_frame = pyarrow_udwf(
    ExponentialSmoothFrame.evaluate,
    schema=FLOAT_A,
    return_type=FLOAT,
    name="smooth_frame",
    alpha=0.9,
    uses_window_frame=True,
)

smooth_two_col = pyarrow_udwf(
    SmoothTwoColumn.evaluate_all,
    schema=ibis.schema({"a": float, "b": int}),
    return_type=FLOAT,
    name="smooth_two_col",
    alpha=0.9,
)


@pytest.mark.parametrize(
    "window,expected",
    [
        (ibis.window(), [0, 0.9, 1.89, 2.889, 3.889, 4.889, 5.889]),
        (ibis.window(group_by=_.c), [0, 0.9, 1.89, 2.889, 4.0, 4.9, 5.89]),
        (ibis.window(order_by=_.b), [0.551, 1.13, 2.3, 2.755, 3.876, 5.0, 5.513]),
    ],
)
def test_smooth_default(df, window, expected):
    con = xo.connect()
    t = con.register(df, table_name="t")

    expr = t.select(
        t.a,
        udwf=exp_smooth.on_expr(t).over(window),
    ).order_by(t.a)

    result = expr.execute()
    actual = result["udwf"].to_list()

    np.testing.assert_allclose(actual, expected, rtol=1e-3)


def test_smooth_bounded(df):
    con = xo.connect()
    t = con.register(df, table_name="t")

    expr = t.select(
        t.a,
        udwf=smooth_two_row.on_expr(t).over(ibis.window()),
    ).order_by(t.a)

    result = expr.execute()
    actual = result["udwf"].to_list()

    np.testing.assert_allclose(actual, [0, 0.9, 1.9, 2.9, 3.9, 4.9, 5.9], rtol=1e-3)


def test_smooth_two_column(df):
    con = xo.connect()
    t = con.register(df, table_name="t")

    expr = t.select(
        t.a,
        udwf=smooth_two_col.on_expr(t).over(ibis.window()),
    ).order_by(t.a)

    result = expr.execute()
    actual = result["udwf"].to_list()

    np.testing.assert_allclose(actual, [0.0, 1.0, 2.0, 2.2, 3.2, 5.0, 6.0], rtol=1e-3)


def test_smooth_rank(df):
    con = xo.connect()
    t = con.register(df, table_name="t")

    expr = t.select(
        t.a,
        udwf=smooth_rank.on_expr(t).over(ibis.window(order_by=t.c)),
    ).order_by(t.a)

    result = expr.execute()
    actual = result["udwf"].to_list()

    np.testing.assert_allclose(actual, [1, 1, 1, 1, 1.9, 2, 2], rtol=1e-3)


@pytest.mark.parametrize(
    "window,expected",
    [
        (ibis.window(), [5.889, 5.889, 5.889, 5.889, 5.889, 5.889, 5.889]),
        (ibis.cumulative_window(), [0.0, 0.9, 1.89, 2.889, 3.889, 4.889, 5.889]),
        (
            ibis.cumulative_window(order_by=_.b),
            [0.551, 1.13, 2.3, 2.755, 3.876, 5.0, 5.513],
        ),
    ],
)
def test_smooth_frame_bounded(df, window, expected):
    con = xo.connect()
    t = con.register(df, table_name="t")

    expr = t.select(
        t.a,
        udwf=smooth_frame.on_expr(t).over(window),
    ).order_by(t.a)

    result = expr.execute()
    actual = result["udwf"].to_list()

    np.testing.assert_allclose(actual, expected, rtol=1e-3)
