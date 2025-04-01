import numpy as np
import pyarrow as pa
import pytest

import xorq as xo
from xorq.expr.udf import pyarrow_udwf
from xorq.internal import WindowEvaluator
from xorq.vendor import ibis
from xorq.vendor.ibis import _


class SmoothBoundedFromPreviousRow(WindowEvaluator):
    """Smooth over from the previous to current row only."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def supports_bounded_execution(self) -> bool:
        return True

    def get_range(self, idx: int, num_rows: int) -> tuple[int, int]:
        # Override the default range of current row since uses_window_frame is False
        # So for the purpose of this test we just smooth from the previous row to
        # current.
        if idx == 0:
            return (0, 0)
        return (idx - 1, idx)

    def evaluate(
        self, values: list[pa.Array], eval_range: tuple[int, int]
    ) -> pa.Scalar:
        (start, stop) = eval_range
        curr_value = 0.0
        values = values[0]
        for idx in range(start, stop + 1):
            if idx == start:
                curr_value = values[idx].as_py()
            else:
                curr_value = values[idx].as_py() * self.alpha + curr_value * (
                    1.0 - self.alpha
                )
        return pa.scalar(curr_value).cast(pa.float64())


@pyarrow_udwf(
    schema=ibis.schema({"a": float}),
    return_type=ibis.dtype(float),
    alpha=0.9,
)
def exp_smooth(self, values: list[pa.Array], num_rows: int) -> pa.Array:
    results = []
    curr_value = 0.0
    values = values[0]
    for idx in range(num_rows):
        if idx == 0:
            curr_value = values[idx].as_py()
        else:
            curr_value = values[idx].as_py() * self.alpha + curr_value * (
                1.0 - self.alpha
            )
        results.append(curr_value)

    return pa.array(results)


@pyarrow_udwf(
    schema=ibis.schema({"a": float}),
    return_type=ibis.dtype(float),
    supports_bounded_execution=True,
    get_range=SmoothBoundedFromPreviousRow.get_range,
    alpha=0.9,
)
def smooth_two_row(
    self, values: list[pa.Array], eval_range: tuple[int, int]
) -> pa.Scalar:
    (start, stop) = eval_range
    curr_value = 0.0
    values = values[0]
    for idx in range(start, stop + 1):
        if idx == start:
            curr_value = values[idx].as_py()
        else:
            curr_value = values[idx].as_py() * self.alpha + curr_value * (
                1.0 - self.alpha
            )
    return pa.scalar(curr_value).cast(pa.float64())


@pyarrow_udwf(
    schema=ibis.schema({"a": float}),
    return_type=ibis.dtype(float),
    alpha=0.9,
    include_rank=True,
)
def smooth_rank(
    self, num_rows: int, ranks_in_partition: list[tuple[int, int]]
) -> pa.Array:
    results = []
    for idx in range(num_rows):
        if idx == 0:
            prior_value = 1.0
        matching_row = [
            i
            for i in range(len(ranks_in_partition))
            if ranks_in_partition[i][0] <= idx and ranks_in_partition[i][1] > idx
        ][0] + 1
        curr_value = matching_row * self.alpha + prior_value * (1.0 - self.alpha)
        results.append(curr_value)
        prior_value = matching_row

    return pa.array(results)


@pyarrow_udwf(
    schema=ibis.schema({"a": float}),
    return_type=ibis.dtype(float),
    alpha=0.9,
    uses_window_frame=True,
)
def smooth_frame(
    self, values: list[pa.Array], eval_range: tuple[int, int]
) -> pa.Scalar:
    (start, stop) = eval_range
    curr_value = 0.0
    if len(values) > 1:
        order_by = values[1]  # noqa: F841
        values = values[0]
    else:
        values = values[0]
    for idx in range(start, stop):
        if idx == start:
            curr_value = values[idx].as_py()
        else:
            curr_value = values[idx].as_py() * self.alpha + curr_value * (
                1.0 - self.alpha
            )
    return pa.scalar(curr_value).cast(pa.float64())


@pyarrow_udwf(
    schema=ibis.schema({"a": float, "b": int}),
    return_type=ibis.dtype(float),
    alpha=0.9,
)
def smooth_two_col(self, values: list[pa.Array], num_rows: int) -> pa.Array:
    results = []
    values_a = values[0]
    values_b = values[1]
    for idx in range(num_rows):
        if values_b[idx].as_py() > 7:
            if idx == 0:
                results.append(values_a[1].cast(pa.float64()))
            elif idx == num_rows - 1:
                results.append(values_a[num_rows - 2].cast(pa.float64()))
            else:
                results.append(
                    pa.scalar(
                        values_a[idx - 1].as_py() * self.alpha
                        + values_a[idx + 1].as_py() * (1.0 - self.alpha)
                    )
                )
        else:
            results.append(values_a[idx].cast(pa.float64()))

    return pa.array(results)


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
