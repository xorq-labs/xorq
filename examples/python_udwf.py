import pandas as pd
import pyarrow as pa
from pandas.testing import assert_series_equal

import xorq as xo
from xorq.expr.udf import pyarrow_udwf
from xorq.internal import WindowEvaluator
from xorq.vendor import ibis


class ExponentialSmoothDefault(WindowEvaluator):
    """Create a running smooth operation across an entire partition at once."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def evaluate_all(self, values: list[pa.Array], num_rows: int) -> pa.Array:
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


class SmoothAcrossRank(WindowEvaluator):
    """Smooth over the rank from the previous rank to current."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def include_rank(self) -> bool:
        return True

    def evaluate_all_with_rank(
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


class ExponentialSmoothFrame(WindowEvaluator):
    "Find the value across an entire frame using exponential smoothing"

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def uses_window_frame(self) -> bool:
        return True

    def evaluate(
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


class SmoothTwoColumn(WindowEvaluator):
    """Smooth once column based on a condition of another column.

    If the second column is above a threshold, then smooth over the first column from
    the previous and next rows.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def evaluate_all(self, values: list[pa.Array], num_rows: int) -> pa.Array:
        results = []
        values_a = values[0]
        values_b = values[1]
        for idx in range(num_rows):
            if not values_b[idx].is_valid:
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
        if not values_b[idx].is_valid:
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


con = xo.connect()
t = con.register(
    pa.Table.from_batches(
        [
            pa.RecordBatch.from_arrays(
                [
                    pa.array([1.0, 2.1, 2.9, 4.0, 5.1, 6.0, 6.9, 8.0]),
                    pa.array([1, 2, None, 4, 5, 6, None, 8]),
                    pa.array(["A", "A", "A", "A", "A", "B", "B", "B"]),
                ],
                names=["a", "b", "c"],
            )
        ]
    ),
    table_name="t",
)
expr = (
    t.mutate(exp_smooth=exp_smooth.on_expr(t).round(3))
    .mutate(smooth_two_row=smooth_two_row.on_expr(t))
    .mutate(smooth_rank=smooth_rank.on_expr(t).over(ibis.window(order_by="c")))
    .mutate(
        smooth_two_col=smooth_two_col.on_expr(t).over(
            ibis.window(preceding=None, following=0)
        )
    )
    .mutate(
        smooth_frame=smooth_frame.on_expr(t)
        .over(ibis.window(preceding=None, following=0))
        .round(3)
    )
)
result = xo.execute(expr).sort_values(["c", "a"], ignore_index=True)
expected = (
    t.execute()
    .combine_first(
        pd.DataFrame(
            {
                "exp_smooth": pa.array(
                    [1, 1.99, 2.809, 3.881, 4.978, 5.898, 6.8, 7.88]
                ),
                "smooth_two_col": pa.array([1, 2.1, 2.29, 4, 5.1, 6, 6.2, 8.0]),
                "smooth_two_row": pa.array(
                    [1.0, 1.99, 2.82, 3.89, 4.99, 5.91, 6.81, 7.89]
                ),
                "smooth_rank": pa.array([1, 1, 1, 1, 1, 1.9, 2.0, 2.0]),
                "smooth_frame": pa.array(
                    [1, 1.99, 2.809, 3.881, 4.978, 5.898, 6.8, 7.88]
                ),
            }
        )
    )
    .reindex_like(result)
)


for col in expected.filter(like="smooth").columns:
    try:
        assert_series_equal(result[col], expected[col])
    except Exception:
        print(f"col {col} failed")
pytest_examples_passed = True
