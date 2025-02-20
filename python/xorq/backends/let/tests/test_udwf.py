import numpy as np
import pyarrow as pa

import xorq as xo
from xorq.internal import WindowEvaluator, udwf


class ExponentialSmoothDefault(WindowEvaluator):
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


class ExponentialSmoothBounded(WindowEvaluator):
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


class ExponentialSmoothRank(WindowEvaluator):
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
    """This class demonstrates using two columns.

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


smooth_default = udwf(
    ExponentialSmoothDefault(0.9),
    pa.float64(),
    pa.float64(),
    volatility="immutable",
    name="smooth_default",
)

smooth_bounded = udwf(
    ExponentialSmoothBounded(0.9),
    pa.float64(),
    pa.float64(),
    volatility="immutable",
    name="smooth_bounded",
)

smooth_rank = udwf(
    ExponentialSmoothRank(0.9),
    pa.utf8(),
    pa.float64(),
    volatility="immutable",
    name="smooth_rank",
)

smooth_frame = udwf(
    ExponentialSmoothFrame(0.9),
    pa.float64(),
    pa.float64(),
    volatility="immutable",
    name="smooth_frame",
)

smooth_two_col = udwf(
    SmoothTwoColumn(0.9),
    [pa.int64(), pa.int64()],
    pa.float64(),
    volatility="immutable",
    name="smooth_two_col",
)


def test_smooth_frame(df):
    con = xo.connect()
    con.register(df, table_name="t")
    con.register_udwf(smooth_frame)

    query = """
    SELECT
      smooth_frame("t0"."a") OVER (PARTITION BY "t0"."c") AS "smoothed"
    FROM "t" AS "t0"
    """
    actual = con.sql(query).execute()

    assert actual is not None


def test_smooth_default(df):
    con = xo.connect()
    con.register(df, table_name="t")
    con.register_udwf(smooth_default)

    query = """
    SELECT
      smooth_default("t0"."a") OVER ()
    FROM "t" AS "t0"
    """
    actual = con.sql(query).execute()

    assert actual is not None


def test_smooth_default_partitioned(df):
    con = xo.connect()
    con.register(df, table_name="t")
    con.register_udwf(smooth_default)

    query = """
    SELECT
      smooth_default("t0"."a") OVER (PARTITION BY "t0"."c")
    FROM "t" AS "t0"
    """
    actual = con.sql(query).execute()

    assert actual is not None


def test_smooth_default_ordered(df):
    con = xo.connect()
    con.register(df, table_name="t")
    con.register_udwf(smooth_default)

    query = """
        SELECT
           smooth_default("t0"."a") OVER (ORDER BY "t0"."b")
        FROM "t" AS "t0"
        """
    actual = con.sql(query).execute()

    assert actual is not None


def test_smooth_bounded(df):
    con = xo.connect()
    con.register(df, table_name="t")
    con.register_udwf(smooth_bounded)

    query = """
        SELECT
           smooth_bounded("t0"."a") OVER ()
        FROM "t" AS "t0"
        """
    actual = con.sql(query).execute()

    assert actual is not None


def test_smooth_bounded_ignores_frame(df):
    con = xo.connect()
    con.register(df, table_name="t")
    con.register_udwf(smooth_bounded)

    query = """
        SELECT
           "t0"."a", smooth_bounded("t0"."a") OVER (ROWS UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as "udwf"
        FROM "t" AS "t0"
        ORDER BY "t0"."a"
        """
    result = con.sql(query).execute()
    actual = result["udwf"].to_list()

    np.testing.assert_allclose(
        actual,
        [0, 0.9, 1.9, 2.9, 3.9, 4.9, 5.9],
    )


def test_smooth_frame_bounded(df):
    con = xo.connect()
    con.register(df, table_name="t")
    con.register_udwf(smooth_frame)

    query = """
        SELECT
           "t0"."a", smooth_frame("t0"."a") OVER (ORDER BY "t0"."b" ROWS UNBOUNDED PRECEDING AND CURRENT ROW) as "udwf"
        FROM "t" AS "t0"
        ORDER BY "t0"."a"
        """
    result = con.sql(query).execute()
    actual = result["udwf"].to_list()

    np.testing.assert_allclose(
        actual, [0.551, 1.13, 2.3, 2.755, 3.876, 5.0, 5.513], rtol=1e-3
    )


def test_smooth_frame_unbounded(df):
    con = xo.connect()
    con.register(df, table_name="t")
    con.register_udwf(smooth_frame)

    query = """
        SELECT
           "t0"."a", smooth_frame("t0"."a") OVER (ROWS UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as "udwf"
        FROM "t" AS "t0"
        ORDER BY "t0"."a"
        """
    result = con.sql(query).execute()
    actual = result["udwf"].to_list()

    np.testing.assert_allclose(
        actual, [5.889, 5.889, 5.889, 5.889, 5.889, 5.889, 5.889], rtol=1e-3
    )


def test_smooth_rank(df):
    con = xo.connect()
    con.register(df, table_name="t")
    con.register_udwf(smooth_rank)

    query = """
        SELECT
           smooth_rank("t0"."c") OVER (ORDER BY "t0"."c")
        FROM "t" AS "t0"
        """
    actual = con.sql(query).execute()

    assert actual is not None


def test_smooth_two_column(df):
    con = xo.connect()
    con.register(df, table_name="t")
    con.register_udwf(smooth_two_col)

    query = """
        SELECT
           smooth_two_col("t0"."a", "t0"."b") OVER ()
        FROM "t" AS "t0"
        """
    actual = con.sql(query).execute()

    assert actual is not None
