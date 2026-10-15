import numpy as np
import pyarrow as pa

import xorq.api as xo
from xorq.backends.xorq_datafusion.tests.udwf_evaluators import (
    ExponentialSmoothBounded,
    ExponentialSmoothDefault,
    ExponentialSmoothFrame,
    ExponentialSmoothRank,
    SmoothTwoColumn,
)
from xorq.internal import udwf


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
