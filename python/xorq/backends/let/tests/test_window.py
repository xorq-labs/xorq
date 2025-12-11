from __future__ import annotations

from functools import partial
from operator import methodcaller

import pytest
from pytest import param

import xorq.api as xo
import xorq.common.exceptions as com
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.tests.util import assert_frame_equal, assert_series_equal
from xorq.vendor import ibis
from xorq.vendor.ibis.legacy.udf.vectorized import analytic, reduction


np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


# adapted from https://gist.github.com/xmnlab/2c1f93df1a6c6bde4e32c8579117e9cc
def pandas_ntile(x, bucket: int):
    """Divide values into a number of buckets.

    It divides an ordered and grouped data set into a number of buckets
    and assigns the appropriate bucket number to each row.

    Return an integer ranging from 0 to `bucket - 1`, dividing the
    partition as equally as possible.
    """

    # internal ntile function
    def _ntile(x: pd.Series, bucket: int) -> pd.Series:
        n = x.shape[0]
        sub_n = n // bucket
        diff = n - (sub_n * bucket)

        result = []
        for i in range(bucket):
            sub_result = [i] * (sub_n + (1 if diff else 0))
            result.extend(sub_result)
            if diff > 0:
                diff -= 1
        return pd.Series(result, index=x.index)

    if isinstance(x, pd.core.groupby.generic.SeriesGroupBy):
        return x.apply(partial(_ntile, bucket=bucket))
    elif isinstance(x, pd.Series):
        return _ntile(x, bucket)

    raise TypeError(
        "`x` should be `pandas.Series` or `pandas.DataFrame` or "
        "`pd.core.groupby.generic.SeriesGroupBy` or "
        "`pd.core.groupby.generic.DataFrameGroupBy`, "
        f"not {type(x)}."
    )


with pytest.warns(FutureWarning, match="v9.0"):

    @reduction(input_type=[dt.double], output_type=dt.double)
    def mean_udf(s: pd.Series) -> float:
        return s.mean()

    @analytic(input_type=[dt.double], output_type=dt.double)
    def calc_zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std()


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        param(
            lambda t, win: t.float_col.lag().over(win),
            lambda t: t.float_col.shift(1),
            id="lag",
        ),
        param(
            lambda t, win: t.float_col.lead().over(win),
            lambda t: t.float_col.shift(-1),
            id="lead",
        ),
        param(
            lambda t, win: t.id.rank().over(win),
            lambda t: t.id.rank(method="min").astype("int64") - 1,
            id="rank",
        ),
        param(
            lambda t, win: t.id.dense_rank().over(win),
            lambda t: t.id.rank(method="dense").astype("int64") - 1,
            id="dense_rank",
        ),
        param(
            lambda t, win: t.id.percent_rank().over(win),
            lambda t: t.apply(
                lambda df: (
                    df.sort_values("id").id.rank(method="min").sub(1).div(len(df) - 1)
                )
            ).reset_index(drop=True, level=[0]),
            id="percent_rank",
        ),
        param(
            lambda t, win: t.id.cume_dist().over(win),
            lambda t: t.id.rank(method="min") / t.id.transform(len),
            id="cume_dist",
        ),
        param(
            lambda t, win: t.float_col.first().over(win),
            lambda t: t.float_col.transform("first"),
            id="first",
        ),
        param(
            lambda t, win: t.float_col.last().over(win),
            lambda t: t.float_col.transform("last"),
            id="last",
        ),
        param(
            lambda t, win: t.double_col.nth(3).over(win),
            lambda t: t.double_col.apply(
                lambda s: pd.concat(
                    [
                        pd.Series(np.nan, index=s.index[:3], dtype="float64"),
                        pd.Series(s.iloc[3], index=s.index[3:], dtype="float64"),
                    ]
                )
            ),
            id="nth",
        ),
        param(
            lambda _, win: ibis.row_number().over(win),
            lambda t: t.cumcount(),
            id="row_number",
        ),
        param(
            lambda t, win: t.double_col.cumsum().over(win),
            lambda t: t.double_col.cumsum(),
            id="cumsum",
        ),
        param(
            lambda t, win: t.double_col.cummean().over(win),
            lambda t: (t.double_col.expanding().mean().reset_index(drop=True, level=0)),
            id="cummean",
        ),
        param(
            lambda t, win: t.float_col.cummin().over(win),
            lambda t: t.float_col.cummin(),
            id="cummin",
        ),
        param(
            lambda t, win: t.float_col.cummax().over(win),
            lambda t: t.float_col.cummax(),
            id="cummax",
        ),
        param(
            lambda t, win: (t.double_col == 0).any().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: s.eq(0).any())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id="cumany",
        ),
        param(
            lambda t, win: (t.double_col == 0).notany().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: ~s.eq(0).any())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id="cumnotany",
        ),
        param(
            lambda t, win: (t.double_col == 0).all().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: s.eq(0).all())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id="cumall",
        ),
        param(
            lambda t, win: (t.double_col == 0).notall().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: ~s.eq(0).all())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id="cumnotall",
        ),
        param(
            lambda t, win: t.double_col.sum().over(win),
            lambda gb: gb.double_col.cumsum(),
            id="sum",
        ),
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda gb: (
                gb.double_col.expanding().mean().reset_index(drop=True, level=0)
            ),
            id="mean",
        ),
        param(
            lambda t, win: t.float_col.min().over(win),
            lambda gb: gb.float_col.cummin(),
            id="min",
        ),
        param(
            lambda t, win: t.float_col.max().over(win),
            lambda gb: gb.float_col.cummax(),
            id="max",
        ),
        param(
            lambda t, win: t.double_col.count().over(win),
            # pandas doesn't including the current row, but following=0 implies
            # that we must, so we add one to the pandas result
            lambda gb: gb.double_col.cumcount() + 1,
            id="count",
        ),
    ],
)
def test_grouped_bounded_expanding_window(
    alltypes, alltypes_df, result_fn, expected_fn
):
    expr = alltypes.mutate(
        val=result_fn(
            alltypes,
            win=xo.window(
                following=0,
                group_by=[alltypes.string_col],
                order_by=[alltypes.id],
            ),
        )
    )

    result = expr.execute().set_index("id").sort_index()
    column = expected_fn(
        alltypes_df.sort_values("id").groupby("string_col", group_keys=True)
    )
    if column.index.nlevels > 1:
        column = column.droplevel(0)
    expected = alltypes_df.assign(val=column).set_index("id").sort_index()

    left, right = result.val, expected.val

    assert_series_equal(left, right)


@pytest.mark.parametrize(
    ("preceding", "following"),
    [
        (0, 2),
        (None, (0, 2)),
    ],
    ids=["zero-two", "none-zero-two"],
)
def test_grouped_bounded_following_window(alltypes, alltypes_df, preceding, following):
    window = xo.window(
        preceding=preceding,
        following=following,
        group_by=[alltypes.string_col],
        order_by=[alltypes.id],
    )

    expr = alltypes.mutate(val=alltypes.id.mean().over(window))

    result = expr.execute().set_index("id").sort_index()

    # shift id column before applying Pandas rolling window summarizer to
    # simulate forward-looking window aggregation
    gdf = alltypes_df.sort_values("id").groupby("string_col")
    gdf.id = gdf.apply(lambda t: t.id.shift(-2))
    expected = (
        alltypes_df.assign(
            val=gdf.id.rolling(3, min_periods=1)
            .mean()
            .sort_index(level=1)
            .reset_index(drop=True)
        )
        .set_index("id")
        .sort_index()
    )

    # discard first 2 rows of each group to account for the shift
    n = len(gdf) * 2
    left, right = result.val.shift(-n), expected.val.shift(-n)

    assert_series_equal(left, right)


@pytest.mark.parametrize(
    "window_fn, window_size",
    [
        param(
            lambda t: xo.window(
                preceding=2,
                following=0,
                group_by=[t.string_col],
                order_by=[t.id],
            ),
            3,
            id="preceding-2-following-0",
        ),
        param(
            lambda t: xo.window(
                preceding=(2, 0),
                following=None,
                group_by=[t.string_col],
                order_by=[t.id],
            ),
            3,
            id="preceding-2-following-0-tuple",
        ),
        param(
            lambda t: ibis.trailing_window(
                preceding=2, group_by=[t.string_col], order_by=[t.id]
            ),
            3,
            id="trailing-2",
        ),
        param(
            lambda t: xo.window(
                # snowflake doesn't allow windows larger than 1000
                preceding=999,
                following=0,
                group_by=[t.string_col],
                order_by=[t.id],
            ),
            1000,
            id="large-preceding-999-following-0",
        ),
        param(
            lambda t: xo.window(
                preceding=1000, following=0, group_by=[t.string_col], order_by=[t.id]
            ),
            1001,
            id="large-preceding-1000-following-0",
        ),
    ],
)
def test_grouped_bounded_preceding_window(
    alltypes, alltypes_df, window_fn, window_size
):
    window = window_fn(alltypes)
    expr = alltypes.mutate(val=alltypes.double_col.sum().over(window))

    result = expr.execute().set_index("id").sort_index()
    gdf = alltypes_df.sort_values("id").groupby("string_col")
    expected = (
        alltypes_df.assign(
            val=gdf.double_col.rolling(window_size, min_periods=1)
            .sum()
            .sort_index(level=1)
            .reset_index(drop=True)
        )
        .set_index("id")
        .sort_index()
    )

    left, right = result.val, expected.val

    assert_series_equal(left, right)


@pytest.mark.parametrize(
    ("ibis_method_name", "pandas_fn"),
    [
        param("sum", lambda s: s.cumsum(), id="sum"),
        param("min", lambda s: s.cummin(), id="min"),
        param("mean", lambda s: s.expanding().mean(), id="mean"),
    ],
)
def test_simple_ungrouped_unbound_following_window(
    alltypes, ibis_method_name, pandas_fn
):
    ibis_method = methodcaller(ibis_method_name)
    t = alltypes.filter(alltypes.double_col < 50).order_by("id")
    df = t.execute()

    w = xo.window(rows=(0, None), order_by=t.id)
    expr = ibis_method(t.double_col).over(w).name("double_col")
    result = expr.execute()
    expected = pandas_fn(df.double_col[::-1])[::-1]
    assert_series_equal(result, expected)


def test_simple_ungrouped_window_with_scalar_order_by(alltypes):
    t = alltypes.filter(alltypes.double_col < 50).order_by("id")
    w = xo.window(rows=(0, None), order_by=ibis.null())
    expr = t.double_col.sum().over(w).name("double_col")
    # hard to reproduce this in pandas, so just test that it actually executes
    expr.execute()


@pytest.mark.parametrize(
    ("result_fn", "expected_fn", "ordered"),
    [
        # Reduction ops
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda df: pd.Series([df.double_col.mean()] * len(df.double_col)),
            True,
            id="ordered-mean",
        ),
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda df: pd.Series([df.double_col.mean()] * len(df.double_col)),
            False,
            id="unordered-mean",
        ),
        param(
            lambda _, win: ibis.ntile(7).over(win),
            lambda df: pandas_ntile(df.id, 7),
            True,
            id="unordered-ntile",
        ),
        # Analytic ops
        param(
            lambda t, win: t.float_col.lag().over(win),
            lambda df: df.float_col.shift(1),
            True,
            id="ordered-lag",
        ),
        param(
            lambda t, win: t.float_col.lag().over(win),
            lambda df: df.float_col.shift(1),
            False,
            id="unordered-lag",
        ),
        param(
            lambda t, win: t.float_col.lead().over(win),
            lambda df: df.float_col.shift(-1),
            True,
            id="ordered-lead",
        ),
        param(
            lambda t, win: t.float_col.lead().over(win),
            lambda df: df.float_col.shift(-1),
            False,
            id="unordered-lead",
        ),
    ],
)
# Some backends do not support non-grouped window specs
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
def test_ungrouped_unbounded_window(
    alltypes, alltypes_df, result_fn, expected_fn, ordered
):
    # Define a window that is
    # 1) Ungrouped
    # 2) Ordered if `ordered` is True
    # 3) Unbounded
    order_by = [alltypes.id] if ordered else None
    window = xo.window(order_by=order_by)
    expr = alltypes.mutate(val=result_fn(alltypes, win=window))
    result = expr.execute()
    result = result.set_index("id").sort_index()

    # Apply `expected_fn` onto a DataFrame that is
    # 1) Ungrouped
    # 2) Ordered if `ordered` is True
    alltypes_df = alltypes_df.sort_values("id") if ordered else alltypes_df
    expected = alltypes_df.assign(val=expected_fn(alltypes_df))
    expected = expected.set_index("id").sort_index()

    left, right = result.val, expected.val

    assert_series_equal(left, right)


def test_grouped_bounded_range_window(alltypes, alltypes_df):
    # Explanation of the range window spec below:
    #
    # `preceding=10, following=0, order_by='id'`:
    #     The window at a particular row (call its `id` value x) will contain
    #     some other row (call its `id` value y) if x-10 <= y <= x.
    # `group_by='string_col'`:
    #     The window at a particular row will only contain other rows that
    #     have the same 'string_col' value.
    preceding = 10
    window = xo.range_window(
        preceding=preceding,
        following=0,
        order_by="id",
        group_by="string_col",
    )
    expr = alltypes.mutate(val=alltypes.double_col.sum().over(window))
    result = expr.execute().set_index("id").sort_index()

    def gb_fn(alltypes_df):
        indices = np.searchsorted(
            alltypes_df.id, [alltypes_df["prec"], alltypes_df["foll"]], side="left"
        )
        double_col = alltypes_df.double_col.values
        return pd.Series(
            [double_col[start:stop].sum() for start, stop in indices.T],
            index=alltypes_df.index,
        )

    res = (
        # add 1 to get the upper bound without having to make two
        # searchsorted calls
        alltypes_df.assign(prec=lambda t: t.id - preceding, foll=lambda t: t.id + 1)
        .sort_values("id")
        .groupby("string_col")
        .apply(gb_fn)
        .droplevel(0)
    )
    expected = (
        alltypes_df.assign(
            # Mimic our range window spec using .apply()
            val=res
        )
        .set_index("id")
        .sort_index()
    )

    assert_series_equal(result.val, expected.val)


def test_percent_rank_whole_table_no_order_by(alltypes, alltypes_df):
    expr = alltypes.mutate(val=lambda t: t.id.percent_rank())

    result = expr.execute().set_index("id").sort_index()
    column = alltypes_df.id.rank(method="min").sub(1).div(len(alltypes_df) - 1)
    expected = alltypes_df.assign(val=column).set_index("id").sort_index()

    assert_series_equal(result.val, expected.val)


def test_grouped_ordered_window_coalesce(alltypes, alltypes_df):
    t = alltypes
    expr = (
        t.group_by("month")
        .order_by("id")
        .mutate(lagged_value=ibis.coalesce(t.bigint_col.lag(), 0))[
            ["id", "lagged_value"]
        ]
    )
    result = (
        expr.execute()
        .sort_values(["id"])
        .lagged_value.reset_index(drop=True)
        .astype("int64")
    )

    def agg(alltypes_df):
        alltypes_df = alltypes_df.sort_values(["id"])
        alltypes_df = alltypes_df.assign(
            bigint_col=lambda alltypes_df: alltypes_df.bigint_col.shift()
        )
        return alltypes_df

    expected = (
        alltypes_df.groupby("month", group_keys=False)
        .apply(agg)
        .sort_values(["id"])
        .reset_index(drop=True)
        .bigint_col.fillna(0.0)
        .astype("int64")
        .rename("lagged_value")
    )
    assert_series_equal(result, expected)


def test_mutate_window_filter(alltypes):
    t = alltypes
    win = xo.window(order_by=[t.id])
    expr = (
        t.mutate(next_int=t.int_col.lead().over(win))
        .filter(lambda t: t.int_col == 1)
        .select("int_col", "next_int")
        .limit(3)
    )
    res = expr.execute()
    sol = pd.DataFrame({"int_col": [1, 1, 1], "next_int": [2, 2, 2]})
    assert_frame_equal(res, sol, check_dtype=False)


def test_first_last(win_table):
    t = win_table
    w = xo.window(group_by=t.g, order_by=[t.x, t.y], preceding=1, following=0)
    expr = t.mutate(
        x_first=t.x.first().over(w),
        x_last=t.x.last().over(w),
        y_first=t.y.first().over(w),
        y_last=t.y.last().over(w),
    )
    result = expr.execute()
    expected = pd.DataFrame(
        {
            "g": ["a"] * 5,
            "x": range(5),
            "y": [3, 2, 0, 1, 1],
            "x_first": [0, 0, 1, 2, 3],
            "x_last": range(5),
            "y_first": [3, 3, 2, 0, 1],
            "y_last": [3, 2, 0, 1, 1],
        }
    )
    assert_frame_equal(result, expected)


def test_rank_followed_by_over_call_merge_frames(alltypes, alltypes_df):
    # GH #7631
    t = alltypes
    expr = t.int_col.percent_rank().over(xo.window(group_by=t.int_col.notnull()))
    result = expr.execute()

    expected = (
        alltypes_df.sort_values("int_col")
        .groupby(alltypes_df["int_col"].notnull())
        .apply(
            lambda alltypes_df: (
                alltypes_df.int_col.rank(method="min").sub(1).div(len(alltypes_df) - 1)
            )
        )
        .T.reset_index(drop=True)
        .iloc[:, 0]
        .rename(expr.get_name())
    )

    assert_series_equal(
        result.value_counts().sort_index(), expected.value_counts().sort_index()
    )


def test_windowed_order_by_sequence_is_preserved(con):
    table = ibis.memtable({"bool_col": [True, False, False, None, True]})
    window = xo.window(
        order_by=[
            ibis.asc(table["bool_col"].isnull()),
            ibis.asc(table["bool_col"]),
        ],
    )
    expr = table.select(
        rank=table["bool_col"].rank().over(window),
        bool_col=table["bool_col"],
    )
    result = con.execute(expr)
    value = result.bool_col.loc[result["rank"] == 4].item()
    assert pd.isna(value)


def test_duplicate_ordered_sum(con):
    expr = (
        ibis.memtable(
            {"id": range(4), "ranking": [1, 2, 3, 3], "rewards": [10, 20, 30, 40]}
        )
        .mutate(csum=lambda t: t.rewards.cumsum(order_by="ranking"))
        .order_by("id")
    )
    arrow_table = con.to_pyarrow(expr)

    result = arrow_table["csum"].to_pylist()

    assert len(result) == 4

    assert result[0] == 10
    assert result[1] == 30
    # why? because the order_by column is not unique, so both
    #
    # 10 -> 10 + 20 -> 10 + 20 + 30 => 10 -> 30 -> 60
    #
    # *AND*
    #
    # 10 -> 10 + 20 -> 10 + 20 + 40 => 10 -> 30 -> 70
    #
    # are valid, and it *may* depend on how the computation is distributed in
    # the query engine
    #
    # this also means the final cumulative sum can be in the penultimate or
    # final position, since the *output* order doesn't depend on ORDER BY
    # provided by the user
    assert result[2:] in ([60, 100], [70, 100], [100, 60], [100, 70])
