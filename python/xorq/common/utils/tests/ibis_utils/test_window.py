from __future__ import annotations

import pytest
from pytest import param

from xorq.common.utils.ibis_utils import from_ibis
from xorq.tests.util import assert_frame_equal, assert_series_equal


ibis = pytest.importorskip("ibis")


@pytest.mark.parametrize(
    "window_fn",
    [
        param(
            lambda t, win: t.float_col.lag().over(win),
            id="lag",
        ),
        param(
            lambda t, win: t.float_col.lag(2).over(win),
            id="lag_offset",
        ),
        param(
            lambda t, win: t.float_col.lead().over(win),
            id="lead",
        ),
        param(
            lambda t, win: t.float_col.lead(3).over(win),
            id="lead_offset",
        ),
        param(
            lambda t, win: t.id.rank().over(win),
            id="rank",
        ),
        param(
            lambda t, win: t.id.dense_rank().over(win),
            id="dense_rank",
        ),
        param(
            lambda t, win: t.id.percent_rank().over(win),
            id="percent_rank",
        ),
        param(
            lambda t, win: t.id.cume_dist().over(win),
            id="cume_dist",
        ),
        param(
            lambda t, win: t.float_col.first().over(win),
            id="first",
        ),
        param(
            lambda t, win: t.float_col.last().over(win),
            id="last",
        ),
        param(
            lambda t, win: t.double_col.nth(3).over(win),
            id="nth",
        ),
        param(
            lambda _, win: ibis.row_number().over(win),
            id="row_number",
        ),
        param(
            lambda t, win: t.double_col.cumsum().over(win),
            id="cumsum",
        ),
        param(
            lambda t, win: t.double_col.cummean().over(win),
            id="cummean",
        ),
        param(
            lambda t, win: t.float_col.cummin().over(win),
            id="cummin",
        ),
        param(
            lambda t, win: t.float_col.cummax().over(win),
            id="cummax",
        ),
        param(
            lambda t, win: (t.double_col == 0).any().over(win),
            id="cumany",
        ),
        param(
            lambda t, win: (t.double_col == 0).notany().over(win),
            id="cumnotany",
        ),
        param(
            lambda t, win: (t.double_col == 0).all().over(win),
            id="cumall",
        ),
        param(
            lambda t, win: (t.double_col == 0).notall().over(win),
            id="cumnotall",
        ),
        param(
            lambda t, win: t.double_col.sum().over(win),
            id="sum",
        ),
        param(
            lambda t, win: t.double_col.mean().over(win),
            id="mean",
        ),
        param(
            lambda t, win: t.float_col.min().over(win),
            id="min",
        ),
        param(
            lambda t, win: t.float_col.max().over(win),
            id="max",
        ),
        param(
            lambda t, win: t.double_col.count().over(win),
            id="count",
        ),
    ],
)
def test_window_functions_grouped_ordered(ibis_alltypes, window_fn):
    win = ibis.window(
        group_by=[ibis_alltypes.string_col],
        order_by=[ibis_alltypes.id],
    )
    expr = ibis_alltypes.mutate(val=window_fn(ibis_alltypes, win))

    xorq_expr = from_ibis(expr)
    actual = xorq_expr.execute()
    expected = expr.execute()

    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


@pytest.mark.parametrize(
    "window_fn",
    [
        param(
            lambda t, win: ibis.row_number().over(win),
            id="row_number",
        ),
        param(
            lambda t, win: t.double_col.sum().over(win),
            id="sum",
        ),
        param(
            lambda t, win: t.double_col.mean().over(win),
            id="mean",
        ),
        param(
            lambda t, win: t.float_col.lag().over(win),
            id="lag",
        ),
        param(
            lambda t, win: t.float_col.lead().over(win),
            id="lead",
        ),
    ],
)
@pytest.mark.parametrize(
    "ordered",
    [
        param(True, id="ordered"),
        param(False, id="unordered"),
    ],
)
def test_window_functions_ungrouped(ibis_alltypes, window_fn, ordered):
    order_by = [ibis_alltypes.id] if ordered else None
    win = ibis.window(order_by=order_by)
    expr = ibis_alltypes.mutate(val=window_fn(ibis_alltypes, win))

    xorq_expr = from_ibis(expr)

    actual = xorq_expr.execute()
    expected = expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


@pytest.mark.parametrize(
    ("preceding", "following"),
    [
        param(2, 0, id="preceding_2_following_0"),
        param((2, 0), None, id="preceding_tuple"),
        param(0, 2, id="preceding_0_following_2"),
        param(None, (0, 2), id="following_tuple"),
        param(1, 1, id="preceding_1_following_1"),
    ],
)
def test_window_frame_rows_bounded(ibis_alltypes, preceding, following):
    win = ibis.window(
        preceding=preceding,
        following=following,
        group_by=[ibis_alltypes.string_col],
        order_by=[ibis_alltypes.id],
    )
    expr = ibis_alltypes.mutate(val=ibis_alltypes.double_col.sum().over(win))

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


def test_window_frame_trailing(ibis_alltypes):
    win = ibis.trailing_window(
        preceding=2,
        group_by=[ibis_alltypes.string_col],
        order_by=[ibis_alltypes.id],
    )
    expr = ibis_alltypes.mutate(val=ibis_alltypes.double_col.sum().over(win))

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


def test_window_frame_unbounded_following(ibis_alltypes):
    win = ibis.window(
        rows=(0, None),
        order_by=[ibis_alltypes.id],
    )
    expr = ibis_alltypes.mutate(val=ibis_alltypes.double_col.sum().over(win))

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


def test_window_frame_unbounded_preceding(ibis_alltypes):
    win = ibis.window(
        following=0,
        group_by=[ibis_alltypes.string_col],
        order_by=[ibis_alltypes.id],
    )
    expr = ibis_alltypes.mutate(val=ibis_alltypes.double_col.sum().over(win))

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


def test_window_range_specification(ibis_alltypes):
    win = ibis.range_window(
        preceding=10,
        following=0,
        order_by="id",
        group_by="string_col",
    )
    expr = ibis_alltypes.mutate(val=ibis_alltypes.double_col.sum().over(win))

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


@pytest.mark.parametrize(
    "group_by_cols",
    [
        param([ibis._.string_col], id="single_column"),
        param([ibis._.string_col, ibis._.bool_col], id="multiple_columns"),
    ],
)
def test_window_partition_by(ibis_alltypes, group_by_cols):
    win = ibis.window(
        group_by=group_by_cols,
        order_by=[ibis_alltypes.id],
    )
    expr = ibis_alltypes.mutate(val=ibis.row_number().over(win))

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


@pytest.mark.parametrize(
    "order_by_spec",
    [
        param([ibis._.id], id="single_asc"),
        param([ibis.desc(ibis._.id)], id="single_desc"),
        param([ibis._.string_col, ibis._.id], id="multiple_asc"),
        param(
            [ibis.asc(ibis._.string_col), ibis.desc(ibis._.id)],
            id="multiple_mixed",
        ),
    ],
)
def test_window_order_by(ibis_alltypes, order_by_spec):
    win = ibis.window(
        group_by=[ibis_alltypes.string_col],
        order_by=order_by_spec,
    )
    expr = ibis_alltypes.mutate(val=ibis.row_number().over(win))

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


def test_window_ntile(ibis_alltypes):
    win = ibis.window(
        group_by=[ibis_alltypes.string_col],
        order_by=[ibis_alltypes.id],
    )

    expr = ibis_alltypes.mutate(val=ibis.ntile(7).over(win))

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
    result = xorq_expr.execute()
    assert not result.empty


def test_window_multiple_functions(ibis_alltypes):
    win = ibis.window(
        group_by=[ibis_alltypes.string_col],
        order_by=[ibis_alltypes.id],
    )
    expr = ibis_alltypes.mutate(
        row_num=ibis.row_number().over(win),
        running_sum=ibis_alltypes.double_col.sum().over(win),
        running_avg=ibis_alltypes.double_col.mean().over(win),
        prev_val=ibis_alltypes.float_col.lag().over(win),
    )

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


def test_window_with_filter(ibis_alltypes):
    win = ibis.window(order_by=[ibis_alltypes.id])
    expr = (
        ibis_alltypes.mutate(next_int=ibis_alltypes.int_col.lead().over(win))
        .filter(lambda t: t.int_col == 1)
        .select("int_col", "next_int")
    )

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(expected, actual, check_dtype=False)


def test_window_coalesce(ibis_alltypes):
    expr = (
        ibis_alltypes.group_by("month")
        .order_by("id")
        .mutate(lagged_value=ibis.coalesce(ibis_alltypes.bigint_col.lag(), 0))
        .select(["id", "lagged_value"])
    )

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


def test_window_percent_rank_no_order(ibis_alltypes):
    expr = ibis_alltypes.mutate(val=ibis_alltypes.id.percent_rank())

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


def test_window_rank_over_call(ibis_alltypes):
    expr = ibis_alltypes.int_col.percent_rank().over(
        ibis.window(group_by=ibis_alltypes.int_col.notnull())
    )

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_series_equal(expected.sort_values(), actual.sort_values(), check_dtype=False)


def test_window_order_by_with_nulls(ibis_alltypes):
    win = ibis.window(
        order_by=[
            ibis.asc(ibis_alltypes.bool_col.isnull()),
            ibis.asc(ibis_alltypes.bool_col),
        ],
    )
    expr = ibis_alltypes.select(
        rank=ibis_alltypes.bool_col.rank().over(win),
        bool_col=ibis_alltypes.bool_col,
    )

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(expected, actual, check_dtype=False)


def test_window_cumsum_shorthand(ibis_alltypes):
    expr = ibis_alltypes.mutate(csum=ibis_alltypes.double_col.cumsum(order_by="id"))

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )


@pytest.mark.parametrize(
    ("window_fn", "aggr_fn"),
    [
        param(
            lambda t: ibis.window(group_by=[t.string_col], order_by=[t.id]),
            lambda t: t.double_col.sum(),
            id="sum",
        ),
        param(
            lambda t: ibis.window(group_by=[t.string_col], order_by=[t.id]),
            lambda t: t.double_col.mean(),
            id="mean",
        ),
        param(
            lambda t: ibis.window(group_by=[t.string_col], order_by=[t.id]),
            lambda t: t.double_col.min(),
            id="min",
        ),
        param(
            lambda t: ibis.window(group_by=[t.string_col], order_by=[t.id]),
            lambda t: t.double_col.max(),
            id="max",
        ),
        param(
            lambda t: ibis.window(group_by=[t.string_col], order_by=[t.id]),
            lambda t: t.double_col.count(),
            id="count",
        ),
        param(
            lambda t: ibis.window(group_by=[t.string_col], order_by=[t.id]),
            lambda t: t.double_col.std(),
            id="std",
        ),
        param(
            lambda t: ibis.window(group_by=[t.string_col], order_by=[t.id]),
            lambda t: t.double_col.var(),
            id="var",
        ),
    ],
)
def test_window_aggregate_functions(ibis_alltypes, window_fn, aggr_fn):
    win = window_fn(ibis_alltypes)
    expr = ibis_alltypes.mutate(val=aggr_fn(ibis_alltypes).over(win))

    xorq_expr = from_ibis(expr)

    expected = expr.execute()
    actual = xorq_expr.execute()
    assert_frame_equal(
        expected.sort_values("id"), actual.sort_values("id"), check_dtype=False
    )
