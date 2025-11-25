import pytest

import xorq.api as xo
from xorq.common.utils.ibis_utils import from_ibis


ibis = pytest.importorskip("ibis")


def test_window_function_conversion(t):
    """Test basic window function conversion from ibis to xorq."""
    expr = t.select(
        [
            t.c.mean()
            .over(ibis.window(preceding=5, following=0, group_by=t.a))
            .name("mean_c")
        ]
    )

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


@pytest.mark.parametrize(
    "preceding,following",
    [
        (None, None),
        (5, 0),
        (0, 5),
        (5, 5),
    ],
)
def test_window_preceding_following(ibis_alltypes, preceding, following):
    """Test window functions with different preceding/following values."""
    expr = ibis_alltypes.select(
        [
            ibis_alltypes.int_col.mean()
            .over(
                ibis.window(
                    preceding=preceding,
                    following=following,
                    group_by=ibis_alltypes.tinyint_col,
                )
            )
            .name("mean_int")
        ]
    )

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
    assert not xo.execute(xorq_expr).empty


def test_row_number_simple(ibis_alltypes):
    """Test simple row_number without window specification."""
    expr = ibis_alltypes.select([ibis.row_number().name("row_num")])

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
    assert not xo.execute(xorq_expr).empty


def test_row_number_with_window(t):
    """Test row_number with complex window specification."""
    expr = t.select(
        [
            ibis.row_number()
            .over(
                ibis.window(
                    group_by=[t.a, t.b],
                    order_by=[t.c.desc(), t.d],
                    preceding=5,
                    following=0,
                )
            )
            .name("row_num")
        ]
    )

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None


def test_window_with_order_by(ibis_alltypes):
    """Test window function with order_by clause."""
    expr = ibis_alltypes.select(
        [
            ibis.row_number()
            .over(
                ibis.window(
                    group_by=ibis_alltypes.tinyint_col,
                    order_by=ibis_alltypes.int_col.desc(),
                )
            )
            .name("ordered_row_num")
        ]
    )

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
    assert not xo.execute(xorq_expr).empty


def test_multiple_window_expressions(ibis_alltypes):
    """Test multiple window expressions in a single select."""
    expr = ibis_alltypes.select(
        [
            ibis.row_number()
            .over(ibis.window(group_by=ibis_alltypes.tinyint_col))
            .name("simple_row_num"),
            ibis.row_number()
            .over(
                ibis.window(
                    group_by=[ibis_alltypes.tinyint_col, ibis_alltypes.bool_col],
                    order_by=ibis_alltypes.int_col.desc(),
                )
            )
            .name("ordered_row_num"),
            ibis_alltypes.double_col.mean()
            .over(
                ibis.window(
                    preceding=3,
                    following=0,
                    group_by=ibis_alltypes.tinyint_col,
                )
            )
            .name("mean_double"),
        ]
    )

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
    assert not xo.execute(xorq_expr).empty


@pytest.mark.parametrize(
    "agg_fn",
    [
        lambda col: col.mean(),
        lambda col: col.sum(),
        lambda col: col.min(),
        lambda col: col.max(),
    ],
)
def test_window_aggregation_functions(ibis_alltypes, agg_fn):
    """Test different aggregation functions with windows."""
    expr = ibis_alltypes.select(
        [
            agg_fn(ibis_alltypes.double_col)
            .over(
                ibis.window(
                    preceding=5,
                    following=0,
                    group_by=ibis_alltypes.tinyint_col,
                )
            )
            .name("agg_result")
        ]
    )

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
    assert not xo.execute(xorq_expr).empty


def test_window_with_multiple_group_by(ibis_alltypes):
    """Test window function with multiple group_by columns."""
    expr = ibis_alltypes.select(
        [
            ibis_alltypes.int_col.mean()
            .over(
                ibis.window(
                    group_by=[ibis_alltypes.tinyint_col, ibis_alltypes.bool_col],
                    order_by=ibis_alltypes.double_col,
                )
            )
            .name("grouped_mean")
        ]
    )

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
    assert not xo.execute(xorq_expr).empty


def test_window_with_multiple_order_by(ibis_alltypes):
    """Test window function with multiple order_by columns."""
    expr = ibis_alltypes.select(
        [
            ibis.row_number()
            .over(
                ibis.window(
                    group_by=ibis_alltypes.tinyint_col,
                    order_by=[
                        ibis_alltypes.int_col.desc(),
                        ibis_alltypes.double_col,
                    ],
                )
            )
            .name("multi_order_row_num")
        ]
    )

    xorq_expr = from_ibis(expr)
    assert xorq_expr is not None
    assert not xo.execute(xorq_expr).empty
