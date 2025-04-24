from __future__ import annotations

import decimal
import math
import operator
from operator import and_, lshift, or_, rshift, xor
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from pytest import param

import xorq as xo
from xorq.tests.util import assert_series_equal, default_series_rename
from xorq.vendor import ibis
from xorq.vendor.ibis import _
from xorq.vendor.ibis import literal as L
from xorq.vendor.ibis.expr import datatypes as dt


@pytest.mark.parametrize(
    ("expr",),
    [
        param(
            xo.literal(1, type=dt.int8),
            id="int8",
        ),
        param(
            xo.literal(1, type=dt.int16),
            id="int16",
        ),
        param(
            xo.literal(1, type=dt.int32),
            id="int32",
        ),
        param(
            xo.literal(1, type=dt.int64),
            id="int64",
        ),
        param(
            xo.literal(1, type=dt.uint8),
            id="uint8",
        ),
        param(
            xo.literal(1, type=dt.uint16),
            id="uint16",
        ),
        param(
            xo.literal(1, type=dt.uint32),
            id="uint32",
        ),
        param(
            xo.literal(1, type=dt.uint64),
            id="uint64",
        ),
        param(
            xo.literal(1, type=dt.float32),
            id="float32",
        ),
        param(
            xo.literal(1, type=dt.float64),
            id="float64",
        ),
    ],
)
def test_numeric_literal(con, expr):
    result = con.execute(expr)
    assert result == 1


@pytest.mark.parametrize(
    ("expr", "expected_result"),
    [
        param(
            xo.literal(decimal.Decimal("1.1"), type=dt.decimal),
            decimal.Decimal("1.1"),
            id="default",
        ),
        param(
            xo.literal(decimal.Decimal("1.1"), type=dt.Decimal(38, 9)),
            decimal.Decimal("1.1"),
            id="decimal-small",
        ),
    ],
)
def test_decimal_literal(con, expr, expected_result):
    result = con.execute(expr)
    if type(expected_result) in (float, decimal.Decimal) and math.isnan(
        expected_result
    ):
        assert math.isnan(result) and type(result) is type(expected_result)
    else:
        assert result == expected_result


@pytest.mark.parametrize(
    ("operand_fn", "expected_operand_fn"),
    [
        param(
            lambda t: t.float_col,
            lambda t: t.float_col,
            id="float-column",
        ),
        param(
            lambda t: t.double_col,
            lambda t: t.double_col,
            id="double-column",
        ),
        param(
            lambda t: xo.literal(1.3),
            lambda t: 1.3,
            id="float-literal",
        ),
        param(
            lambda t: xo.literal(np.nan),
            lambda t: np.nan,
            id="nan-literal",
        ),
        param(
            lambda t: xo.literal(np.inf),
            lambda t: np.inf,
            id="inf-literal",
        ),
        param(
            lambda t: xo.literal(-np.inf),
            lambda t: -np.inf,
            id="-inf-literal",
        ),
    ],
)
@pytest.mark.parametrize(
    ("expr_fn", "expected_expr_fn"),
    [
        param(
            operator.methodcaller("isnan"),
            np.isnan,
            id="isnan",
        ),
    ],
)
def test_isnan_isinf(
    con,
    alltypes,
    df,
    operand_fn,
    expected_operand_fn,
    expr_fn,
    expected_expr_fn,
):
    expr = expr_fn(operand_fn(alltypes)).name("tmp")
    expected = expected_expr_fn(expected_operand_fn(df))

    result = con.execute(expr)

    if isinstance(expected, pd.Series):
        assert_series_equal(result, expected)
    else:
        try:
            assert result == expected
        except ValueError:
            assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(L(-5).abs(), 5, id="abs-neg"),
        param(L(5).abs(), 5, id="abs"),
        param(
            L(5.5).round(),
            6.0,
            id="round",
        ),
        param(
            L(5.556).round(2),
            5.56,
            id="round-digits",
        ),
        param(L(5.556).ceil(), 6.0, id="ceil"),
        param(L(5.556).floor(), 5.0, id="floor"),
        param(
            L(5.556).exp(),
            math.exp(5.556),
            id="exp",
        ),
        param(
            L(5.556).sign(),
            1,
            id="sign-pos",
        ),
        param(
            L(-5.556).sign(),
            -1,
            id="sign-neg",
        ),
        param(
            L(0).sign(),
            0,
            id="sign-zero",
        ),
        param(L(5.556).sqrt(), math.sqrt(5.556), id="sqrt"),
        param(
            L(5.556).log(2),
            math.log(5.556, 2),
            id="log-base",
        ),
        param(
            L(5.556).ln(),
            math.log(5.556),
            id="ln",
        ),
        param(
            L(5.556).log2(),
            math.log(5.556, 2),
            id="log2",
        ),
        param(
            L(5.556).log10(),
            math.log10(5.556),
            id="log10",
        ),
        param(
            L(5.556).radians(),
            math.radians(5.556),
            id="radians",
        ),
        param(
            L(5.556).degrees(),
            math.degrees(5.556),
            id="degrees",
        ),
        param(
            L(11) % 3,
            11 % 3,
            id="mod",
        ),
        param(
            xo.greatest(L(10), L(1)),
            10,
            id="greatest",
        ),
        param(
            xo.least(L(10), L(1)),
            1,
            id="least",
        ),
    ],
)
def test_math_functions_literals(con, expr, expected):
    result = con.execute(expr.name("tmp"))
    if isinstance(result, decimal.Decimal):
        assert result == decimal.Decimal(str(expected))
    else:
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(L(0.0).acos(), math.acos(0.0), id="acos"),
        param(L(0.0).asin(), math.asin(0.0), id="asin"),
        param(L(0.0).atan(), math.atan(0.0), id="atan"),
        param(L(0.0).atan2(1.0), math.atan2(0.0, 1.0), id="atan2"),
        param(L(0.0).cos(), math.cos(0.0), id="cos"),
        param(L(1.0).cot(), 1.0 / math.tan(1.0), id="cot"),
        param(L(0.0).sin(), math.sin(0.0), id="sin"),
        param(L(0.0).tan(), math.tan(0.0), id="tan"),
    ],
)
def test_trig_functions_literals(con, expr, expected):
    result = con.execute(expr.name("tmp"))
    assert pytest.approx(result) == expected


@pytest.mark.parametrize(
    ("expr", "expected_fn"),
    [
        param(_.dc.acos(), np.arccos, id="acos"),
        param(_.dc.asin(), np.arcsin, id="asin"),
        param(_.dc.atan(), np.arctan, id="atan"),
        param(_.dc.atan2(_.dc), lambda c: np.arctan2(c, c), id="atan2"),
        param(_.dc.cos(), np.cos, id="cos"),
        param(_.dc.cot(), lambda c: 1.0 / np.tan(c), id="cot"),
        param(_.dc.sin(), np.sin, id="sin"),
        param(_.dc.tan(), np.tan, id="tan"),
    ],
)
def test_trig_functions_columns(expr, alltypes, df, expected_fn):
    dc_max = df.double_col.max()
    expr = alltypes.mutate(dc=(_.double_col / dc_max).nullif(0)).select(tmp=expr)
    result = expr.tmp.to_pandas()
    expected = expected_fn((df.double_col / dc_max).replace(0.0, np.nan)).rename("tmp")
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        param(
            lambda t: (-t.double_col).abs(),
            lambda t: (-t.double_col).abs(),
            id="abs-neg",
        ),
        param(
            lambda t: t.double_col.abs(),
            lambda t: t.double_col.abs(),
            id="abs",
        ),
        param(
            lambda t: t.double_col.ceil(),
            lambda t: np.ceil(t.double_col).astype("int64"),
            id="ceil",
        ),
        param(
            lambda t: t.double_col.floor(),
            lambda t: np.floor(t.double_col).astype("int64"),
            id="floor",
        ),
        param(
            lambda t: t.double_col.sign(),
            lambda t: np.sign(t.double_col),
            id="sign",
        ),
        param(
            lambda t: (-t.double_col).sign(),
            lambda t: np.sign(-t.double_col),
            id="sign-negative",
        ),
    ],
)
def test_simple_math_functions_columns(con, alltypes, df, expr_fn, expected_fn):
    expr = expr_fn(alltypes).name("tmp")
    expected = expected_fn(df)
    result = con.execute(expr)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        param(
            lambda t: t.double_col.add(1).sqrt(),
            lambda t: np.sqrt(t.double_col + 1),
            id="sqrt",
        ),
        param(
            lambda t: t.double_col.add(1).exp(),
            lambda t: np.exp(t.double_col + 1),
            id="exp",
        ),
        param(
            lambda t: t.double_col.add(1).log(2),
            lambda t: np.log2(t.double_col + 1),
            id="log2",
        ),
        param(
            lambda t: t.double_col.add(1).ln(),
            lambda t: np.log(t.double_col + 1),
            id="ln",
        ),
        param(
            lambda t: t.double_col.add(1).log10(),
            lambda t: np.log10(t.double_col + 1),
            id="log10",
        ),
    ],
)
def test_complex_math_functions_columns(con, alltypes, df, expr_fn, expected_fn):
    expr = expr_fn(alltypes).name("tmp")
    expected = expected_fn(df)
    result = con.execute(expr)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        param(
            lambda t: xo.least(t.bigint_col, t.int_col),
            lambda t: pd.Series(list(map(min, t.bigint_col, t.int_col))),
            id="least-all-columns",
        ),
        param(
            lambda t: xo.least(t.bigint_col, t.int_col, -2),
            lambda t: pd.Series(list(map(min, t.bigint_col, t.int_col, [-2] * len(t)))),
            id="least-scalar",
        ),
        param(
            lambda t: xo.greatest(t.bigint_col, t.int_col),
            lambda t: pd.Series(list(map(max, t.bigint_col, t.int_col))),
            id="greatest-all-columns",
        ),
        param(
            lambda t: xo.greatest(t.bigint_col, t.int_col, -2),
            lambda t: pd.Series(list(map(max, t.bigint_col, t.int_col, [-2] * len(t)))),
            id="greatest-scalar",
        ),
    ],
)
def test_backend_specific_numerics(
    con, df, alltypes, expr_fn: Callable, expected_fn: Callable
):
    expr = expr_fn(alltypes)
    result = default_series_rename(con.execute(expr.name("tmp")))
    expected = default_series_rename(expected_fn(df))
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.pow,
    ],
    ids=lambda op: op.__name__,
)
def test_binary_arithmetic_operations(alltypes, df, op):
    smallint_col = alltypes.smallint_col + 1  # make it nonzero
    smallint_series = df.smallint_col + 1

    expr = op(alltypes.double_col, smallint_col).name("tmp")

    result = expr.execute()
    expected = op(df.double_col, smallint_series)
    if op is operator.floordiv:
        # defined in ops.FloorDivide.output_type
        # -> returns int64 whereas pandas float64
        result = result.astype("float64")

    expected = expected.astype("float64")
    assert_series_equal(result, expected, check_exact=False)


def test_mod(alltypes, df):
    expr = operator.mod(alltypes.smallint_col, alltypes.smallint_col + 1).name("tmp")

    result = expr.execute()
    expected = operator.mod(df.smallint_col, df.smallint_col + 1)
    assert_series_equal(result, expected, check_dtype=False)


def test_floating_mod(alltypes, df):
    expr = operator.mod(alltypes.double_col, alltypes.smallint_col + 1).name("tmp")

    result = expr.execute()
    expected = operator.mod(df.double_col, df.smallint_col + 1)
    assert_series_equal(result, expected, check_exact=False)


@pytest.mark.parametrize(
    ("column", "denominator"),
    [
        param(
            "tinyint_col",
            0,
        ),
        param(
            "smallint_col",
            0,
        ),
        param(
            "int_col",
            0,
        ),
        param(
            "bigint_col",
            0,
        ),
        param(
            "float_col",
            0,
        ),
        param(
            "double_col",
            0,
        ),
        param(
            "tinyint_col",
            0.0,
        ),
        param(
            "smallint_col",
            0.0,
        ),
        param(
            "int_col",
            0.0,
        ),
        param(
            "bigint_col",
            0.0,
        ),
        param(
            "float_col",
            0.0,
        ),
        param(
            "double_col",
            0.0,
        ),
    ],
)
def test_divide_by_zero(alltypes, df, column, denominator):
    expr = alltypes[column] / denominator
    result = expr.name("tmp").execute()

    expected = df[column].div(denominator)
    expected = expected.astype("float64")

    assert_series_equal(result.astype("float64"), expected)


def test_random(con):
    expr = xo.random()
    result = con.execute(expr)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_histogram(con, alltypes):
    n = 10
    hist = con.execute(alltypes.int_col.histogram(n).name("hist"))
    vc = hist.value_counts().sort_index()
    vc_np, _bin_edges = np.histogram(alltypes.int_col.execute(), bins=n)
    assert vc.tolist() == vc_np.tolist()


@pytest.mark.parametrize("const", ["pi", "e"])
def test_constants(con, const):
    expr = getattr(ibis, const)
    result = con.execute(expr)
    assert pytest.approx(result) == getattr(math, const)


@pytest.mark.parametrize("op", [and_, or_, xor])
@pytest.mark.parametrize(
    ("left_fn", "right_fn"),
    [
        param(lambda t: t.int_col, lambda t: t.int_col, id="col_col"),
        param(lambda _: 3, lambda t: t.int_col, id="scalar_col"),
        param(lambda t: t.int_col, lambda _: 3, id="col_scalar"),
    ],
)
def test_bitwise_columns(con, alltypes, df, op, left_fn, right_fn):
    expr = op(left_fn(alltypes), right_fn(alltypes)).name("tmp")
    result = con.execute(expr)

    expected = op(left_fn(df), right_fn(df)).rename("tmp")
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("op", "left_fn", "right_fn"),
    [
        param(
            lshift,
            lambda t: t.int_col,
            lambda t: t.int_col,
            id="lshift_col_col",
        ),
        param(
            lshift,
            lambda _: 3,
            lambda t: t.int_col,
            id="lshift_scalar_col",
        ),
        param(lshift, lambda t: t.int_col, lambda _: 3, id="lshift_col_scalar"),
        param(rshift, lambda t: t.int_col, lambda t: t.int_col, id="rshift_col_col"),
        param(rshift, lambda _: 3, lambda t: t.int_col, id="rshift_scalar_col"),
        param(rshift, lambda t: t.int_col, lambda _: 3, id="rshift_col_scalar"),
    ],
)
def test_bitwise_shift(alltypes, df, op, left_fn, right_fn):
    expr = op(left_fn(alltypes), right_fn(alltypes)).name("tmp")
    result = expr.execute()

    pandas_left = getattr(left := left_fn(df), "values", left)
    pandas_right = getattr(right := right_fn(df), "values", right)
    expected = pd.Series(
        op(pandas_left, pandas_right),
        name="tmp",
        dtype="int64",
    )
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "op",
    [
        and_,
        or_,
        xor,
        lshift,
        rshift,
    ],
)
@pytest.mark.parametrize(
    ("left", "right"),
    [param(4, L(2), id="int_col"), param(L(4), 2, id="col_int")],
)
def test_bitwise_scalars(con, op, left, right):
    expr = op(left, right)
    result = con.execute(expr)
    expected = op(4, 2)
    assert result == expected


def test_bitwise_not_scalar(con):
    expr = ~L(2)
    result = con.execute(expr)
    expected = -3
    assert result == expected


def test_bitwise_not_col(alltypes, df):
    expr = (~alltypes.int_col).name("tmp")
    result = expr.execute()
    expected = ~df.int_col
    assert_series_equal(result, expected.rename("tmp"))


@pytest.mark.parametrize(
    ("ibis_func", "pandas_func"),
    [
        param(lambda x: x.clip(lower=0), lambda x: x.clip(lower=0), id="lower-int"),
        param(
            lambda x: x.clip(lower=0.0), lambda x: x.clip(lower=0.0), id="lower-float"
        ),
        param(lambda x: x.clip(upper=0), lambda x: x.clip(upper=0), id="upper-int"),
        param(
            lambda x: x.clip(lower=x - 1, upper=x + 1),
            lambda x: x.clip(lower=x - 1, upper=x + 1),
            id="lower-upper-expr",
        ),
        param(
            lambda x: x.clip(lower=0, upper=1),
            lambda x: x.clip(lower=0, upper=1),
            id="lower-upper-int",
        ),
        param(
            lambda x: x.clip(lower=0, upper=1.0),
            lambda x: x.clip(lower=0, upper=1.0),
            id="lower-upper-float",
        ),
        param(
            lambda x: x.nullif(1).clip(lower=0),
            lambda x: x.where(x != 1).clip(lower=0),
            id="null-lower",
        ),
        param(
            lambda x: x.nullif(1).clip(upper=0),
            lambda x: x.where(x != 1).clip(upper=0),
            id="null-upper",
        ),
    ],
)
def test_clip(alltypes, df, ibis_func, pandas_func):
    result = ibis_func(alltypes.int_col).execute()
    expected = pandas_func(df.int_col).astype(result.dtype)
    assert_series_equal(result, expected, check_names=False)
