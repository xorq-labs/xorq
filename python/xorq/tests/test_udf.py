from __future__ import annotations

import pandas.testing as tm
import pyarrow as pa
import pytest

import xorq as xo
import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.types as ir
from xorq import udf


pc = pytest.importorskip("pyarrow.compute")


@xo.udf.scalar.pyarrow
def my_string_length(arr: dt.string) -> dt.int64:
    # arr is a pyarrow.StringArray
    return pc.cast(pc.multiply(pc.utf8_length(arr), 2), target_type="int64")


@xo.udf.scalar.pyarrow
def my_add(arr1: dt.int64, arr2: dt.int64) -> dt.int64:
    return pc.add(arr1, arr2)


@xo.udf.scalar.pyarrow
def small_add(arr1: dt.int32, arr2: dt.int32) -> dt.int64:
    return pc.cast(pc.add(arr1, arr2), pa.int64())


@xo.udf.agg.builtin
def my_mean(arr: dt.float64) -> dt.float64:
    return pc.mean(arr)


def test_udf(alltypes):
    data_string_col = alltypes.date_string_col.execute()
    expected = data_string_col.str.len() * 2

    expr = my_string_length(alltypes.date_string_col)
    assert isinstance(expr, ir.Column)

    result = expr.execute()
    tm.assert_series_equal(result, expected, check_names=False)


def test_multiple_argument_udf(alltypes):
    expr = small_add(alltypes.smallint_col, alltypes.int_col).name("tmp")
    result = expr.execute()

    df = alltypes[["smallint_col", "int_col"]].execute()
    expected = (df.smallint_col + df.int_col).astype("int64")

    tm.assert_series_equal(result, expected.rename("tmp"))


def test_builtin_scalar_udf(con):
    @udf.scalar.builtin
    def to_hex(a: int) -> str:
        """Convert an integer to a hex string."""

    expr = to_hex(42)
    result = con.execute(expr)
    assert result == "2a"


def test_builtin_agg_udf(con):
    @udf.agg.builtin
    def median(a: float) -> float:
        """Median of a column."""

    expr = median(con.tables.batting.G)
    result = con.execute(expr)
    assert result == con.tables.batting.G.execute().median()


def test_builtin_agg_udf_filtered(con):
    @udf.agg.builtin
    def median(a: float, where: bool = True) -> float:
        """Median of a column."""

    median(con.tables.batting.G).execute()


@pytest.mark.parametrize(
    "method",
    [
        "make_pandas_expr_udf",
        "make_pandas_udf",
        "scalar",
        "agg",
        "pyarrow_udwf",
    ],
)
def test_top_level_udf(method):
    assert hasattr(xo, method)
    assert hasattr(udf, method)
