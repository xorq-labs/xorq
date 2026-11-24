from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.types as ir
from xorq import udf
from xorq.caching import ParquetCache


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


def _cached_after(first: ir.Table, second: ir.Table, tmp_path) -> list:
    cache = ParquetCache.from_kwargs(source=first._find_backend(), base_path=tmp_path)
    first.cache(cache).execute()
    return second.cache(cache).execute().iloc[:, -1].tolist()


def test_post_process_fn_reaches_cache_key(tmp_path) -> None:
    con = xo.connect()
    t = con.register(pd.DataFrame({"x": [0.0, 1.0, 2.0]}), "t")
    schema = xo.schema({"x": "float64"})
    model = con.register(pd.DataFrame({"m": [b"x"]}), "m").m.first().as_table()

    def fn(computed, df):
        return pd.Series([float(computed)] * len(df))

    first, second = (
        t.mutate(
            p=udf.make_pandas_expr_udf(
                model, fn, schema, dt.float64, post_process_fn=pp, name="pp"
            ).on_expr(t)
        )
        for pp in (lambda b: 1.0, lambda b: 2.0)
    )
    assert first.ls.tokenized != second.ls.tokenized
    assert _cached_after(first, second, tmp_path) == [2.0, 2.0, 2.0]


def test_pyarrow_udwf_fn_reaches_cache_key(tmp_path) -> None:
    def ones(self, values, num_rows):
        return pa.array(np.ones(num_rows))

    def zeros(self, values, num_rows):
        return pa.array(np.zeros(num_rows))

    t = xo.connect().register(pd.DataFrame({"x": [0.0, 1.0, 2.0]}), "t")
    schema = xo.schema({"x": "float64"})
    first, second = (
        t.mutate(w=udf.pyarrow_udwf(fn, schema, dt.float64, name="w").on_expr(t))
        for fn in (ones, zeros)
    )
    assert first.ls.tokenized != second.ls.tokenized
    assert _cached_after(first, second, tmp_path) == [0.0, 0.0, 0.0]
