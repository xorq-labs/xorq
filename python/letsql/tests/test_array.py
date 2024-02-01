from __future__ import annotations

import ibis
import ibis.expr.types as ir
import numpy as np
import pandas as pd
import pytest

from letsql.tests.util import assert_series_equal


def test_array_column(alltypes, df):
    expr = ibis.array([alltypes["double_col"], alltypes["double_col"]])
    assert isinstance(expr, ir.ArrayColumn)

    result = expr.execute()
    expected = df.apply(
        lambda row: [row["double_col"], row["double_col"]],
        axis=1,
    )
    assert_series_equal(result, expected, check_names=False)


def test_array_scalar(con):
    expr = ibis.array([1.0, 2.0, 3.0])
    assert isinstance(expr, ir.ArrayScalar)

    result = con.execute(expr.name("tmp"))
    expected = np.array([1.0, 2.0, 3.0])

    assert np.array_equal(result, expected)


def test_array_repeat(con):
    expr = ibis.array([1.0, 2.0]) * 2

    result = con.execute(expr.name("tmp"))
    expected = np.array([1.0, 2.0, 1.0, 2.0])

    assert np.array_equal(result, expected)


def test_array_concat(con):
    left = ibis.literal([1, 2, 3])
    right = ibis.literal([2, 1])
    expr = left + right
    result = con.execute(expr.name("tmp"))
    expected = np.array([1, 2, 3, 2, 1])
    assert np.array_equal(result, expected)


def test_array_concat_variadic(con):
    left = ibis.literal([1, 2, 3])
    right = ibis.literal([2, 1])
    expr = left.concat(right, right, right)
    result = con.execute(expr.name("tmp"))
    expected = np.array([1, 2, 3, 2, 1, 2, 1, 2, 1])
    assert np.array_equal(result, expected)


def test_array_radd_concat(con):
    left = [1]
    right = ibis.literal([2])
    expr = left + right
    result = con.execute(expr.name("tmp"))
    expected = np.array([1, 2])

    assert np.array_equal(result, expected)


def test_array_length(con):
    expr = ibis.literal([1, 2, 3]).length()
    assert con.execute(expr.name("tmp")) == 3


def test_list_literal(con):
    arr = [1, 2, 3]
    expr = ibis.literal(arr)
    result = con.execute(expr.name("tmp"))

    assert np.array_equal(result, arr)


def test_np_array_literal(con):
    arr = np.array([1, 2, 3])
    expr = ibis.literal(arr)
    result = con.execute(expr.name("tmp"))

    assert np.array_equal(result, arr)


def test_array_contains(con, array_types):
    t = array_types
    expr = t.x.contains(1)
    result = con.execute(expr)
    expected = t.x.execute().map(lambda lst: 1 in lst)
    assert_series_equal(result, expected, check_names=False)


@pytest.mark.skip(reason="failing in datafusion 34+ version")
def test_array_position(con):
    t = ibis.memtable({"a": [[1], [], [42, 42], []]})
    expr = t.a.index(42)
    result = con.execute(expr)
    expected = pd.Series([-1, -1, 0, -1], dtype="object")
    assert_series_equal(result, expected, check_names=False, check_dtype=False)


def test_array_remove(con):
    t = ibis.memtable({"a": [[3, 2], [], [42, 2], [2, 2], []]})
    expr = t.a.remove(2)
    result = con.execute(expr)
    expected = pd.Series([[3], [], [42], [], []], dtype="object")
    assert_series_equal(result, expected, check_names=False)
