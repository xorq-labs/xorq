from __future__ import annotations

import letsql
import ibis.expr.datatypes as dt
import pytest

from letsql.tests.util import default_series_rename, assert_series_equal


@pytest.mark.parametrize(
    ("column", "raw_value"),
    [
        ("double_col", 0.0),
        ("double_col", 10.1),
        ("float_col", 1.1),
        ("float_col", 2.2),
    ],
)
def test_floating_scalar_parameter(alltypes, df, column, raw_value):
    value = letsql.param(dt.double)
    expr = (alltypes[column] + value).name("tmp")
    expected = df[column] + raw_value
    result = expr.execute(params={value: raw_value})
    expected = default_series_rename(expected)
    assert_series_equal(result, expected, check_dtype=False)
