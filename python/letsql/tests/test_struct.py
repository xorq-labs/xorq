from __future__ import annotations

from collections.abc import Mapping

import ibis
import ibis.expr.datatypes as dt
import numpy as np
import pandas as pd
import pytest
from pytest import param

from letsql.tests.util import assert_series_equal, assert_frame_equal


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        param(
            "a",
            [1.0, 2.0, 2.0, 3.0, 3.0, np.nan],
            id="a",
        ),
        param("b", ["apple", "banana", "banana", "orange", "orange", None], id="b"),
        param(
            "c",
            [2, 2, 3, 3, 4, np.nan],
            id="c",
        ),
    ],
)
def test_single_field(struct, field, expected):
    expr = struct.select(field=lambda t: t.abc[field]).order_by("field")
    result = expr.execute()
    assert_series_equal(
        result.field, pd.Series(expected, name="field"), check_dtype=False
    )


def test_all_fields(struct, struct_df):
    result = struct.abc.execute()
    expected = struct_df.abc

    assert {
        row if not isinstance(row, Mapping) else tuple(row.items()) for row in result
    } == {
        row if not isinstance(row, Mapping) else tuple(row.items()) for row in expected
    }


_SIMPLE_DICT = dict(a=1, b="2", c=3.0)
_STRUCT_LITERAL = ibis.struct(
    _SIMPLE_DICT,
    type="struct<a: int64, b: string, c: float64>",
)
_NULL_STRUCT_LITERAL = ibis.null().cast("struct<a: int64, b: string, c: float64>")


@pytest.mark.parametrize("field", ["a", "b", "c"])
def test_literal(con, field):
    query = _STRUCT_LITERAL[field]
    dtype = query.type().to_pandas()
    result = pd.Series([con.execute(query)], dtype=dtype)
    result = result.replace({np.nan: None})
    expected = pd.Series([_SIMPLE_DICT[field]])
    assert_series_equal(result, expected.astype(dtype))


def test_struct_column(alltypes, df):
    t = alltypes
    expr = t.select(s=ibis.struct(dict(a=t.string_col, b=1, c=t.bigint_col)))
    assert expr.s.type() == dt.Struct(dict(a=dt.string, b=dt.int8, c=dt.int64))
    result = expr.execute()
    expected = pd.DataFrame(
        {"s": [dict(a=a, b=1, c=c) for a, c in zip(df.string_col, df.bigint_col)]}
    )
    assert_frame_equal(result, expected)


def test_field_access_after_case(con):
    s = ibis.struct({"a": 3})
    x = ibis.case().when(True, s).else_(ibis.struct({"a": 4})).end()
    y = x.a
    assert con.to_pandas(y) == 3


def test_collect_into_struct(alltypes):
    from ibis import _

    t = alltypes
    expr = (
        t[_.string_col.isin(("0", "1"))]
        .group_by(group="string_col")
        .agg(
            val=lambda t: ibis.struct(
                dict(key=t.bigint_col.collect().cast("array<int64>"))
            )
        )
    )
    result = expr.execute()
    assert result.shape == (2, 2)
    assert set(result.group) == {"0", "1"}
    val = result.val
    assert len(val.loc[result.group == "0"].iat[0]["key"]) == 730
    assert len(val.loc[result.group == "1"].iat[0]["key"]) == 730
