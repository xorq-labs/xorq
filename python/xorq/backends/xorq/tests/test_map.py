import pandas as pd
import pytest
from pytest import param

import xorq.api as xo
from xorq.tests.util import assert_series_equal


def test_map_get_with_compatible_value_smaller(con):
    value = xo.literal({"A": 1000, "B": 2000})
    expr = value.get("C", 3)
    assert con.execute(expr) == 3


def test_map_get_with_compatible_value_bigger(con):
    value = xo.literal({"A": 1, "B": 2})
    expr = value.get("C", 3000)
    assert con.execute(expr) == 3000


def test_map_get_with_incompatible_value_different_kind(con):
    value = xo.literal({"A": 1000, "B": 2000})
    expr = value.get("C", 3.0)
    assert con.execute(expr) == 3.0


@pytest.mark.parametrize(
    ("map", "key"),
    [
        param(
            xo.map(
                xo.literal(["a", "b"]), xo.literal(["c", "d"], type="array<string>")
            ),
            xo.literal(None, type="string"),
            id="non_null_map_null_key",
        ),
        param(
            xo.map(
                xo.literal(None, type="array<string>"),
                xo.literal(None, type="array<string>"),
            ),
            xo.literal(None, type="string"),
            id="null_both_null_key",
        ),
    ],
)
@pytest.mark.parametrize("method", ["get", "contains"])
def test_map_get_contains_nulls(con, map, key, method):
    expr = getattr(map, method)
    assert con.execute(expr(key)) is None


def test_scalar_isin_literal_map_keys(con):
    mapping = xo.literal({"a": 1, "b": 2})
    a = xo.literal("a")
    c = xo.literal("c")
    true = a.isin(mapping.keys())
    false = c.isin(mapping.keys())
    assert con.execute(true) == True  # noqa: E712
    assert con.execute(false) == False  # noqa: E712


def test_map_scalar_contains_key_scalar(con):
    mapping = xo.literal({"a": 1, "b": 2})
    a = xo.literal("a")
    c = xo.literal("c")
    true = mapping.contains(a)
    false = mapping.contains(c)
    assert con.execute(true) == True  # noqa: E712
    assert con.execute(false) == False  # noqa: E712


def test_map_scalar_contains_key_column(alltypes, alltypes_df):
    value = {"1": "a", "3": "c"}
    mapping = xo.literal(value)
    expr = mapping.contains(alltypes.string_col).name("tmp")
    result = expr.execute()
    expected = alltypes_df.string_col.apply(lambda x: x in value).rename("tmp")
    assert_series_equal(result, expected)


def test_map_column_contains_key_scalar(alltypes, alltypes_df):
    expr = xo.map(xo.array([alltypes.string_col]), xo.array([alltypes.int_col]))
    series = alltypes_df.apply(lambda row: {row["string_col"]: row["int_col"]}, axis=1)

    result = expr.contains("1").name("tmp").execute()
    series = series.apply(lambda x: "1" in x).rename("tmp")

    assert_series_equal(result, series)


def test_map_column_contains_key_column(alltypes):
    map_expr = xo.map(xo.array([alltypes.string_col]), xo.array([alltypes.int_col]))
    expr = map_expr.contains(alltypes.string_col).name("tmp")
    result = expr.execute()
    assert result.all()


@pytest.mark.parametrize(
    ("keys", "values"),
    [
        param(["a", "b"], [1, 2], id="string"),
        param(["a", "b"], ["1", "2"], id="int"),
    ],
)
def test_map_construct_dict(con, keys, values):
    expr = xo.map(keys, values)
    result = con.execute(expr.name("tmp"))
    assert result == dict(zip(keys, values))


def test_map_construct_array_column(con, alltypes, alltypes_df):
    expr = xo.map(xo.array([alltypes.string_col]), xo.array([alltypes.int_col]))
    result = con.execute(expr)
    expected = alltypes_df.apply(
        lambda row: {row["string_col"]: row["int_col"]}, axis=1
    )

    assert result.to_list() == expected.to_list()


@pytest.mark.parametrize("null_value", [None, xo.null()], ids=["none", "null"])
def test_map_get_with_null_on_null_type_with_null(con, null_value):
    value = xo.literal({"A": None, "B": None})
    expr = value.get("C", null_value)
    result = con.execute(expr)
    assert pd.isna(result)


def test_map_contains_null(con):
    expr = xo.map(["a"], xo.literal([None], type="array<string>"))
    assert con.execute(expr.contains("a"))
    assert not con.execute(expr.contains("b"))


def test_map_length(con):
    expr = xo.literal(dict(a="A", b="B")).length()
    assert con.execute(expr) == 2
