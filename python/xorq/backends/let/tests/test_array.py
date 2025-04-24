from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest
from pytest import param

import xorq as xo
import xorq.vendor.ibis.expr.types as ir
from xorq.tests.util import assert_frame_equal, assert_series_equal


@pytest.fixture(scope="module")
def flatten_data():
    return {
        "empty": {"data": [[], [], []], "type": "array<!array<!int64>>"},
        "happy": {
            "data": [[["abc"]], [["bcd"]], [["def"]]],
            "type": "array<!array<!string>>",
        },
        "nulls_only": {"data": [None, None, None], "type": "array<array<string>>"},
        "mixed_nulls": {"data": [[[]], None, [[None]]], "type": "array<array<string>>"},
    }


def test_array_column(alltypes, df):
    expr = xo.array([alltypes["double_col"], alltypes["double_col"]])
    assert isinstance(expr, ir.ArrayColumn)

    result = expr.execute()
    expected = df.apply(
        lambda row: [row["double_col"], row["double_col"]],
        axis=1,
    )
    assert_series_equal(result, expected, check_names=False)


def test_array_scalar(con):
    expr = xo.array([1.0, 2.0, 3.0])
    assert isinstance(expr, ir.ArrayScalar)

    result = con.execute(expr.name("tmp"))
    expected = np.array([1.0, 2.0, 3.0])

    assert np.array_equal(result, expected)


def test_array_repeat(con):
    expr = xo.array([1.0, 2.0]) * 2

    result = con.execute(expr.name("tmp"))
    expected = np.array([1.0, 2.0, 1.0, 2.0])

    assert np.array_equal(result, expected)


def test_array_concat(con):
    left = xo.literal([1, 2, 3])
    right = xo.literal([2, 1])
    expr = left + right
    result = con.execute(expr.name("tmp"))
    expected = np.array([1, 2, 3, 2, 1])
    assert np.array_equal(result, expected)


def test_array_concat_variadic(con):
    left = xo.literal([1, 2, 3])
    right = xo.literal([2, 1])
    expr = left.concat(right, right, right)
    result = con.execute(expr.name("tmp"))
    expected = np.array([1, 2, 3, 2, 1, 2, 1, 2, 1])
    assert np.array_equal(result, expected)


def test_array_radd_concat(con):
    left = [1]
    right = xo.literal([2])
    expr = left + right
    result = con.execute(expr.name("tmp"))
    expected = np.array([1, 2])

    assert np.array_equal(result, expected)


def test_array_length(con):
    expr = xo.literal([1, 2, 3]).length()
    assert con.execute(expr.name("tmp")) == 3


def test_list_literal(con):
    arr = [1, 2, 3]
    expr = xo.literal(arr)
    result = con.execute(expr.name("tmp"))

    assert np.array_equal(result, arr)


def test_np_array_literal(con):
    arr = np.array([1, 2, 3])
    expr = xo.literal(arr)
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
    t = xo.memtable({"a": [[1], [], [42, 42], []]})
    expr = t.a.index(42)
    result = con.execute(expr)
    expected = pd.Series([-1, -1, 0, -1], dtype="object")
    assert_series_equal(result, expected, check_names=False, check_dtype=False)


def test_array_remove(con):
    t = xo.memtable({"a": [[3, 2], [], [42, 2], [2, 2], []]})
    expr = t.a.remove(2)
    result = con.execute(expr)
    expected = pd.Series([[3], [], [42], [], []], dtype="object")
    assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        param("empty", pd.Series([[], [], []], dtype="object"), id="empty"),
        param(
            "happy", pd.Series([["abc"], ["bcd"], ["def"]], dtype="object"), id="happy"
        ),
    ],
)
def test_array_flatten(con, flatten_data, column, expected):
    data = flatten_data[column]
    t = xo.memtable({column: data["data"]}, schema={column: data["type"]})
    expr = t[column].flatten()
    result = con.execute(expr)
    assert_series_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


@pytest.mark.parametrize("step", [-2, -1, 1, 2])
@pytest.mark.parametrize(
    ("start", "stop"),
    [
        param(-7, -7),
        param(-7, 0),
        param(-7, 7),
        param(0, -7),
        param(0, 0),
        param(0, 7),
        param(7, -7),
        param(7, 0),
        param(7, 7),
    ],
)
def test_range_start_stop_step(con, start, stop, step):
    expr = xo.range(start, stop, step)
    result = con.execute(expr)
    assert list(result) == list(range(start, stop, step))


@pytest.mark.parametrize("n", [-2, 0, 2])
def test_range_single_argument(con, n):
    expr = xo.range(n)
    result = con.execute(expr)
    assert list(result) == list(range(n))


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        param(
            {"a": [[1, 3, 3], [], [42, 42], []]},
            [[1, 3], [], [42], []],
            id="not_null",
        ),
    ],
)
def test_array_unique(con, data, expected):
    t = xo.memtable(data)
    expr = t.a.unique()
    result = con.execute(expr)
    assert_series_equal(result, pd.Series(expected, dtype="object"))


def test_unnest_simple(array_types):
    expected = (
        array_types.execute()
        .x.explode()
        .reset_index(drop=True)
        .astype("Float64")
        .rename("tmp")
    )
    expr = array_types.x.cast("!array<float64>").unnest()
    result = expr.execute().astype("Float64").rename("tmp")

    assert_series_equal(result, expected)


def test_unnest_complex(array_types):
    df = array_types.execute()
    expr = (
        array_types.select(["grouper", "x"])
        .mutate(x=lambda t: t.x.unnest())
        .group_by("grouper")
        .aggregate(count_flat=lambda t: t.x.count())
        .order_by("grouper")
    )
    expected = (
        df[["grouper", "x"]]
        .explode("x")
        .groupby("grouper")
        .x.count()
        .rename("count_flat")
        .reset_index()
        .sort_values("grouper")
        .reset_index(drop=True)
    )
    result = expr.execute()
    assert_frame_equal(result, expected)

    # test that unnest works with to_pyarrow
    assert len(expr.to_pyarrow()) == len(result)


def test_unnest_idempotent(array_types):
    df = array_types.execute()
    expr = (
        array_types.select(
            ["scalar_column", array_types.x.cast("!array<int64>").unnest().name("x")]
        )
        .group_by("scalar_column")
        .aggregate(x=lambda t: t.x.collect())
        .order_by("scalar_column")
    )
    result = expr.execute()
    expected = (
        df[["scalar_column", "x"]]
        .assign(x=df.x.map(lambda arr: sorted(i for i in arr if not pd.isna(i))))
        .sort_values("scalar_column")
        .reset_index(drop=True)
    )
    assert_frame_equal(result, expected)


def test_unnest_no_nulls(array_types):
    df = array_types.execute()
    expr = (
        array_types.select(
            ["scalar_column", array_types.x.cast("!array<int64>").unnest().name("y")]
        )
        .filter(lambda t: t.y.notnull())
        .group_by("scalar_column")
        .aggregate(x=lambda t: t.y.collect())
        .order_by("scalar_column")
    )
    result = expr.execute()
    expected = (
        df[["scalar_column", "x"]]
        .explode("x")
        .dropna(subset=["x"])
        .groupby("scalar_column")
        .x.apply(lambda xs: [x for x in xs if x is not None])
        .reset_index()
    )
    assert_frame_equal(result, expected)


def test_unnest_default_name(array_types):
    df = array_types.execute()
    expr = (
        array_types.x.cast("!array<int64>") + xo.array([1]).cast("!array<int64>")
    ).unnest()
    assert expr.get_name().startswith("ArrayConcat(")

    result = expr.name("x").execute()
    expected = df.x.map(lambda x: x + [1]).explode("x")

    assert_series_equal(result.astype(object).fillna(pd.NA), expected.fillna(pd.NA))


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (1, 3),
        (1, 1),
        (2, 3),
        (2, 5),
        (None, 3),
        (None, None),
        (3, None),
        (-3, None),
        (-3, -1),
        (None, -3),
    ],
)
def test_array_slice(array_types, start, stop):
    expr = array_types.select(sliced=array_types.y[start:stop])
    expected = array_types.y.execute().map(lambda x: x[start:stop])
    result = expr.sliced.execute()
    assert_series_equal(result, expected)


def _agg_with_nulls(agg, x):
    if x is None:
        return None
    x = [y for y in x if not pd.isna(y)]
    if not x:
        return None
    return agg(x)


@pytest.mark.parametrize(
    ("agg", "baseline_func"),
    [
        (ir.ArrayValue.anys, partial(_agg_with_nulls, any)),
        (ir.ArrayValue.alls, partial(_agg_with_nulls, all)),
    ],
    ids=["anys", "alls"],
)
@pytest.mark.parametrize(
    "data",
    [
        param(
            [[True, None], [False, None], [None]],
            id="nulls",
        ),
        param([[True, False], [True], [False], [], None], id="no-nulls"),
    ],
)
def test_array_agg_bool(con, data, agg, baseline_func):
    t = xo.memtable({"x": data, "id": range(len(data))})
    t = t.mutate(y=agg(t.x))
    assert t.y.type().is_boolean()
    # sort so debugging is easier
    df = con.to_pandas(t.order_by("id"))
    result = df.y.tolist()
    result = [x if pd.notna(x) else None for x in result]
    expected = [baseline_func(x) for x in df.x]
    assert result == expected
