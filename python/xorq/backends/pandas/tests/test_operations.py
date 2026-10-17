from __future__ import annotations

import operator
from operator import methodcaller

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from pytest import param

import xorq.api as xo
from xorq.api import _
from xorq.backends.pandas import Backend
from xorq.common.exceptions import UnsupportedOperationError
from xorq.tests.util import assert_frame_equal, assert_series_equal
from xorq.vendor import ibis


def test_table_column(t, df):
    expr = t.plain_int64
    result = expr.execute()
    expected = df.plain_int64
    assert_series_equal(result, expected)


def test_literal(client):
    assert client.execute(ibis.literal(1)) == 1


def test_selection(t, df):
    expr = t[((t.plain_strings == "a") | (t.plain_int64 == 3)) & (t.dup_strings == "d")]
    result = expr.execute()
    expected = df[
        ((df.plain_strings == "a") | (df.plain_int64 == 3)) & (df.dup_strings == "d")
    ].reset_index(drop=True)

    assert_frame_equal(result[expected.columns], expected)


def test_mutate(t, df):
    expr = t.mutate(x=t.plain_int64 + 1, y=t.plain_int64 * 2)
    result = expr.execute()
    expected = df.assign(x=df.plain_int64 + 1, y=df.plain_int64 * 2)
    assert_frame_equal(result[expected.columns], expected)


def test_project_scope_does_not_override(t, df):
    col = t.plain_int64
    expr = t[
        [
            col.name("new_col"),
            col.sum().over(ibis.window(group_by="dup_strings")).name("grouped"),
        ]
    ]
    result = expr.execute()
    expected = pd.concat(
        [
            df[["plain_int64", "dup_strings"]].rename(
                columns={"plain_int64": "new_col"}
            ),
            df.groupby("dup_strings")
            .plain_int64.transform("sum")
            .reset_index(drop=True)
            .rename("grouped"),
        ],
        axis=1,
    )[["new_col", "grouped"]]
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "where",
    [
        param(lambda _: None, id="none"),
        param(lambda t: t.dup_strings == "d", id="simple"),
        param(lambda t: (t.dup_strings == "d") | (t.plain_int64 < 100), id="complex"),
    ],
)
@pytest.mark.parametrize(
    ("ibis_func", "pandas_func"),
    [
        param(methodcaller("abs"), np.abs, id="abs"),
        param(methodcaller("ceil"), np.ceil, id="ceil"),
        param(methodcaller("exp"), np.exp, id="exp"),
        param(methodcaller("floor"), np.floor, id="floor"),
        param(methodcaller("ln"), np.log, id="log"),
        param(methodcaller("log10"), np.log10, id="log10"),
        param(methodcaller("log", 2), lambda x: np.log(x) / np.log(2), id="logb"),
        param(methodcaller("log2"), np.log2, id="log2"),
        param(
            methodcaller("round", 0), lambda x: x.round(0).astype("int64"), id="round0"
        ),
        param(methodcaller("round", -2), methodcaller("round", -2), id="roundm2"),
        param(methodcaller("round", 2), methodcaller("round", 2), id="round2"),
        param(methodcaller("round"), lambda x: x.round().astype("int64"), id="round"),
        param(methodcaller("sign"), np.sign, id="sign"),
        param(methodcaller("sqrt"), np.sqrt, id="sqrt"),
    ],
)
def test_aggregation_group_by(t, df, where, ibis_func, pandas_func):
    ibis_where = where(t)
    expr = t.group_by(t.dup_strings).aggregate(
        avg_plain_int64=t.plain_int64.mean(where=ibis_where),
        sum_plain_float64=t.plain_float64.sum(where=ibis_where),
        mean_float64_positive=ibis_func(t.float64_positive).mean(where=ibis_where),
        neg_mean_int64_with_zeros=(-t.int64_with_zeros).mean(where=ibis_where),
        nunique_dup_ints=t.dup_ints.nunique(),
    )
    result = expr.execute()

    pandas_where = where(df)
    mask = slice(None) if pandas_where is None else pandas_where
    expected = (
        df.groupby("dup_strings")
        .agg(
            {
                "plain_int64": lambda x, mask=mask: x[mask].mean(),
                "plain_float64": lambda x, mask=mask: x[mask].sum(),
                "dup_ints": "nunique",
                "float64_positive": (
                    lambda x, mask=mask, func=pandas_func: func(x[mask]).mean()
                ),
                "int64_with_zeros": lambda x, mask=mask: (-x[mask]).mean(),
            }
        )
        .reset_index()
        .rename(
            columns={
                "plain_int64": "avg_plain_int64",
                "plain_float64": "sum_plain_float64",
                "dup_ints": "nunique_dup_ints",
                "float64_positive": "mean_float64_positive",
                "int64_with_zeros": "neg_mean_int64_with_zeros",
            }
        )
    )
    lhs = result[expected.columns]
    rhs = expected
    assert_frame_equal(lhs, rhs)


def test_aggregation_without_group_by(t, df):
    expr = t.aggregate(
        avg_plain_int64=t.plain_int64.mean(),
        sum_plain_float64=t.plain_float64.sum(),
    )
    result = expr.execute()[["avg_plain_int64", "sum_plain_float64"]]
    new_names = {
        "plain_float64": "sum_plain_float64",
        "plain_int64": "avg_plain_int64",
    }
    expected = (
        pd.Series(
            [df["plain_int64"].mean(), df["plain_float64"].sum()],
            index=["plain_int64", "plain_float64"],
        )
        .to_frame()
        .T.rename(columns=new_names)
    )
    assert_frame_equal(result[expected.columns], expected)


def test_group_by_with_having(t, df):
    expr = (
        t.group_by(t.dup_strings)
        .having(t.plain_float64.sum() == 5)
        .aggregate(avg_a=t.plain_int64.mean(), sum_c=t.plain_float64.sum())
    )
    result = expr.execute()

    expected = (
        df.groupby("dup_strings")
        .agg({"plain_int64": "mean", "plain_float64": "sum"})
        .reset_index()
        .rename(columns={"plain_int64": "avg_a", "plain_float64": "sum_c"})
    )
    expected = expected.loc[expected.sum_c == 5, ["avg_a", "sum_c"]]

    assert_frame_equal(result[expected.columns], expected)


def test_group_by_rename_key(t, df):
    expr = t.group_by(t.dup_strings.name("foo")).aggregate(
        dup_string_count=t.dup_strings.count()
    )
    assert "foo" in expr.schema()
    result = expr.execute()
    assert "foo" in result.columns

    expected = (
        df.groupby("dup_strings")
        .dup_strings.count()
        .rename("dup_string_count")
        .reset_index()
        .rename(columns={"dup_strings": "foo"})
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("reduction", ["mean", "sum", "count", "std", "var"])
@pytest.mark.parametrize(
    "where",
    [
        lambda t: (t.plain_strings == "a") | (t.plain_strings == "c"),
        lambda t: (
            (t.dup_strings == "d") & ((t.plain_int64 == 1) | (t.plain_int64 == 3))
        ),
        lambda t: None,
    ],
)
def test_reduction(t, df, reduction, where):
    func = getattr(t.plain_int64, reduction)
    mask = where(t)
    expr = func(where=mask)
    result = expr.execute()

    df_mask = where(df)
    expected_func = getattr(
        df.loc[df_mask if df_mask is not None else slice(None), "plain_int64"],
        reduction,
    )
    expected = expected_func()
    assert result == expected


@pytest.mark.parametrize(
    "reduction",
    [
        lambda x: x.any(),
        lambda x: x.all(),
        lambda x: ~(x.any()),
        lambda x: ~(x.all()),
    ],
)
def test_boolean_aggregation(t, df, reduction):
    expr = reduction(t.plain_int64 == 1)
    result = expr.execute()
    expected = reduction(df.plain_int64 == 1)
    assert result == expected


@pytest.mark.parametrize("column", ["float64_with_zeros", "int64_with_zeros"])
def test_nullif_zero(t, df, column):
    expr = t[column].nullif(0)
    result = expr.execute()
    expected = df[column].replace(0, np.nan)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("left", "right", "expected", "compare"),
    [
        param(
            lambda t: ibis.literal(1),
            lambda t: ibis.literal(1),
            lambda df: np.nan,
            np.testing.assert_array_equal,  # treats NaNs as equal
            id="literal_literal_equal",
        ),
        param(
            lambda t: ibis.literal(1),
            lambda t: ibis.literal(2),
            lambda df: 1,
            np.testing.assert_equal,
            id="literal_literal_not_equal",
        ),
        param(
            lambda t: t.dup_strings,
            lambda t: ibis.literal("a"),
            lambda df: df.dup_strings.where(df.dup_strings != "a"),
            assert_series_equal,
            id="series_literal",
        ),
        param(
            lambda t: t.dup_strings,
            lambda t: t.dup_strings,
            lambda df: df.dup_strings.where(df.dup_strings != df.dup_strings),
            assert_series_equal,
            id="series_series",
        ),
        param(
            lambda t: ibis.literal("a"),
            lambda t: t.dup_strings,
            lambda _: pd.Series(["a", np.nan, "a"], name="dup_strings"),
            assert_series_equal,
            id="literal_series",
        ),
    ],
)
def test_nullif(t, df, left, right, expected, compare):
    expr = left(t).nullif(right(t))
    result = Backend().execute(expr)
    compare(result, expected(df))


def test_nullif_inf():
    df = pd.DataFrame({"a": [np.inf, 3.14, -np.inf, 42.0]})
    con = Backend().connect({"t": df})
    t = con.table("t")
    expr = t.a.nullif(np.inf).nullif(-np.inf)
    result = expr.execute()
    expected = pd.Series([np.nan, 3.14, np.nan, 42.0], name="a")
    assert_series_equal(result, expected)


def test_group_concat(t, df):
    expr = t.group_by(t.dup_strings).aggregate(foo=t.plain_int64.group_concat(","))
    result = expr.execute()
    expected = (
        df.groupby("dup_strings")
        .apply(lambda df: ",".join(df.plain_int64.astype(str)))
        .reset_index()
        .rename(columns={0: "foo"})
    )
    assert_frame_equal(result[expected.columns], expected)


@pytest.fixture
def concat_table():
    df = pd.DataFrame(
        {
            "g": [1, 1, 1, 2, 2],
            "i": pd.array([1, 2, None, 3, 4], dtype="Int64"),
            "s": ["a", "b", None, "c", "d"],
            "k": [1, 1, 2, 1, 2],
        }
    )
    return Backend().connect({"t": df}).table("t")


def _concat_values(expr):
    values = expr.order_by("g").execute().r
    return [None if pd.isna(value) else value for value in values]


# expected values are what duckdb returns for the same expressions:
# nulls take no part, and a group left with nothing is null
@pytest.mark.parametrize(
    ("column", "key", "expected"),
    [
        param("i", 1, ["1,2", "3"], id="int-where"),
        param("s", 1, ["a,b", "c"], id="str-where"),
        param("i", 2, [None, "4"], id="int-where-null-rows"),
        param("s", 2, [None, "d"], id="str-where-null-rows"),
    ],
)
def test_group_concat_where(concat_table, column, key, expected):
    t = concat_table
    expr = t.group_by("g").agg(r=t[column].group_concat(",", where=t.k == key))
    assert _concat_values(expr) == expected


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        param("i", ["1,2", "3,4"], id="int"),
        param("s", ["a,b", "c,d"], id="str"),
    ],
)
def test_group_concat_skips_nulls(concat_table, column, expected):
    t = concat_table
    expr = t.group_by("g").agg(r=t[column].group_concat(","))
    assert _concat_values(expr) == expected


@pytest.mark.parametrize("offset", [0, 2])
def test_frame_limit(t, df, offset):
    n = 5
    df_expr = t.limit(n, offset=offset)
    result = df_expr.execute()
    expected = df.iloc[offset : offset + n].reset_index(drop=True)
    assert_frame_equal(result[expected.columns], expected)


@pytest.mark.parametrize("offset", [0, 2])
def test_series_limit(t, df, offset):
    with pytest.raises(AttributeError):
        t.plain_int64.limit(5, offset=offset)


@pytest.mark.parametrize(
    ("key", "pandas_by", "pandas_ascending"),
    [
        (lambda t, col: [ibis.desc(t[col])], lambda col: [col], False),
        (
            lambda t, col: [t[col], ibis.desc(t.plain_int64)],
            lambda col: [col, "plain_int64"],
            [True, False],
        ),
        (
            lambda t, col: [ibis.desc(t.plain_int64 * 2)],
            lambda col: ["plain_int64"],
            False,
        ),
    ],
)
@pytest.mark.parametrize(
    "column",
    ["plain_datetimes_naive", "plain_datetimes_ny", "plain_datetimes_utc"],
)
def test_order_by(t, df, column, key, pandas_by, pandas_ascending):
    expr = t.order_by(key(t, column))
    result = expr.execute()
    expected = df.sort_values(
        pandas_by(column), ascending=pandas_ascending
    ).reset_index(drop=True)
    assert_frame_equal(result[expected.columns], expected)


def test_complex_order_by(t, df):
    expr = t.order_by([ibis.desc(t.plain_int64 * t.plain_float64), t.plain_float64])
    result = expr.execute()
    expected = (
        df.assign(foo=df.plain_int64 * df.plain_float64)
        .sort_values(["foo", "plain_float64"], ascending=[False, True])
        .drop(["foo"], axis=1)
        .reset_index(drop=True)
    )

    assert_frame_equal(result[expected.columns], expected)


def test_count_distinct(t, df):
    expr = t.dup_strings.nunique()
    result = expr.execute()
    expected = df.dup_strings.nunique()
    assert result == expected


def test_value_counts(t, df):
    expr = t.dup_strings.value_counts()
    result = expr.execute()
    expected = (
        df.dup_strings.value_counts()
        .rename("dup_strings")
        .reset_index(name="dup_strings_count")
        .rename(columns={"index": "dup_strings"})
        .sort_values(["dup_strings"])
        .reset_index(drop=True)
    )
    assert_frame_equal(result[expected.columns], expected)


def test_table_count(t, df):
    expr = t.count()
    result = expr.execute()
    expected = len(df)
    assert result == expected


def test_weighted_average(t, df):
    expr = t.group_by(t.dup_strings).aggregate(
        avg=(t.plain_float64 * t.plain_int64).sum() / t.plain_int64.sum()
    )
    result = expr.execute()
    expected = (
        df.groupby("dup_strings")
        .apply(
            lambda df: (df.plain_int64 * df.plain_float64).sum() / df.plain_int64.sum()
        )
        .reset_index()
        .rename(columns={0: "avg"})
    )
    assert_frame_equal(result[expected.columns], expected)


def test_group_by_multiple_keys(t, df):
    expr = t.group_by([t.dup_strings, t.dup_ints]).aggregate(
        avg_plain_float64=t.plain_float64.mean()
    )
    result = expr.execute()
    expected = (
        df.groupby(["dup_strings", "dup_ints"])
        .agg({"plain_float64": "mean"})
        .reset_index()
        .rename(columns={"plain_float64": "avg_plain_float64"})
    )
    assert_frame_equal(result[expected.columns], expected)


def test_mutate_after_group_by(t, df):
    gb = t.group_by(t.dup_strings).aggregate(avg_plain_float64=t.plain_float64.mean())
    expr = gb.mutate(x=gb.avg_plain_float64)
    result = expr.execute()
    expected = (
        df.groupby("dup_strings")
        .agg({"plain_float64": "mean"})
        .reset_index()
        .rename(columns={"plain_float64": "avg_plain_float64"})
    )
    expected = expected.assign(x=expected.avg_plain_float64)
    assert_frame_equal(result[expected.columns], expected)


def test_groupby_with_unnamed_arithmetic(t, df):
    expr = t.group_by(t.dup_strings).aggregate(
        naive_variance=((t.plain_float64**2).sum() - t.plain_float64.mean() ** 2)
        / t.plain_float64.count()
    )
    result = expr.execute()
    expected = (
        df.groupby("dup_strings")
        .agg({"plain_float64": lambda x: ((x**2).sum() - x.mean() ** 2) / x.count()})
        .reset_index()
        .rename(columns={"plain_float64": "naive_variance"})
    )
    assert_frame_equal(result[expected.columns], expected)


def test_isnull(t, df):
    expr = t.strings_with_nulls.isnull()
    result = expr.execute()
    expected = df.strings_with_nulls.isnull()
    assert_series_equal(result, expected)


def test_notnull(t, df):
    expr = t.strings_with_nulls.notnull()
    result = expr.execute()
    expected = df.strings_with_nulls.notnull()
    assert_series_equal(result, expected)


@pytest.mark.parametrize("raw_value", [0.0, 1.0])
def test_scalar_parameter(t, df, raw_value):
    value = xo.param("value", "float64")
    expr = t.float64_with_zeros == value
    result = expr.execute(params={"value": raw_value})
    expected = df.float64_with_zeros == raw_value
    assert_series_equal(result, expected)


@pytest.mark.parametrize("elements", [[1], (1,), {1}, frozenset({1})])
def test_isin(t, df, elements):
    expr = t.plain_float64.isin(elements)
    expected = df.plain_float64.isin(elements)
    result = expr.execute()
    assert_series_equal(result, expected)


@pytest.mark.parametrize("elements", [[1], (1,), {1}, frozenset({1})])
def test_notin(t, df, elements):
    expr = t.plain_float64.notin(elements)
    expected = ~df.plain_float64.isin(elements)
    result = expr.execute()
    assert_series_equal(result, expected)


def test_cast_on_group_by(t, df):
    expr = t.group_by(t.dup_strings).aggregate(
        casted=(t.float64_with_zeros == 0).cast("int64").sum()
    )
    result = expr.execute()
    expected = (
        df.groupby("dup_strings")
        .float64_with_zeros.apply(lambda s: (s == 0).astype("int64").sum())
        .reset_index()
        .rename(columns={"float64_with_zeros": "casted"})
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "opname", ["add", "mul", "sub", "truediv", "floordiv", "mod", "pow"]
)
@pytest.mark.parametrize(
    "argfunc", [param(lambda c: (1.0, c), id="1c"), param(lambda c: (c, 1.0), id="c1")]
)
def test_left_binary_op(t, df, opname, argfunc):
    op = getattr(operator, opname)
    left, right = argfunc(t.float64_with_zeros)
    expr = op(left, right)
    result = expr.execute()
    expected = op(*argfunc(df.float64_with_zeros))
    assert_series_equal(
        result,
        expected,
        check_dtype=not (isinstance(right, float) and opname == "floordiv"),
    )


@pytest.mark.parametrize(
    "opname", ["add", "mul", "sub", "truediv", "floordiv", "mod", "pow"]
)
@pytest.mark.parametrize(
    "argfunc", [param(lambda c: (1.0, c), id="1c"), param(lambda c: (c, 1.0), id="c1")]
)
def test_left_binary_op_gb(t, df, opname, argfunc):
    op = getattr(operator, opname)
    left, right = argfunc(t.float64_with_zeros)
    expr = t.group_by("dup_strings").aggregate(foo=op(left, right).sum())
    result = expr.execute()
    expected = (
        df.groupby("dup_strings")
        .float64_with_zeros.apply(lambda s: op(*argfunc(s)).sum())
        .reset_index()
        .rename(columns={"float64_with_zeros": "foo"})
    )
    assert_frame_equal(
        result,
        expected,
        check_dtype=not (isinstance(right, float) and opname == "floordiv"),
    )


@pytest.mark.parametrize(
    "left_f",
    [
        param(lambda e: e - 1, id="sub"),
        param(lambda _: 0.0, id="zero"),
        param(lambda _: None, id="none"),
    ],
)
@pytest.mark.parametrize(
    "right_f",
    [
        param(lambda e: e + 1, id="add"),
        param(lambda _: 1.0, id="one"),
        param(lambda _: None, id="none"),
    ],
)
def test_ifelse_series(t, df, left_f, right_f):
    col_expr = t["plain_int64"]
    result = ibis.ifelse(
        col_expr > col_expr.mean(), left_f(col_expr), right_f(col_expr)
    ).execute()

    series = df["plain_int64"]
    cond = series > series.mean()
    left = left_f(series)
    if not isinstance(left, pd.Series):
        left = pd.Series(np.repeat(left, len(cond)), name=cond.name)
    expected = left.where(cond, right_f(series))

    assert_series_equal(
        result.astype(object).fillna(pd.NA),
        expected.astype(object).fillna(pd.NA),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ("cond", "expected_func"),
    [
        param(True, lambda df: df["plain_int64"].astype("float64"), id="true"),
        param(False, lambda df: pd.Series(np.repeat(3.0, len(df))), id="false"),
    ],
)
def test_ifelse_scalar(t, df, cond, expected_func):
    expr = ibis.ifelse(cond, t["plain_int64"], 3.0)
    result = expr.execute()
    expected = expected_func(df)
    assert_series_equal(result, expected)


def test_ifelse_long(batting, batting_df):
    col_expr = batting["AB"]
    result = ibis.ifelse(col_expr > col_expr.mean(), col_expr, 0.0).execute()

    series = batting_df["AB"]
    expected = series.where(series > series.mean(), other=0.0).astype("float64")

    assert_series_equal(result, expected)


def test_round(t, df):
    precision = 2
    mult = 3.33333
    result = (t.count() * mult).round(precision).execute()
    expected = np.around(len(df) * mult, precision)
    npt.assert_almost_equal(result, expected, decimal=precision)


# expected values are what duckdb and the xorq datafusion backend return for
# the same expression: a null stays null, ties round half away from zero on
# both signs (numpy rounds half to even), a value just below a half does not
# move, an infinity stays (datafusion refuses it), and integers come back
# unchanged (datafusion alone loses the int64 values past 2**53)
@pytest.mark.parametrize(
    ("dtype", "data", "expected"),
    [
        param("float64", [1.26, 2.5, None, 4.75], [1.0, 3.0, None, 5.0], id="null"),
        param("float64", [1.26, 2.5, 3.5, 4.75], [1, 3, 4, 5], id="ties"),
        param("float64", [-2.5, -0.5, 0.5, -1.5], [-3, -1, 1, -2], id="ties-negative"),
        param(
            "float64",
            [0.49999999999999994, -0.49999999999999994, 4503599627370497.0],
            [0, 0, 4503599627370497],
            id="not-a-tie",
        ),
        param("float64", [np.inf, -np.inf, 1.5], [np.inf, -np.inf, 2.0], id="inf"),
        param("Float64", [2.5, None, -0.5], [3.0, None, -1.0], id="nullable-float64"),
        param("Float32", [2.5, None, -0.5], [3.0, None, -1.0], id="nullable-float32"),
        param("int8", [-128, 127], [-128, 127], id="int8"),
        param(
            "int64",
            [9007199254740993, -9007199254740993],
            [9007199254740993, -9007199254740993],
            id="int64",
        ),
    ],
)
def test_round_no_digits(dtype, data, expected):
    df = pd.DataFrame({"k": range(len(data)), "a": pd.Series(data, dtype=dtype)})
    t = Backend().connect({"t": df}).table("t")
    result = t.select("k", r=t.a.round()).order_by("k").execute().r
    assert [None if pd.isna(value) else value for value in result] == expected


# expected values are what duckdb and the xorq datafusion backend return: the
# tie is decided on the value scaled by 10**digits, half away from zero, so
# 0.35 (3.5 once scaled) and 2.65 (26.5) both round up where numpy rounds to
# even; a null stays null and an infinity stays
@pytest.mark.parametrize(
    ("digits", "data", "expected"),
    [
        param(1, [0.25, 0.35, 2.65, -0.25], [0.3, 0.4, 2.7, -0.3], id="ties"),
        param(1, [1.5, None, np.inf], [1.5, None, np.inf], id="null-and-inf"),
        param(2, [1.005, 2.675, 0.125], [1.0, 2.68, 0.13], id="two-digits"),
        param(0, [0.5, 2.5, -0.5, 1.4], [1.0, 3.0, -1.0, 1.0], id="zero-digits"),
        param(
            -1, [15.0, 25.0, -15.0, 1e16 + 2], [20.0, 30.0, -20.0, 1e16], id="negative"
        ),
    ],
)
def test_round_digits(digits, data, expected):
    df = pd.DataFrame({"k": range(len(data)), "a": data})
    t = Backend().connect({"t": df}).table("t")
    result = t.select("k", r=t.a.round(digits)).order_by("k").execute().r
    assert result.dtype == "float64"
    assert [None if pd.isna(value) else value for value in result] == expected


def test_round_widens_before_arithmetic():
    # round() is typed int64, so negating int8 -128 gives 128 on the xorq
    # datafusion backend (duckdb refuses the negation as an overflow)
    df = pd.DataFrame({"a": pd.Series([-128, 127], dtype="int8")})
    t = Backend().connect({"t": df}).table("t")
    assert (-t.a.round()).execute().tolist() == [128, -127]


def test_quantile_groupby(batting, batting_df):
    def q_fun(x, quantile):
        res = x.quantile(quantile).tolist()
        return [res for _ in range(len(x))]

    frac = 0.2
    result = (
        batting.group_by("teamID")
        .mutate(res=lambda x: x.RBI.quantile([frac, 1 - frac]))
        .res.execute()
    )
    expected = (
        batting_df.groupby("teamID")
        .RBI.transform(q_fun, quantile=[frac, 1 - frac])
        .rename("res")
    )
    assert_series_equal(result, expected)


def test_summary_numeric(batting, batting_df):
    expr = batting.aggregate(
        count=_.G.count(),
        nulls=_.G.isnull().sum(),
        min=_.G.min(),
        max=_.G.max(),
        sum=_.G.sum(),
        mean=_.G.mean(),
        approx_nunique=_.G.nunique(),
    )
    result = expr.execute()
    assert len(result) == 1

    G = batting_df.G
    expected = {
        "count": G.count(),
        "nulls": G.isnull().sum(),
        "min": G.min(),
        "max": G.max(),
        "sum": G.sum(),
        "mean": G.mean(),
        "approx_nunique": G.nunique(),
    }
    assert dict(result.iloc[0]) == expected


def test_summary_non_numeric(batting, batting_df):
    expr = batting.aggregate(
        count=_.teamID.count(),
        nulls=_.teamID.isnull().sum(),
        uniques=_.teamID.nunique(),
    )
    result = expr.execute()
    assert len(result) == 1
    assert len(result.columns) == 3
    expected = {
        "count": batting_df.teamID.count(),
        "nulls": batting_df.teamID.isnull().sum(),
        "uniques": batting_df.teamID.nunique(),
    }
    assert dict(result.iloc[0]) == expected


def test_non_range_index():
    def do_replace(col):
        return col.cases(
            *(
                (1, "one"),
                (2, "two"),
            ),
            else_="unk",
        )

    df = pd.DataFrame(
        {
            "A": pd.Series({i: i % 3 for i in (0, 1, 2, 4)}),
            "B": 0,
        }
    )
    expr = (
        xo.pandas.connect({"t": df})
        .table("t")
        .mutate(A=lambda t: t["A"].pipe(do_replace))
    )
    assert df.index.equals(expr.execute().index)


def test_table_distinct(t, df):
    expr = t[["dup_strings"]].distinct()
    result = expr.execute()
    expected = df[["dup_strings"]].drop_duplicates()
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("distinct", [True, False])
def test_union(client, df1, distinct):
    t = client.table("df1")
    expr = t.union(t, distinct=distinct)
    result = expr.execute()
    expected = df1 if distinct else pd.concat([df1, df1], axis=0, ignore_index=True)
    assert_frame_equal(result, expected)


def test_intersect(client, df1, intersect_df2):
    t1 = client.table("df1")
    t2 = client.table("intersect_df2")
    expr = t1.intersect(t2)
    result = expr.execute()
    expected = df1.merge(intersect_df2, on=list(df1.columns))
    assert_frame_equal(result, expected)


def test_difference(client, df1, intersect_df2):
    t1 = client.table("df1")
    t2 = client.table("intersect_df2")
    expr = t1.difference(t2)
    result = expr.execute()
    merged = df1.merge(intersect_df2, on=list(df1.columns), how="outer", indicator=True)
    expected = merged[merged["_merge"] != "both"].drop("_merge", axis=1)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "distinct",
    [
        param(
            True,
            marks=pytest.mark.xfail(
                raises=TypeError,
                reason="Pandas cannot compute the distinct element of an array column",
            ),
        ),
        False,
    ],
)
def test_union_with_list_types(t, df, distinct):
    expr = t.union(t, distinct=distinct)
    result = expr.execute()
    expected = df if distinct else pd.concat([df, df], axis=0, ignore_index=True)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda column: column + column,
            id="sum",
        ),
        pytest.param(
            lambda column: column + 1,
            id="sum_scalar",
        ),
        pytest.param(
            lambda column: column - column,
            id="subtract",
        ),
        pytest.param(
            lambda column: column - 1,
            id="subtract_scalar",
        ),
        pytest.param(
            lambda column: column * column,
            id="multiply",
        ),
        pytest.param(
            lambda column: column * 2,
            id="multiply_scalar",
        ),
        pytest.param(
            lambda column: column % column,
            id="mod",
            marks=pytest.mark.xfail(
                raises=ZeroDivisionError,
                reason=("Ibis cannot modulo divide two unsigned integer columns."),
            ),
        ),
        pytest.param(
            lambda column: column % 2,
            id="mod_scalar",
        ),
        pytest.param(
            lambda column: column / column,
            id="divide",
        ),
        pytest.param(
            lambda column: column / 2,
            id="divide_scalar",
        ),
        pytest.param(
            lambda column: column // column,
            id="floordivide",
        ),
        pytest.param(
            lambda column: column // 2,
            id="floordivide_scalar",
        ),
    ],
)
def test_unsigned_integers(t, df, operation):
    expr = operation(t.plain_uint64)
    result = expr.execute()
    expected = operation(df.plain_uint64)
    expected_dtype = expr.type().to_pandas()
    assert result.dtype == expected_dtype
    assert_series_equal(result, expected.astype(expected_dtype))


@pytest.mark.parametrize(
    ("column", "expr", "expected"),
    [
        param(
            pd.array([1, None, 3], dtype="Int64"),
            lambda c: c.cases((1, 10), else_=0),
            [10, 0, 0],
            id="int64",
        ),
        param(
            pd.array([True, None, False], dtype="boolean"),
            lambda c: c.cases((True, 1), else_=0),
            [1, 0, 0],
            id="boolean",
        ),
        param(
            pd.array([1, None, 3], dtype="Int64"),
            lambda c: xo.cases((c > 2, "big"), else_="small"),
            ["small", "small", "big"],
            id="searched",
        ),
        # a null case value matches no row, and keeps the column's shape
        # even when it is the only case
        param(
            pd.array([1, None, 3], dtype="Int64"),
            lambda c: c.cases((None, 10), else_=0),
            [0, 0, 0],
            id="null-case-value",
        ),
    ],
)
def test_case_on_nullable_column(column, expr, expected):
    # pandas' nullable arrays hold pd.NA where np.select wants a boolean
    # ndarray; a missing condition is a false one
    t = xo.pandas.connect({"t": pd.DataFrame({"c": column})}).table("t")
    assert expr(t.c).execute().tolist() == expected


def test_simple_case_null_base():
    # a null base matches no branch
    con = xo.pandas.connect()
    expr = xo.literal(None, type="int64").cases((1, 10), else_=0)
    assert con.execute(expr) == 0


def test_simple_case_null_base_and_null_branch():
    # null never equals null, so the else branch still wins
    con = xo.pandas.connect()
    expr = xo.literal(None, type="int64").cases((None, 10), else_=0)
    assert con.execute(expr) == 0


@pytest.mark.parametrize("how", ["sample", "pop"])
@pytest.mark.parametrize("method", ["var", "std"])
def test_variance_and_standard_deviation(t, df, method, how):
    expr = getattr(t.plain_float64, method)(how=how)
    ddof = {"sample": 1, "pop": 0}[how]
    expected = getattr(df.plain_float64, method)(ddof=ddof)
    assert expr.execute() == pytest.approx(expected)


@pytest.mark.parametrize("method", ["corr", "cov"])
def test_correlation_and_covariance(t, df, method):
    expr = getattr(t.plain_int64, method)(t.plain_float64)
    expected = getattr(df.plain_int64, method)(df.plain_float64)
    assert expr.execute() == pytest.approx(expected)


@pytest.mark.parametrize("method", ["corr", "cov"])
def test_correlation_and_covariance_where(t, df, method):
    expr = getattr(t.plain_int64, method)(t.plain_float64, where=t.dup_ints == 1)
    mask = df.dup_ints == 1
    expected = getattr(df.plain_int64[mask], method)(df.plain_float64[mask])
    assert expr.execute() == pytest.approx(expected)


@pytest.mark.parametrize(("method", "position"), [("first", 0), ("last", -1)])
def test_first_and_last_where(t, df, method, position):
    expr = getattr(t.plain_strings, method)(where=t.dup_ints == 1)
    expected = df.plain_strings[df.dup_ints == 1].iat[position]
    assert expr.execute() == expected


@pytest.mark.parametrize(
    "make_expr",
    [
        param(lambda t: t.plain_int64.first(order_by=t.plain_float64), id="first"),
        param(lambda t: t.plain_int64.last(order_by=t.plain_float64), id="last"),
        param(lambda t: t.plain_int64.collect(order_by=t.plain_float64), id="collect"),
        param(
            lambda t: t.plain_strings.group_concat(",", order_by=t.plain_float64),
            id="group_concat",
        ),
    ],
)
def test_ordered_reduction_is_unsupported(t, make_expr):
    with pytest.raises(UnsupportedOperationError, match="order_by"):
        make_expr(t).execute()


def test_between_scalar_and_column_bounds(client):
    df = pd.DataFrame(
        {"a": [1.26, 2.5, 3.5, 4.75], "lo": [0, 3, 0, 4], "hi": [2, 3, 5, 5]}
    )
    t = client.create_table("between_t", df)

    assert t.a.between(1, 3).execute().tolist() == [True, True, False, False]
    assert t.a.between(t.lo, t.hi).execute().tolist() == [True, False, True, True]


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        param(
            lambda t: (1, 3),
            [True, True, None, False, True, False, False],
            id="scalar_bounds",
        ),
        param(
            lambda t: (t.lo, t.hi),
            [True, False, None, None, None, False, False],
            id="column_bounds",
        ),
    ],
)
def test_between_null_operand_is_null(client, bounds, expected):
    # expected is the literal output of duckdb (xo.duckdb.connect()) and of the
    # xorq datafusion backend (xo.connect()) for the same table: a null value
    # or a null bound gives null, not False, unless the other bound already
    # excludes the value (FALSE AND NULL is FALSE)
    df = pd.DataFrame(
        {
            "a": [1.26, 2.5, None, 4.75, 2.0, 5.0, 5.0],
            "lo": [0, 3, 0, 4, None, 10, None],
            "hi": [2, 3, 5, None, 5, None, 1],
        }
    )
    t = client.create_table("between_null_t", df, overwrite=True)

    result = t.a.between(*bounds(t)).execute()
    assert_series_equal(result.astype("boolean"), pd.Series(expected, dtype="boolean"))


def test_filter_on_negated_between_drops_null_rows(client):
    # duckdb and datafusion keep one row (4.75): NOT NULL is NULL, so the row
    # with the null value is dropped like the rows inside the range; a row a
    # known bound excludes is kept even when the other bound is null
    df = pd.DataFrame(
        {
            "a": [1.26, 2.5, None, 4.75, 5.0],
            "lo": [0, 0, 0, 0, 10],
            "hi": [3, 3, 3, 3, None],
        }
    )
    t = client.create_table("between_filter_t", df)

    assert t.filter(~t.a.between(1, 3)).count().execute() == 2
    assert t.filter(~t.a.between(t.lo, t.hi)).count().execute() == 2


# duckdb drops the row whose string is null: the predicate is null there,
# and a filter keeps only rows whose predicate is true
@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        param(lambda t: t.s.startswith("a"), ["ab"], id="startswith"),
        param(lambda t: t.s.endswith("d"), ["cd"], id="endswith"),
        param(lambda t: t.s.contains("c"), ["cd"], id="contains"),
        param(lambda t: t.s.like("a%"), ["ab"], id="like"),
    ],
)
def test_filter_on_null_bearing_string_predicate_drops_null_rows(
    client, predicate, expected
):
    df = pd.DataFrame({"s": ["ab", None, "cd"]})
    t = client.create_table("filter_null_string_t", df, overwrite=True)
    assert t.filter(predicate(t)).s.execute().tolist() == expected
    assert t.filter(predicate(t)).count().execute() == len(expected)


def test_arbitrary_returns_a_present_value(client):
    df = pd.DataFrame({"g": [1, 1, 2, 2], "s": ["ab", "c", None, "d"]})
    t = client.create_table("arbitrary_t", df)

    assert t.s.arbitrary().execute() in {"ab", "c", "d"}
    by_group = t.group_by("g").agg(any=t.s.arbitrary()).execute().sort_values("g")
    assert by_group["any"].tolist()[0] in {"ab", "c"}
    assert by_group["any"].tolist()[1] == "d"
