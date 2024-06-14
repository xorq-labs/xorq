from __future__ import annotations

import decimal
from operator import invert, neg

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import numpy as np
import pandas as pd
import pytest
import toolz
from ibis import _
from ibis.common.annotations import ValidationError
from pytest import param

from letsql.tests.util import assert_frame_equal, assert_series_equal


def test_null_literal(con):
    expr = ibis.null()
    assert pd.isna(con.execute(expr))
    assert con.execute(expr.typeof()) == "Null"

    assert expr.type() == dt.null
    assert pd.isna(con.execute(expr.cast(str).upper()))


def test_boolean_literal(con):
    expr = ibis.literal(False, type=dt.boolean)
    result = con.execute(expr)
    assert not result
    assert type(result) in (np.bool_, bool)
    assert con.execute(expr.typeof()) == "Boolean"


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(ibis.NA.fillna(5), 5, id="na_fillna"),
        param(ibis.literal(5).fillna(10), 5, id="non_na_fillna"),
        param(ibis.literal(5).nullif(5), None, id="nullif_null"),
        param(ibis.literal(10).nullif(5), 10, id="nullif_not_null"),
    ],
)
def test_scalar_fillna_nullif(con, expr, expected):
    if expected is None:
        assert pd.isna(con.execute(expr))
    else:
        assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("col", "filt"),
    [
        param(
            "nan_col",
            _.nan_col.isnan(),
            id="nan_col",
        ),
        param(
            "none_col",
            _.none_col.isnull(),
            id="none_col",
        ),
    ],
)
def test_isna(alltypes, col, filt):
    table = alltypes.select(
        nan_col=ibis.literal(np.nan), none_col=ibis.NA.cast("float64")
    )
    df = table.execute()

    result = table[filt].execute().reset_index(drop=True)
    expected = df[df[col].isna()].reset_index(drop=True)

    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "value",
    [
        None,
    ],
)
def test_column_fillna(alltypes, value):
    table = alltypes.mutate(missing=ibis.literal(value).cast("float64"))
    pd_table = table.execute()

    res = table.mutate(missing=table.missing.fillna(0.0)).execute()
    sol = pd_table.assign(missing=pd_table.missing.fillna(0.0))
    assert_frame_equal(res.reset_index(drop=True), sol.reset_index(drop=True))


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(ibis.coalesce(5, None, 4), 5, id="generic"),
        param(ibis.coalesce(ibis.NA, 4, ibis.NA), 4, id="null_start_end"),
        param(ibis.coalesce(ibis.NA, ibis.NA, 3.14), 3.14, id="non_null_last"),
    ],
)
def test_coalesce(con, expr, expected):
    result = con.execute(expr.name("tmp"))

    if isinstance(result, decimal.Decimal):
        # in case of Impala the result is decimal
        # >>> decimal.Decimal('5.56') == 5.56
        # False
        assert result == decimal.Decimal(str(expected))
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("column", "elements"),
    [
        ("int_col", [1, 2, 3]),
        ("int_col", (1, 2, 3)),
        ("string_col", ["1", "2", "3"]),
        ("string_col", ("1", "2", "3")),
        ("int_col", {1}),
        ("int_col", frozenset({1})),
    ],
)
def test_isin(alltypes, sorted_df, column, elements):
    sorted_alltypes = alltypes.order_by("id")
    expr = sorted_alltypes[
        "id", sorted_alltypes[column].isin(elements).name("tmp")
    ].order_by("id")
    result = expr.execute().tmp

    expected = sorted_df[column].isin(elements)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("column", "elements"),
    [
        ("int_col", [1, 2, 3]),
        ("int_col", (1, 2, 3)),
        ("string_col", ["1", "2", "3"]),
        ("string_col", ("1", "2", "3")),
        ("int_col", {1}),
        ("int_col", frozenset({1})),
    ],
)
def test_notin(alltypes, sorted_df, column, elements):
    sorted_alltypes = alltypes.order_by("id")
    expr = sorted_alltypes[
        "id", sorted_alltypes[column].notin(elements).name("tmp")
    ].order_by("id")
    result = expr.execute().tmp

    expected = ~sorted_df[column].isin(elements)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("predicate_fn", "expected_fn"),
    [
        param(lambda t: t["bool_col"], lambda df: df["bool_col"], id="no_op"),
        param(lambda t: ~t["bool_col"], lambda df: ~df["bool_col"], id="negate"),
        param(
            lambda t: t.bool_col & t.bool_col,
            lambda df: df.bool_col & df.bool_col,
            id="and",
        ),
        param(
            lambda t: t.bool_col | t.bool_col,
            lambda df: df.bool_col | df.bool_col,
            id="or",
        ),
        param(
            lambda t: t.bool_col ^ t.bool_col,
            lambda df: df.bool_col ^ df.bool_col,
            id="xor",
        ),
    ],
)
def test_filter(alltypes, sorted_df, predicate_fn, expected_fn):
    sorted_alltypes = alltypes.order_by("id")
    table = sorted_alltypes[predicate_fn(sorted_alltypes)].order_by("id")
    result = table.execute()
    expected = sorted_df[expected_fn(sorted_df)]

    assert_frame_equal(result, expected)


def test_case_where(alltypes, df):
    table = alltypes
    table = table.mutate(
        new_col=(
            ibis.case()
            .when(table["int_col"] == 1, 20)
            .when(table["int_col"] == 0, 10)
            .else_(0)
            .end()
            .cast("int64")
        )
    )

    result = table.execute()

    expected = df.copy()
    mask_0 = expected["int_col"] == 1
    mask_1 = expected["int_col"] == 0

    expected["new_col"] = 0
    expected.loc[mask_0, "new_col"] = 20
    expected.loc[mask_1, "new_col"] = 10

    assert_frame_equal(result, expected)


def test_table_fillna_invalid(alltypes):
    with pytest.raises(
        com.IbisTypeError, match=r"Column 'invalid_col' is not found in table"
    ):
        alltypes.fillna({"invalid_col": 0.0})

    with pytest.raises(
        com.IbisTypeError, match="Cannot fillna on column 'string_col' of type.*"
    ):
        alltypes[["int_col", "string_col"]].fillna(0)

    with pytest.raises(
        com.IbisTypeError, match="Cannot fillna on column 'int_col' of type.*"
    ):
        alltypes.fillna({"int_col": "oops"})


@pytest.mark.parametrize(
    "replacements",
    [
        param({"int_col": 20}, id="int"),
        param(
            {"double_col": -1, "string_col": "missing"},
            id="double-int-str",
        ),
        param(
            {"double_col": -1.5, "string_col": "missing"},
            id="double-str",
        ),
    ],
)
def test_table_fillna_mapping(alltypes, replacements):
    table = alltypes.mutate(
        int_col=alltypes.int_col.nullif(1),
        double_col=alltypes.double_col.nullif(3.0),
        string_col=alltypes.string_col.nullif("2"),
    ).select("id", "int_col", "double_col", "string_col")
    pd_table = table.execute()

    result = table.fillna(replacements).execute().reset_index(drop=True)
    expected = pd_table.fillna(replacements).reset_index(drop=True)

    assert_frame_equal(result, expected, check_dtype=False)


def test_table_fillna_scalar(alltypes):
    table = alltypes.mutate(
        int_col=alltypes.int_col.nullif(1),
        double_col=alltypes.double_col.nullif(3.0),
        string_col=alltypes.string_col.nullif("2"),
    ).select("id", "int_col", "double_col", "string_col")
    pd_table = table.execute()

    res = table[["int_col", "double_col"]].fillna(0).execute().reset_index(drop=True)
    sol = pd_table[["int_col", "double_col"]].fillna(0).reset_index(drop=True)
    assert_frame_equal(res, sol, check_dtype=False)

    res = table[["string_col"]].fillna("missing").execute().reset_index(drop=True)
    sol = pd_table[["string_col"]].fillna("missing").reset_index(drop=True)
    assert_frame_equal(res, sol, check_dtype=False)


def test_mutate_rename(alltypes):
    table = alltypes.select(["bool_col", "string_col"])
    table = table.mutate(dupe_col=table["bool_col"])
    result = table.execute()
    # check_dtype is False here because there are dtype diffs between
    # Pyspark and Pandas on Java 8 - filling the 'none_col' with an int
    # results in float in Pyspark, and int in Pandas. This diff does
    # not exist in Java 11.
    assert list(result.columns) == ["bool_col", "string_col", "dupe_col"]


def test_dropna_invalid(alltypes):
    with pytest.raises(
        com.IbisTypeError, match=r"Column 'invalid_col' is not found in table"
    ):
        alltypes.dropna(subset=["invalid_col"])

    with pytest.raises(ValidationError):
        alltypes.dropna(how="invalid")


@pytest.mark.parametrize("how", ["any", "all"])
@pytest.mark.parametrize(
    "subset", [None, [], "col_1", ["col_1", "col_2"], ["col_1", "col_3"]]
)
def test_dropna_table(alltypes, how, subset):
    is_two = alltypes.int_col == 2
    is_four = alltypes.int_col == 4

    table = alltypes.mutate(
        col_1=is_two.ifelse(ibis.NA, alltypes.float_col),
        col_2=is_four.ifelse(ibis.NA, alltypes.float_col),
        col_3=(is_two | is_four).ifelse(ibis.NA, alltypes.float_col),
    ).select("col_1", "col_2", "col_3")

    table_pandas = table.execute()
    result = table.dropna(subset, how).execute().reset_index(drop=True)
    expected = table_pandas.dropna(how=how, subset=subset).reset_index(drop=True)

    assert_frame_equal(result, expected)


def test_select_sort_sort(alltypes):
    query = alltypes[alltypes.year, alltypes.bool_col]
    query = query.order_by(query.year).order_by(query.bool_col)


@pytest.mark.parametrize(
    "key, df_kwargs",
    [
        param("id", {"by": "id"}),
        param(_.id, {"by": "id"}),
        param(lambda _: _.id, {"by": "id"}),
        param(
            ibis.desc("id"),
            {"by": "id", "ascending": False},
        ),
        param(
            ["id", "int_col"],
            {"by": ["id", "int_col"]},
        ),
        param(
            ["id", ibis.desc("int_col")],
            {"by": ["id", "int_col"], "ascending": [True, False]},
        ),
    ],
)
def test_order_by(alltypes, df, key, df_kwargs):
    result = alltypes.filter(_.id < 100).order_by(key).execute()
    expected = df.loc[df.id < 100].sort_values(**df_kwargs)
    assert_frame_equal(result, expected)


def test_order_by_random(alltypes):
    expr = alltypes.filter(_.id < 100).order_by(ibis.random()).limit(5)
    r1 = expr.execute()
    r2 = expr.execute()
    assert len(r1) == 5
    assert len(r2) == 5
    # Ensure that multiple executions returns different results
    assert not r1.equals(r2)


@pytest.mark.parametrize(
    ("ibis_op", "pandas_op"),
    [
        param(
            _.string_col.isin([]),
            lambda df: df.string_col.isin([]),
            id="isin",
        ),
        param(
            _.string_col.notin([]),
            lambda df: ~df.string_col.isin([]),
            id="notin",
        ),
        param(
            (_.string_col.length() * 1).isin([1]),
            lambda df: (df.string_col.str.len() * 1).isin([1]),
            id="isin_non_empty",
        ),
        param(
            (_.string_col.length() * 1).notin([1]),
            lambda df: ~(df.string_col.str.len() * 1).isin([1]),
            id="notin_non_empty",
        ),
    ],
)
def test_isin_notin(alltypes, df, ibis_op, pandas_op):
    expr = alltypes[ibis_op]
    expected = df.loc[pandas_op(df)].sort_values(["id"]).reset_index(drop=True)
    result = expr.execute().sort_values(["id"]).reset_index(drop=True)
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected", "op"),
    [
        param(True, True, toolz.identity, id="true_noop"),
        param(False, False, toolz.identity, id="false_noop"),
        param(True, False, invert, id="true_invert"),
        param(False, True, invert, id="false_invert"),
        param(True, False, neg, id="true_negate"),
        param(False, True, neg, id="false_negate"),
    ],
)
def test_logical_negation_literal(con, expr, expected, op):
    assert con.execute(op(ibis.literal(expr)).name("tmp")) == expected


@pytest.mark.parametrize(
    "op",
    [
        toolz.identity,
        invert,
        neg,
    ],
)
def test_logical_negation_column(alltypes, df, op):
    result = op(alltypes["bool_col"]).name("tmp").execute()
    expected = op(df["bool_col"])
    assert_series_equal(result, expected, check_names=False)


def test_ifelse_select(alltypes, df):
    table = alltypes
    table = table.select(
        [
            "int_col",
            (
                ibis.ifelse(table["int_col"] == 0, 42, -1)
                .cast("int64")
                .name("where_col")
            ),
        ]
    )

    result = table.execute()

    expected = df.loc[:, ["int_col"]].copy()

    expected["where_col"] = -1
    expected.loc[expected["int_col"] == 0, "where_col"] = 42

    assert_frame_equal(result, expected)


def test_ifelse_column(alltypes, df):
    expr = ibis.ifelse(alltypes["int_col"] == 0, 42, -1).cast("int64").name("where_col")
    result = expr.execute()

    expected = pd.Series(
        np.where(df.int_col == 0, 42, -1),
        name="where_col",
        dtype="int64",
    )

    assert_series_equal(result, expected)


def test_select_filter(alltypes, df):
    t = alltypes

    expr = t.select("int_col", "string_col").filter(t.string_col == "4")
    result = expr.execute()

    expected = df.loc[df.string_col == "4", ["int_col", "string_col"]].reset_index(
        drop=True
    )
    assert_frame_equal(result, expected)


def test_select_filter_select(alltypes, df):
    t = alltypes
    expr = t.select("int_col", "string_col").filter(t.string_col == "4").int_col
    result = expr.execute().rename("int_col")

    expected = df.loc[df.string_col == "4", "int_col"].reset_index(drop=True)
    assert_series_equal(result, expected)


def test_interactive(alltypes, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", True)

    expr = alltypes.mutate(
        str_col=_.string_col.replace("1", "").nullif("2"),
        date_col=_.timestamp_col.date(),
        delta_col=lambda t: ibis.now() - t.timestamp_col,
    )

    repr(expr)


def test_correlated_subquery(alltypes):
    expr = alltypes[_.double_col > _.view().double_col]
    assert expr.compile() is not None


def test_uncorrelated_subquery(batting, batting_df):
    subset_batting = batting[batting.yearID <= 2000]
    expr = batting[_.yearID == subset_batting.yearID.max()]["playerID", "yearID"]
    result = expr.execute()

    expected = batting_df[batting_df.yearID == 2000][["playerID", "yearID"]]
    assert_frame_equal(result, expected)


def test_int_column(alltypes):
    expr = alltypes.mutate(x=ibis.literal(1)).x
    result = expr.execute()
    assert expr.type() == dt.int8
    assert result.dtype == np.int8


def test_int_scalar(alltypes):
    expr = alltypes.smallint_col.min()
    result = expr.execute()
    assert expr.type() == dt.int16
    assert result.dtype == np.int16


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "bytes",
        "str",
        "int",
        "float",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "timestamp",
        "date",
        "time",
    ],
)
def test_literal_na(con, dtype):
    expr = ibis.literal(None, type=dtype)
    result = con.execute(expr)
    assert pd.isna(result)


def test_memtable_bool_column(con):
    t = ibis.memtable({"a": [True, False, True]})
    assert_series_equal(con.execute(t.a), pd.Series([True, False, True], name="a"))


def test_memtable_construct(con, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    monkeypatch.setattr(ibis.options, "default_backend", con)

    pa_t = pa.Table.from_pydict(
        {
            "a": list("abc"),
            "b": [1, 2, 3],
            "c": [1.0, 2.0, 3.0],
            "d": [None, "b", None],
        }
    )
    t = ibis.memtable(pa_t)
    assert_frame_equal(t.execute().fillna(pd.NA), pa_t.to_pandas().fillna(pd.NA))


@pytest.mark.xfail(reason="datafusion 38.0.0 introduced a bug")
def test_pivot_wider(diamonds):
    expr = (
        diamonds.group_by(["cut", "color"])
        .agg(carat=_.carat.mean())
        .pivot_wider(
            names_from="cut", values_from="carat", names_sort=True, values_agg="mean"
        )
    )
    df = expr.execute()
    assert set(df.columns) == {"color"} | set(
        diamonds[["cut"]].distinct().cut.execute()
    )
    assert len(df) == diamonds.color.nunique().execute()


@pytest.mark.parametrize(
    ("slc", "expected_count_fn"),
    [
        ###################
        ### NONE/ZERO start
        # no stop
        param(slice(None, 0), lambda _: 0, id="[:0]"),
        param(slice(None, None), lambda t: t.count().to_pandas(), id="[:]"),
        param(slice(0, 0), lambda _: 0, id="[0:0]"),
        param(slice(0, None), lambda t: t.count().to_pandas(), id="[0:]"),
        # positive stop
        param(slice(None, 2), lambda _: 2, id="[:2]"),
        param(slice(0, 2), lambda _: 2, id="[0:2]"),
        ##################
        ### NEGATIVE start
        # zero stop
        param(slice(-3, 0), lambda _: 0, id="[-3:0]"),
        # negative stop
        param(slice(-3, -3), lambda _: 0, id="[-3:-3]"),
        param(slice(-3, -4), lambda _: 0, id="[-3:-4]"),
        param(slice(-3, -5), lambda _: 0, id="[-3:-5]"),
        ##################
        ### POSITIVE start
        # no stop
        param(slice(3, 0), lambda _: 0, id="[3:0]"),
        param(
            slice(3, None),
            lambda t: t.count().to_pandas() - 3,
            id="[3:]",
        ),
        # positive stop
        param(slice(3, 2), lambda _: 0, id="[3:2]"),
        param(
            slice(3, 4),
            lambda _: 1,
            id="[3:4]",
        ),
    ],
)
def test_static_table_slice(slc, expected_count_fn, functional_alltypes):
    t = functional_alltypes

    rows = t[slc]
    count = rows.count().to_pandas()

    expected_count = expected_count_fn(t)
    assert count == expected_count


def test_sample(functional_alltypes):
    t = functional_alltypes.filter(_.int_col >= 2)

    total_rows = t.count().execute()
    empty = t.limit(1).execute().iloc[:0]

    df = t.sample(0.1, method="row").execute()
    assert len(df) <= total_rows
    assert_frame_equal(empty, df.iloc[:0])

    df = t.sample(0.1, method="block").execute()
    assert len(df) <= total_rows
    assert_frame_equal(empty, df.iloc[:0])


def test_sample_memtable(con):
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    res = con.execute(ibis.memtable(df).sample(0.5))
    assert len(res) <= 4
    assert_frame_equal(res.iloc[:0], df.iloc[:0])


def test_hexdigest(alltypes):
    h1 = alltypes.order_by("id").string_col.hexdigest().execute(limit=10)
    df = alltypes.order_by("id").execute(limit=10)

    import hashlib

    def hash_256(col):
        return hashlib.sha256(col.encode()).hexdigest()

    h2 = df["string_col"].apply(hash_256).rename("HexDigest(string_col)")

    assert_series_equal(h1, h2)


def test_typeof(con):
    # Other tests also use the typeof operation, but only this test has this operation required.
    expr = ibis.literal(1).typeof()
    result = con.execute(expr)

    assert result is not None
