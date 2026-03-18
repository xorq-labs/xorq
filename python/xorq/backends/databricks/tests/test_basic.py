from __future__ import annotations

import pandas as pd


def test_execute_returns_dataframe(functional_alltypes):
    result = functional_alltypes.limit(5).execute()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5


def test_column_names(functional_alltypes):
    # read_files() adds _rescued_data; check that the expected columns are present
    expected = {
        "id",
        "bool_col",
        "tinyint_col",
        "smallint_col",
        "int_col",
        "bigint_col",
        "float_col",
        "double_col",
        "date_string_col",
        "string_col",
        "timestamp_col",
        "year",
        "month",
    }
    assert expected.issubset(set(functional_alltypes.columns))


def test_filter(functional_alltypes):
    result = functional_alltypes.filter(functional_alltypes.bool_col).execute()
    assert result["bool_col"].all()


def test_filter_numeric(batting):
    result = batting.filter(batting.HR > 40).execute()
    assert (result["HR"] > 40).all()
    assert len(result) > 0


def test_projection(functional_alltypes):
    cols = ["id", "int_col", "string_col"]
    result = functional_alltypes.select(cols).limit(10).execute()
    assert list(result.columns) == cols
    assert len(result) == 10


def test_mutate(functional_alltypes):
    # Use double_col (float64) to avoid int32/int64 promotion differences
    result = (
        functional_alltypes.mutate(scaled=functional_alltypes.double_col * 2)
        .select("double_col", "scaled")
        .limit(5)
        .execute()
    )
    assert (result["scaled"] == result["double_col"] * 2).all()


def test_aggregation_count(batting):
    result = batting.count().execute()
    assert result > 0


def test_aggregation_groupby(diamonds):
    result = (
        diamonds.group_by("cut")
        .agg(avg_price=diamonds.price.mean(), n=diamonds.price.count())
        .execute()
    )
    assert set(result.columns) == {"cut", "avg_price", "n"}
    assert len(result) == 5  # Ideal, Premium, Very Good, Good, Fair


def test_sort(diamonds):
    result = diamonds.order_by(diamonds.price.desc()).limit(5).execute()
    prices = result["price"].tolist()
    assert prices == sorted(prices, reverse=True)


def test_limit(functional_alltypes):
    result = functional_alltypes.limit(3).execute()
    assert len(result) == 3


def test_distinct(diamonds):
    cuts = diamonds.select("cut").distinct().execute()
    assert len(cuts) == 5


def test_sql_passthrough(con):
    result = con.sql("SELECT 1 AS n").execute()
    assert result["n"].iloc[0] == 1


def test_chained_filter_project(batting):
    filtered = batting.filter(batting.yearID >= 2000).select(
        "playerID", "yearID", "HR", "AB"
    )
    result = filtered.filter(filtered.AB > 100).limit(20).execute()
    assert (result["yearID"] >= 2000).all()
    assert (result["AB"] > 100).all()
    assert len(result) <= 20
