from __future__ import annotations

import datetime

import pandas as pd
import pytest
from pytest import param

import xorq.api as xo
from xorq.tests.util import assert_frame_equal, assert_series_equal


@pytest.fixture(scope="session")
def con():
    return xo.connect()


@pytest.fixture(scope="session")
def alltypes(con, parquet_dir):
    return con.read_parquet(
        parquet_dir / "functional_alltypes.parquet", table_name="alltypes"
    )


@pytest.mark.parametrize(
    "input_ts",
    [
        param("2023-01-15 14:30:45", id="timestamp"),
        param("2023-12-31 23:59:59", id="year_end"),
    ],
)
def test_date_extraction(compiler, con, input_ts):
    ts = xo.timestamp(input_ts)
    expr = ts.date()

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)
    expected = con.execute(expr)

    assert result == expected
    assert isinstance(result, datetime.date)


def test_date_extraction_column(compiler, con, alltypes):
    expr = alltypes.timestamp_col.date()

    yaml_dict = compiler.to_yaml(expr)
    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    result = roundtrip_expr.execute()
    expected = expr.execute()

    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "input_ts",
    [
        param("2023-01-15 14:30:45", id="afternoon"),
        param("2023-12-31 00:00:01", id="midnight"),
    ],
)
def test_time_extraction(compiler, con, input_ts):
    ts = xo.timestamp(input_ts)
    expr = ts.time()

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)
    expected = con.execute(expr)

    assert result == expected
    assert isinstance(result, datetime.time)


def test_time_extraction_column(compiler, con, alltypes):
    expr = alltypes.timestamp_col.time()

    yaml_dict = compiler.to_yaml(expr)
    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    result = roundtrip_expr.execute()
    expected = expr.execute()

    assert_series_equal(result, expected)


def test_timestamp_now(compiler, con):
    expr = xo.now()

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)

    assert isinstance(result, datetime.datetime)


def test_date_now(compiler, con):
    expr = xo.today()

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)
    expected = datetime.date.today()

    assert result == expected
    assert isinstance(result, datetime.date)


@pytest.mark.parametrize(
    ("date_str", "interval_days", "expected_str"),
    [
        param("2023-01-15", 10, "2023-01-25", id="add_days"),
        param("2023-12-25", 7, "2024-01-01", id="year_boundary"),
        param("2023-02-28", 1, "2023-03-01", id="month_boundary"),
    ],
)
def test_date_add(compiler, con, date_str, interval_days, expected_str):
    date = xo.literal(date_str).cast("date")
    interval = xo.interval(days=interval_days)
    expr = date + interval

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)
    expected = datetime.date.fromisoformat(expected_str)

    assert result == expected


def test_date_add_column(compiler, con, alltypes):
    date_col = alltypes.timestamp_col.date()
    interval = xo.interval(days=5)
    expr = date_col + interval

    yaml_dict = compiler.to_yaml(expr)
    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    result = roundtrip_expr.execute()
    expected = expr.execute()

    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("date_str", "interval_days", "expected_str"),
    [
        param("2023-01-25", 10, "2023-01-15", id="sub_days"),
        param("2024-01-01", 7, "2023-12-25", id="year_boundary"),
        param("2023-03-01", 1, "2023-02-28", id="month_boundary"),
    ],
)
def test_date_sub(compiler, con, date_str, interval_days, expected_str):
    date = xo.literal(date_str).cast("date")
    interval = xo.interval(days=interval_days)
    expr = date - interval

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)
    expected = datetime.date.fromisoformat(expected_str)

    assert result == expected


def test_date_sub_column(compiler, con, alltypes):
    date_col = alltypes.timestamp_col.date()
    interval = xo.interval(days=3)
    expr = date_col - interval

    yaml_dict = compiler.to_yaml(expr)
    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    result = roundtrip_expr.execute()
    expected = expr.execute()

    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("date1_str", "date2_str"),
    [
        param("2023-01-25", "2023-01-15", id="same_month"),
        param("2024-01-01", "2023-12-25", id="year_boundary"),
        param("2023-03-01", "2023-02-28", id="month_boundary"),
    ],
)
def test_date_diff(compiler, con, date1_str, date2_str):
    date1 = xo.date(date1_str)
    date2 = xo.date(date2_str)
    expr = date1 - date2

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    assert expr.equals(roundtrip_expr)


def test_date_diff_column(compiler, con, alltypes):
    date1 = alltypes.timestamp_col.date()
    date2 = xo.literal("2010-01-01").cast("date")
    expr = date1 - date2

    yaml_dict = compiler.to_yaml(expr)
    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    assert expr.equals(roundtrip_expr)


@pytest.mark.parametrize(
    ("date1_str", "date2_str", "unit", "expected"),
    [
        param("2023-10-01", "2023-09-01", "month", 1, id="month_diff"),
        param("2023-01-10", "2023-01-01", "day", 9, id="day_diff"),
        param("2024-01-01", "2023-01-01", "year", 1, id="year_diff"),
    ],
)
def test_date_delta(compiler, con, date1_str, date2_str, unit, expected):
    date1 = xo.literal(date1_str).cast("date")
    date2 = xo.literal(date2_str).cast("date")
    expr = date1.delta(date2, unit)

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)

    assert result == expected


def test_date_delta_column(compiler, con, alltypes):
    date1 = alltypes.timestamp_col.date()
    date2 = xo.literal("2010-01-01").cast("date")
    expr = date1.delta(date2, "day")

    yaml_dict = compiler.to_yaml(expr)
    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    result = roundtrip_expr.execute()
    expected = expr.execute()

    assert_series_equal(result, expected)


# Test TimestampDelta operation
@pytest.mark.parametrize(
    ("ts1_str", "ts2_str", "unit", "expected"),
    [
        param("2023-01-01 12:00:00", "2023-01-01 10:00:00", "hour", 2, id="hour_diff"),
        param(
            "2023-01-01 10:30:00", "2023-01-01 10:00:00", "minute", 30, id="minute_diff"
        ),
        param("2023-01-02 00:00:00", "2023-01-01 00:00:00", "day", 1, id="day_diff"),
    ],
)
def test_timestamp_delta(compiler, con, ts1_str, ts2_str, unit, expected):
    ts1 = xo.timestamp(ts1_str)
    ts2 = xo.timestamp(ts2_str)
    expr = ts1.delta(ts2, unit)

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)

    assert result == expected


def test_timestamp_delta_column(compiler, con, alltypes):
    ts1 = alltypes.timestamp_col
    ts2 = xo.timestamp("2010-01-01 00:00:00")
    expr = ts1.delta(ts2, "hour")

    yaml_dict = compiler.to_yaml(expr)
    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    result = roundtrip_expr.execute()
    expected = expr.execute()

    assert_series_equal(result, expected)


# Test TimeDelta operation
@pytest.mark.parametrize(
    ("time1_str", "time2_str", "unit", "expected"),
    [
        param("14:30:00", "12:00:00", "hour", 2, id="hour_diff"),
        param("14:30:00", "14:00:00", "minute", 30, id="minute_diff"),
        param("14:00:30", "14:00:00", "second", 30, id="second_diff"),
    ],
)
def test_time_delta(compiler, con, time1_str, time2_str, unit, expected):
    time1 = xo.time(time1_str)
    time2 = xo.time(time2_str)
    expr = time1.delta(time2, unit)

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)

    assert result == expected


def test_date_extraction_null(compiler, con):
    expr = xo.null(xo.dtype("timestamp")).date()

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)

    assert pd.isna(result)


def test_time_extraction_null(compiler, con):
    expr = xo.null(xo.dtype("timestamp")).time()

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)

    assert pd.isna(result)


def test_date_add_null(compiler, con):
    date = xo.null(xo.dtype("date"))
    interval = xo.interval(days=5)
    expr = date + interval

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)

    assert pd.isna(result)


def test_date_diff_null(compiler, con):
    date1 = xo.null(xo.dtype("date"))
    date2 = xo.literal("2023-01-01").cast("date")
    expr = date1 - date2

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    assert expr.equals(roundtrip_expr)


def test_date_delta_null(compiler, con):
    date1 = xo.null(xo.dtype("date"))
    date2 = xo.literal("2023-01-01").cast("date")
    expr = date1.delta(date2, "day")

    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)

    result = con.execute(roundtrip_expr)

    assert pd.isna(result)


# Test with table operations
def test_date_operations_in_projection(compiler, con, alltypes):
    expr = alltypes.select(
        id=alltypes.id,
        date=alltypes.timestamp_col.date(),
        time=alltypes.timestamp_col.time(),
        date_plus_5=alltypes.timestamp_col.date() + xo.interval(days=5),
    ).limit(5)

    yaml_dict = compiler.to_yaml(expr)
    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    result = roundtrip_expr.execute().reset_index(drop=True)
    expected = expr.execute().reset_index(drop=True)

    assert_frame_equal(result, expected)


def test_date_operations_in_filter(compiler, con, alltypes):
    target_date = xo.literal("2010-03-01").cast("date")
    expr = (
        alltypes.filter(alltypes.timestamp_col.date() >= target_date)
        .select("id", "timestamp_col")
        .limit(5)
    )

    yaml_dict = compiler.to_yaml(expr)
    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    result = roundtrip_expr.execute().reset_index(drop=True)
    expected = expr.execute().reset_index(drop=True)

    assert_frame_equal(result, expected)


def test_date_operations_in_aggregation(compiler, con, alltypes):
    expr = (
        alltypes.order_by(alltypes.id, alltypes.int_col)
        .limit(20)
        .group_by(date=alltypes.timestamp_col.date())
        .aggregate(count=alltypes.id.count())
    )

    # Test YAML round-trip
    yaml_dict = compiler.to_yaml(expr)
    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)

    actual, expected = (
        e.execute().sort_values(e.columns).reset_index(drop=True)
        for e in (roundtrip_expr, expr)
    )

    assert_frame_equal(actual, expected, check_dtype=False)
