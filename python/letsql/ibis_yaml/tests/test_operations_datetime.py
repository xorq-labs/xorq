"""Test datetime operation translations."""

from datetime import datetime

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations.temporal as tm


def test_date_extract(compiler):
    dt_expr = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))

    year = dt_expr.year()
    year_yaml = compiler.compile_to_yaml(year)
    assert year_yaml["op"] == "ExtractYear"
    assert year_yaml["args"][0]["value"] == "2024-03-14T15:09:26"
    assert year_yaml["type"]["name"] == "Int32"
    roundtrip_year = compiler.compile_from_yaml(year_yaml)
    assert roundtrip_year.equals(year)

    month = dt_expr.month()
    month_yaml = compiler.compile_to_yaml(month)
    assert month_yaml["op"] == "ExtractMonth"
    roundtrip_month = compiler.compile_from_yaml(month_yaml)
    assert roundtrip_month.equals(month)

    day = dt_expr.day()
    day_yaml = compiler.compile_to_yaml(day)
    assert day_yaml["op"] == "ExtractDay"
    roundtrip_day = compiler.compile_from_yaml(day_yaml)
    assert roundtrip_day.equals(day)


def test_time_extract(compiler):
    dt_expr = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))

    hour = dt_expr.hour()
    hour_yaml = compiler.compile_to_yaml(hour)
    assert hour_yaml["op"] == "ExtractHour"
    assert hour_yaml["args"][0]["value"] == "2024-03-14T15:09:26"
    assert hour_yaml["type"]["name"] == "Int32"
    roundtrip_hour = compiler.compile_from_yaml(hour_yaml)
    assert roundtrip_hour.equals(hour)

    minute = dt_expr.minute()
    minute_yaml = compiler.compile_to_yaml(minute)
    assert minute_yaml["op"] == "ExtractMinute"
    roundtrip_minute = compiler.compile_from_yaml(minute_yaml)
    assert roundtrip_minute.equals(minute)

    second = dt_expr.second()
    second_yaml = compiler.compile_to_yaml(second)
    assert second_yaml["op"] == "ExtractSecond"
    roundtrip_second = compiler.compile_from_yaml(second_yaml)
    assert roundtrip_second.equals(second)


def test_timestamp_arithmetic(compiler):
    ts = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))
    delta = ibis.interval(days=1)

    plus_day = ts + delta
    yaml_dict = compiler.compile_to_yaml(plus_day)
    assert yaml_dict["op"] == "TimestampAdd"
    assert yaml_dict["type"]["name"] == "Timestamp"
    assert yaml_dict["args"][1]["type"]["name"] == "Interval"
    roundtrip_plus = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_plus.equals(plus_day)

    minus_day = ts - delta
    yaml_dict = compiler.compile_to_yaml(minus_day)
    assert yaml_dict["op"] == "TimestampSub"
    assert yaml_dict["type"]["name"] == "Timestamp"
    assert yaml_dict["args"][1]["type"]["name"] == "Interval"
    roundtrip_minus = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_minus.equals(minus_day)


def test_timestamp_diff(compiler):
    ts1 = ibis.literal(datetime(2024, 3, 14))
    ts2 = ibis.literal(datetime(2024, 3, 15))
    diff = ts2 - ts1
    yaml_dict = compiler.compile_to_yaml(diff)
    assert yaml_dict["op"] == "TimestampDiff"
    assert yaml_dict["type"]["name"] == "Interval"
    roundtrip_expr = compiler.compile_from_yaml(yaml_dict)
    assert roundtrip_expr.equals(diff)


def test_temporal_unit_yaml(compiler):
    interval_date = ibis.literal(5, type=dt.Interval(unit=tm.DateUnit("D")))
    yaml_date = compiler.compile_to_yaml(interval_date)
    assert yaml_date["type"]["name"] == "Interval"
    assert yaml_date["type"]["unit"]["name"] == "DateUnit"
    assert yaml_date["type"]["unit"]["value"] == "D"
    roundtrip_date = compiler.compile_from_yaml(yaml_date)
    assert roundtrip_date.equals(interval_date)

    interval_time = ibis.literal(10, type=dt.Interval(unit=tm.TimeUnit("h")))
    yaml_time = compiler.compile_to_yaml(interval_time)
    assert yaml_time["type"]["name"] == "Interval"
    assert yaml_time["type"]["unit"]["name"] == "TimeUnit"
    assert yaml_time["type"]["unit"]["value"] == "h"
    roundtrip_time = compiler.compile_from_yaml(yaml_time)
    assert roundtrip_time.equals(interval_time)
