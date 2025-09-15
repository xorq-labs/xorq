"""Test datetime operation translations."""

from datetime import datetime

import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations.temporal as tm


def test_date_extract(compiler):
    dt_expr = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))

    year = dt_expr.year()
    year_yaml = compiler.to_yaml(year)
    expression = year_yaml["expression"]
    assert expression["op"] == "ExtractYear"
    assert expression["args"][0]["value"] == "2024-03-14T15:09:26"
    assert expression["type"]["type"] == "Int32"
    roundtrip_year = compiler.from_yaml(year_yaml)
    assert roundtrip_year.equals(year)

    month = dt_expr.month()
    month_yaml = compiler.to_yaml(month)
    expression = month_yaml["expression"]
    assert expression["op"] == "ExtractMonth"
    roundtrip_month = compiler.from_yaml(month_yaml)
    assert roundtrip_month.equals(month)

    day = dt_expr.day()
    day_yaml = compiler.to_yaml(day)
    expression = day_yaml["expression"]
    assert expression["op"] == "ExtractDay"
    roundtrip_day = compiler.from_yaml(day_yaml)
    assert roundtrip_day.equals(day)


def test_time_extract(compiler):
    dt_expr = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))

    hour = dt_expr.hour()
    hour_yaml = compiler.to_yaml(hour)
    hour_expression = hour_yaml["expression"]
    assert hour_expression["op"] == "ExtractHour"
    assert hour_expression["args"][0]["value"] == "2024-03-14T15:09:26"
    assert hour_expression["type"]["type"] == "Int32"
    roundtrip_hour = compiler.from_yaml(hour_yaml)
    assert roundtrip_hour.equals(hour)

    minute = dt_expr.minute()
    minute_yaml = compiler.to_yaml(minute)
    minute_expression = minute_yaml["expression"]
    assert minute_expression["op"] == "ExtractMinute"
    roundtrip_minute = compiler.from_yaml(minute_yaml)
    assert roundtrip_minute.equals(minute)

    second = dt_expr.second()
    second_yaml = compiler.to_yaml(second)
    second_expression = second_yaml["expression"]
    assert second_expression["op"] == "ExtractSecond"
    roundtrip_second = compiler.from_yaml(second_yaml)
    assert roundtrip_second.equals(second)


def test_timestamp_arithmetic(compiler):
    ts = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))
    delta = ibis.interval(days=1)

    plus_day = ts + delta
    yaml_dict = compiler.to_yaml(plus_day)
    expression = yaml_dict["expression"]
    assert expression["op"] == "TimestampAdd"
    assert expression["type"]["type"] == "Timestamp"
    assert expression["args"][1]["type"]["type"] == "Interval"
    roundtrip_plus = compiler.from_yaml(yaml_dict)
    assert roundtrip_plus.equals(plus_day)

    minus_day = ts - delta
    yaml_dict = compiler.to_yaml(minus_day)
    expression = yaml_dict["expression"]
    assert expression["op"] == "TimestampSub"
    assert expression["type"]["type"] == "Timestamp"
    assert expression["args"][1]["type"]["type"] == "Interval"
    roundtrip_minus = compiler.from_yaml(yaml_dict)
    assert roundtrip_minus.equals(minus_day)


def test_timestamp_diff(compiler):
    ts1 = ibis.literal(datetime(2024, 3, 14))
    ts2 = ibis.literal(datetime(2024, 3, 15))
    diff = ts2 - ts1
    yaml_dict = compiler.to_yaml(diff)
    expression = yaml_dict["expression"]
    assert expression["op"] == "TimestampDiff"
    assert expression["type"]["type"] == "Interval"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(diff)


def test_temporal_unit_yaml(compiler):
    interval_date = ibis.literal(5, type=dt.Interval(unit=tm.DateUnit("D")))
    yaml_date = compiler.to_yaml(interval_date)
    expression_date = yaml_date["expression"]
    assert expression_date["type"]["type"] == "Interval"
    assert expression_date["type"]["unit"]["name"] == "DateUnit"
    assert expression_date["type"]["unit"]["value"] == "D"
    roundtrip_date = compiler.from_yaml(yaml_date)
    assert roundtrip_date.equals(interval_date)

    interval_time = ibis.literal(10, type=dt.Interval(unit=tm.TimeUnit("h")))
    yaml_time = compiler.to_yaml(interval_time)
    expression_time = yaml_time["expression"]
    assert expression_time["type"]["type"] == "Interval"
    assert expression_time["type"]["unit"]["name"] == "TimeUnit"
    assert expression_time["type"]["unit"]["value"] == "h"
    roundtrip_time = compiler.from_yaml(yaml_time)
    assert roundtrip_time.equals(interval_time)
