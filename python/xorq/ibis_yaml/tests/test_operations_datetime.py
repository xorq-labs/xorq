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


def test_strftime(compiler):
    dt_expr = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))
    formatted = dt_expr.strftime("%Y-%m-%d")

    yaml_dict = compiler.to_yaml(formatted)
    expression = yaml_dict["expression"]

    assert expression["op"] == "Strftime"
    assert expression["arg"]["value"] == "2024-03-14T15:09:26"
    assert expression["format_str"]["op"] == "Literal"
    assert expression["format_str"]["value"] == "%Y-%m-%d"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(formatted)


def test_date_arithmetic(compiler):
    from datetime import date

    d = ibis.literal(date(2024, 3, 14))
    delta = ibis.interval(days=1)

    # Test DateAdd
    plus_day = d + delta
    yaml_dict = compiler.to_yaml(plus_day)
    expression = yaml_dict["expression"]
    assert expression["op"] == "DateAdd"
    roundtrip_plus = compiler.from_yaml(yaml_dict)
    assert roundtrip_plus.equals(plus_day)

    # Test DateSub
    minus_day = d - delta
    yaml_dict = compiler.to_yaml(minus_day)
    expression = yaml_dict["expression"]
    assert expression["op"] == "DateSub"
    roundtrip_minus = compiler.from_yaml(yaml_dict)
    assert roundtrip_minus.equals(minus_day)


def test_date_diff(compiler):
    from datetime import date

    d1 = ibis.literal(date(2024, 3, 14))
    d2 = ibis.literal(date(2024, 3, 15))
    diff = d2 - d1
    yaml_dict = compiler.to_yaml(diff)
    expression = yaml_dict["expression"]
    assert expression["op"] == "DateDiff"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(diff)


def test_date_from_timestamp(compiler):
    dt_expr = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))
    date_part = dt_expr.date()
    yaml_dict = compiler.to_yaml(date_part)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Date"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(date_part)


def test_time_from_timestamp(compiler):
    dt_expr = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))
    time_part = dt_expr.time()
    yaml_dict = compiler.to_yaml(time_part)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Time"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(time_part)


def test_timestamp_now(compiler):
    now = ibis.now()
    yaml_dict = compiler.to_yaml(now)
    expression = yaml_dict["expression"]
    assert expression["op"] == "TimestampNow"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    # Both should be TimestampNow operations
    assert type(roundtrip_expr.op()).__name__ == "TimestampNow"


def test_date_now(compiler):
    today = ibis.today()
    yaml_dict = compiler.to_yaml(today)
    expression = yaml_dict["expression"]
    assert expression["op"] == "DateNow"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    # Both should be DateNow operations
    assert type(roundtrip_expr.op()).__name__ == "DateNow"


def test_date_delta(compiler):
    from datetime import date

    d1 = ibis.literal(date(2024, 3, 14))
    d2 = ibis.literal(date(2024, 3, 20))
    delta = d2.delta(d1, "day")
    yaml_dict = compiler.to_yaml(delta)
    expression = yaml_dict["expression"]
    assert expression["op"] == "DateDelta"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(delta)


def test_timestamp_delta(compiler):
    ts1 = ibis.literal(datetime(2024, 3, 14, 10, 0, 0))
    ts2 = ibis.literal(datetime(2024, 3, 14, 15, 0, 0))
    delta = ts2.delta(ts1, "hour")
    yaml_dict = compiler.to_yaml(delta)
    expression = yaml_dict["expression"]
    assert expression["op"] == "TimestampDelta"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(delta)


def test_time_delta(compiler):
    from datetime import time

    t1 = ibis.literal(time(10, 0, 0))
    t2 = ibis.literal(time(15, 0, 0))
    delta = t2.delta(t1, "hour")
    yaml_dict = compiler.to_yaml(delta)
    expression = yaml_dict["expression"]
    assert expression["op"] == "TimeDelta"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(delta)
