"""Test datetime operation translations."""

from datetime import datetime

import pytest

import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations.temporal as tm
from xorq.ibis_yaml.common import TranslationContext, translate_to_yaml


def test_date_extract(compiler):
    dt_expr = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))

    year = dt_expr.year()
    year_yaml = compiler.to_yaml(year)
    expression = year_yaml["expression"]
    assert expression["op"] == "ExtractYear"
    assert expression["arg"]["value"] == "2024-03-14T15:09:26"
    assert year.type().name == "Int32"
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
    assert hour_expression["arg"]["value"] == "2024-03-14T15:09:26"
    assert hour.type().name == "Int32"
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
    assert plus_day.type().name == "Timestamp"
    assert expression["right"]["type"].startswith("interval")
    roundtrip_plus = compiler.from_yaml(yaml_dict)
    assert roundtrip_plus.equals(plus_day)

    minus_day = ts - delta
    yaml_dict = compiler.to_yaml(minus_day)
    expression = yaml_dict["expression"]
    assert expression["op"] == "TimestampSub"
    assert minus_day.type().name == "Timestamp"
    assert expression["right"]["type"].startswith("interval")
    roundtrip_minus = compiler.from_yaml(yaml_dict)
    assert roundtrip_minus.equals(minus_day)


def test_timestamp_diff(compiler):
    ts1 = ibis.literal(datetime(2024, 3, 14))
    ts2 = ibis.literal(datetime(2024, 3, 15))
    diff = ts2 - ts1
    yaml_dict = compiler.to_yaml(diff)
    expression = yaml_dict["expression"]
    assert expression["op"] == "TimestampDiff"
    assert diff.type().name == "Interval"
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(diff)


def test_temporal_unit_yaml(compiler):
    interval_date = ibis.literal(5, type=dt.Interval(unit=tm.DateUnit("D")))
    yaml_date = compiler.to_yaml(interval_date)
    expression_date = yaml_date["expression"]
    # interval dtype + unit inline in the canonical ibis string, e.g. "interval('D')"
    assert expression_date["type"] == str(interval_date.type())
    assert expression_date["type"].startswith("interval")
    assert "'D'" in expression_date["type"]
    roundtrip_date = compiler.from_yaml(yaml_date)
    assert roundtrip_date.equals(interval_date)

    interval_time = ibis.literal(10, type=dt.Interval(unit=tm.TimeUnit("h")))
    yaml_time = compiler.to_yaml(interval_time)
    expression_time = yaml_time["expression"]
    assert expression_time["type"] == str(interval_time.type())
    assert "'h'" in expression_time["type"]
    roundtrip_time = compiler.from_yaml(yaml_time)
    assert roundtrip_time.equals(interval_time)


def test_date_truncate(compiler, t):
    """Test that DateTruncate with DateUnit serializes and roundtrips correctly."""
    truncated = t.e.truncate("M")
    yaml_dict = compiler.to_yaml(truncated)
    expression = yaml_dict["expression"]
    assert expression["op"] == "DateTruncate"
    assert expression["unit"]["op"] == "IntervalUnit"
    assert expression["unit"]["name"] == "DateUnit"
    assert expression["unit"]["value"] == "M"
    roundtrip = compiler.from_yaml(yaml_dict)
    assert roundtrip.equals(truncated)


def test_time_truncate(compiler):
    """Test that TimeTruncate with TimeUnit serializes and roundtrips correctly."""
    t = ibis.table({"t": "time"}, name="time_table")
    truncated = t.t.truncate("h")
    yaml_dict = compiler.to_yaml(truncated)
    expression = yaml_dict["expression"]
    assert expression["op"] == "TimeTruncate"
    assert expression["unit"]["op"] == "IntervalUnit"
    assert expression["unit"]["name"] == "TimeUnit"
    assert expression["unit"]["value"] == "h"
    roundtrip = compiler.from_yaml(yaml_dict)
    assert roundtrip.equals(truncated)


@pytest.mark.parametrize(
    "unit_cls,unit_value,expected_name",
    [
        (tm.DateUnit, "Y", "DateUnit"),
        (tm.DateUnit, "M", "DateUnit"),
        (tm.DateUnit, "D", "DateUnit"),
        (tm.TimeUnit, "h", "TimeUnit"),
        (tm.TimeUnit, "m", "TimeUnit"),
        (tm.TimeUnit, "s", "TimeUnit"),
        (tm.TimestampUnit, "s", "TimestampUnit"),
        (tm.TimestampUnit, "ms", "TimestampUnit"),
        (tm.IntervalUnit, "Y", "DateUnit"),
        (tm.IntervalUnit, "h", "TimeUnit"),
    ],
)
def test_temporal_unit_direct_serialization(
    compiler, unit_cls, unit_value, expected_name
):
    """Test that DateUnit, TimeUnit, and TimestampUnit enums serialize directly."""
    ctx = TranslationContext()
    unit = unit_cls(unit_value)
    result = translate_to_yaml(unit, ctx)
    assert result["op"] == "IntervalUnit"
    assert result["name"] == expected_name
    assert result["value"] == unit_value
