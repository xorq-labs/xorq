"""Test datetime operation translations."""

from datetime import datetime

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations.temporal as tm

from xorq.common.utils.ibis_utils import from_ibis


def test_date_extract():
    dt_expr = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))
    year = dt_expr.year()

    assert from_ibis(year) is not None
    assert from_ibis(dt_expr.month()) is not None
    assert from_ibis(dt_expr.day()) is not None


def test_time_extract():
    dt_expr = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))

    assert from_ibis(dt_expr.hour()) is not None
    assert from_ibis(dt_expr.minute()) is not None
    assert from_ibis(dt_expr.second()) is not None


def test_timestamp_arithmetic():
    ts = ibis.literal(datetime(2024, 3, 14, 15, 9, 26))
    delta = ibis.interval(days=1)

    plus_day = ts + delta
    assert from_ibis(plus_day) is not None

    minus_day = ts - delta
    assert from_ibis(minus_day) is not None


def test_timestamp_diff():
    ts1 = ibis.literal(datetime(2024, 3, 14))
    ts2 = ibis.literal(datetime(2024, 3, 15))
    diff = ts2 - ts1

    assert from_ibis(diff) is not None


def test_temporal_unit_yaml():
    interval_date = ibis.literal(5, type=dt.Interval(unit=tm.DateUnit("D")))
    assert from_ibis(interval_date) is not None

    interval_time = ibis.literal(10, type=dt.Interval(unit=tm.TimeUnit("h")))
    assert from_ibis(interval_time) is not None
