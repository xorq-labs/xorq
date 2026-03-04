from __future__ import annotations

import pytest

import xorq.vendor.ibis.expr.datatypes as dt
from xorq.vendor.ibis.common.temporal import IntervalUnit
from xorq.vendor.ibis.expr.datatypes.parse import parse


@pytest.mark.parametrize(
    "text,expected",
    [
        # boolean
        ("boolean", dt.boolean),
        ("bool", dt.boolean),
        # signed integers
        ("int8", dt.int8),
        ("int16", dt.int16),
        ("int32", dt.int32),
        ("int64", dt.int64),
        ("int", dt.int64),
        # unsigned integers
        ("uint8", dt.uint8),
        ("uint16", dt.uint16),
        ("uint32", dt.uint32),
        ("uint64", dt.uint64),
        # floats
        ("float16", dt.float16),
        ("halffloat", dt.float16),
        ("float32", dt.float32),
        ("float64", dt.float64),
        ("double", dt.float64),
        ("float", dt.float64),
        # temporal
        ("date", dt.date),
        ("time", dt.time),
        ("timestamp", dt.Timestamp()),
        ("timestamp('UTC')", dt.Timestamp(timezone="UTC")),
        ("timestamp('UTC', 3)", dt.Timestamp(timezone="UTC", scale=3)),
        ("timestamp(6)", dt.Timestamp(scale=6)),
        # binary / null
        ("binary", dt.binary),
        ("bytes", dt.binary),
        ("null", dt.null),
        # string family — bare and aliases
        ("str", dt.string),
        ("string", dt.string),
        ("!string", dt.String(nullable=False)),
        ("varchar", dt.string),
        ("varchar(100)", dt.string),
        ("!varchar(100)", dt.String(nullable=False)),
        ("char(10)", dt.string),
        # string(N) — length qualifier carries no semantic meaning and is stripped
        ("string(50)", dt.string),
        ("!string(50)", dt.String(nullable=False)),
        ("string(255)", dt.string),
        ("!string(255)", dt.String(nullable=False)),
        # decimal
        ("decimal", dt.Decimal()),
        ("decimal(10, 2)", dt.Decimal(10, 2)),
        ("decimal(38, 9)", dt.Decimal(38, 9)),
        ("bignumeric", dt.Decimal(76, 38)),
        ("bigdecimal(5, 3)", dt.Decimal(5, 3)),
        # interval
        ("interval('s')", dt.Interval(unit=IntervalUnit.SECOND)),
        ("interval('Y')", dt.Interval(unit=IntervalUnit.YEAR)),
        ("interval('D')", dt.Interval(unit=IntervalUnit.DAY)),
        # composite
        ("array<int64>", dt.Array(dt.int64)),
        ("array<string>", dt.Array(dt.string)),
        ("map<string, int64>", dt.Map(dt.string, dt.int64)),
        ("struct<a: int64, b: string>", dt.Struct({"a": dt.int64, "b": dt.string})),
        # extension types
        ("json", dt.JSON(binary=False)),
        ("jsonb", dt.JSON(binary=True)),
        ("uuid", dt.uuid),
        ("inet", dt.inet),
        ("macaddr", dt.macaddr),
        # nullable modifier
        ("!int32", dt.Int32(nullable=False)),
        ("!boolean", dt.Boolean(nullable=False)),
        ("!date", dt.Date(nullable=False)),
        ("!decimal(10, 2)", dt.Decimal(10, 2, nullable=False)),
        ("!array<int64>", dt.Array(dt.int64, nullable=False)),
    ],
)
def test_parse(text, expected):
    assert parse(text) == expected
