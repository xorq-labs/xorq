import datetime
import decimal

import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.common.exceptions import UnsupportedOperationError
from xorq.expr.datatypes import LargeString


@pytest.mark.parametrize(
    ("typ", "arrow_type", "expected"),
    [
        pytest.param(dt.int8, "Int8", [1, None, None], id="int8"),
        pytest.param(dt.uint8, "UInt8", [1, None, None], id="uint8"),
        pytest.param(dt.uint16, "UInt16", [1, 300, None], id="uint16"),
        pytest.param(dt.uint32, "UInt32", [1, 300, None], id="uint32"),
        pytest.param(dt.uint64, "UInt64", [1, 300, None], id="uint64"),
    ],
)
def test_try_cast_narrow_int(typ, arrow_type, expected):
    t = xo.memtable({"s": ["1", "300", "x"]})
    cast = t.s.try_cast(typ)
    result = t.select(value=cast, computed_type=cast.typeof()).to_pyarrow()

    assert result["value"].to_pylist() == expected
    assert set(result["computed_type"].to_pylist()) == {arrow_type}


@pytest.mark.parametrize(
    ("make_value", "typ", "arrow_type", "expected"),
    [
        pytest.param(
            lambda t: xo.array([t.a]),
            "array<int8>",
            "List(Int8)",
            [[1], [2]],
            id="array-int8",
        ),
        pytest.param(
            lambda t: xo.array([t.a]),
            "array<uint8>",
            "List(UInt8)",
            [[1], [2]],
            id="array-uint8",
        ),
        pytest.param(
            lambda t: xo.struct({"x": t.a, "y": t.s}),
            "struct<x: uint16, y: string>",
            'Struct("x": UInt16, "y": Utf8)',
            [{"x": 1, "y": "p"}, {"x": 2, "y": "q"}],
            id="struct-uint16",
        ),
        pytest.param(
            lambda t: xo.map(xo.array(["k"]), xo.array([t.a])),
            "map<string, uint32>",
            'Map("entries": non-null Struct("key": non-null Utf8, "value": UInt32), unsorted)',
            [[("k", 1)], [("k", 2)]],
            id="map-uint32",
        ),
        pytest.param(
            lambda t: xo.array([xo.struct({"x": t.a})]),
            "array<struct<x: uint64>>",
            'List(Struct("x": UInt64))',
            [[{"x": 1}], [{"x": 2}]],
            id="array-struct-uint64",
        ),
        pytest.param(
            lambda t: xo.struct({"x": t.a, "i": xo.interval(days=1)}),
            "struct<x: int8, i: interval('D')>",
            'Struct("x": Int8, "i": Interval(MonthDayNano))',
            [{"x": 1, "i": (0, 1, 0)}, {"x": 2, "i": (0, 1, 0)}],
            id="struct-int8-interval",
        ),
        pytest.param(
            lambda t: xo.struct({"x": t.a, "b": t.s.cast("binary")}),
            dt.Struct({"x": dt.uint8, "b": dt.BinaryView()}),
            'Struct("x": UInt8, "b": BinaryView)',
            [{"x": 1, "b": b"p"}, {"x": 2, "b": b"q"}],
            id="struct-uint8-binaryview",
        ),
    ],
)
def test_nested_narrow_int_cast(make_value, typ, arrow_type, expected):
    t = xo.memtable({"a": [1, 2], "s": ["p", "q"]})
    cast = make_value(t).cast(typ)
    result = t.select(value=cast, computed_type=cast.typeof()).to_pyarrow()

    assert result["value"].to_pylist() == expected
    assert set(result["computed_type"].to_pylist()) == {arrow_type}


@pytest.mark.parametrize(
    ("typ", "arrow_type"),
    [
        pytest.param("int8", "Int8", id="int8"),
        pytest.param("uint8", "UInt8", id="uint8"),
        pytest.param("array<uint16>", "List(UInt16)", id="array-uint16"),
    ],
)
def test_narrow_int_null_literal(typ, arrow_type):
    t = xo.memtable({"a": [1]})
    null = xo.literal(None, typ)
    result = t.select(value=null, computed_type=null.typeof()).to_pyarrow()

    assert result["value"].to_pylist() == [None]
    assert result["computed_type"].to_pylist() == [arrow_type]


@pytest.mark.parametrize(
    ("make_value", "typ", "arrow_type", "expected"),
    [
        pytest.param(
            lambda t: xo.map(xo.array([t.a]), xo.array([t.s])),
            "map<uint8, string>",
            'Map("entries": non-null Struct("key": non-null UInt8, "value": Utf8), unsorted)',
            [[(1, "p")], [(2, "q")]],
            id="map-uint8-key",
        ),
        pytest.param(
            lambda t: xo.array([xo.array([t.a])]),
            "array<array<uint8>>",
            "List(List(UInt8))",
            [[[1]], [[2]]],
            id="array-array-uint8",
        ),
        pytest.param(
            lambda t: xo.struct({"o": xo.struct({"x": t.a})}),
            "struct<o: struct<x: int8>>",
            'Struct("o": Struct("x": Int8))',
            [{"o": {"x": 1}}, {"o": {"x": 2}}],
            id="struct-struct-int8",
        ),
    ],
)
def test_deeply_nested_narrow_int_cast(make_value, typ, arrow_type, expected):
    t = xo.memtable({"a": [1, 2], "s": ["p", "q"]})
    cast = make_value(t).cast(typ)
    result = t.select(value=cast, computed_type=cast.typeof()).to_pyarrow()

    assert result["value"].to_pylist() == expected
    assert set(result["computed_type"].to_pylist()) == {arrow_type}


_ts = [datetime.datetime(2024, 1, 1, 12), datetime.datetime(2024, 1, 2, 12)]


@pytest.mark.parametrize(
    ("column", "sibling", "arrow_type", "expected"),
    [
        pytest.param("a", dt.int16, "Int16", [1, 2], id="int16"),
        pytest.param("a", dt.int32, "Int32", [1, 2], id="int32"),
        pytest.param("a", dt.int64, "Int64", [1, 2], id="int64"),
        pytest.param("f", dt.float32, "Float32", [1.5, 2.5], id="float32"),
        pytest.param("f", dt.float64, "Float64", [1.5, 2.5], id="float64"),
        pytest.param("b", dt.boolean, "Boolean", [False, True], id="boolean"),
        pytest.param("s", dt.string, "Utf8", ["p", "q"], id="string"),
        pytest.param("s", LargeString(), "LargeUtf8", ["p", "q"], id="largestring"),
        pytest.param(
            "d",
            dt.date,
            "Date32",
            [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
            id="date",
        ),
        pytest.param("tm", dt.time, "Time64(ns)", [datetime.time(12)] * 2, id="time"),
        pytest.param("ts", dt.timestamp, "Timestamp(µs)", _ts, id="timestamp"),
        pytest.param(
            "ts",
            dt.Timestamp(timezone="UTC", scale=3),
            'Timestamp(ms, "UTC")',
            [v.replace(tzinfo=datetime.timezone.utc) for v in _ts],
            id="timestamp-utc-ms",
        ),
        pytest.param(
            "f",
            dt.Decimal(10, 2),
            "Decimal128(10, 2)",
            [decimal.Decimal("1.50"), decimal.Decimal("2.50")],
            id="decimal",
        ),
    ],
)
def test_nested_narrow_int_cast_keeps_sibling_type(
    column, sibling, arrow_type, expected
):
    t = xo.memtable(
        {
            "a": [1, 2],
            "f": [1.5, 2.5],
            "b": [False, True],
            "s": ["p", "q"],
            "d": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
            "ts": _ts,
        }
    ).mutate(tm=lambda t: t.ts.time())
    cast = xo.struct({"x": t.a, "y": t[column]}).cast(
        dt.Struct({"x": dt.uint8, "y": sibling})
    )
    result = t.select(value=cast, computed_type=cast.typeof()).to_pyarrow()

    assert [row["y"] for row in result["value"].to_pylist()] == expected
    assert set(result["computed_type"].to_pylist()) == {
        f'Struct("x": UInt8, "y": {arrow_type})'
    }


@pytest.mark.parametrize(
    ("make_value", "typ", "expected"),
    [
        pytest.param(
            lambda t: xo.array([t.a * 200]),
            "array<uint8>",
            [[200], [None]],
            id="array-uint8",
        ),
        pytest.param(
            lambda t: xo.struct({"x": t.a * 100}),
            "struct<x: int8>",
            [{"x": 100}, {"x": None}],
            id="struct-int8",
        ),
    ],
)
def test_nested_try_cast_out_of_range_is_null(make_value, typ, expected):
    t = xo.memtable({"a": [1, 2]})

    assert make_value(t).try_cast(typ).to_pyarrow().to_pylist() == expected


def test_nested_narrow_int_cast_out_of_range_raises():
    t = xo.memtable({"a": [1, 2]})

    with pytest.raises(ValueError, match="Can't cast value 400 to type UInt8"):
        xo.array([t.a * 200]).cast("array<uint8>").execute()


def test_struct_field_name_with_double_quote_raises():
    t = xo.memtable({"a": [1]})
    cast = xo.struct({'a"b': t.a}).cast(dt.Struct({'a"b': dt.uint8}))

    with pytest.raises(UnsupportedOperationError, match="struct field named"):
        xo.to_sql(cast)


@pytest.mark.parametrize(
    ("typ", "uses_arrow_cast"),
    [
        pytest.param("array<int16>", False, id="array-int16"),
        pytest.param("struct<x: int32, y: string>", False, id="struct-int32"),
        pytest.param("array<uint8>", True, id="array-uint8"),
    ],
)
def test_only_misspelled_casts_use_arrow_cast(typ, uses_arrow_cast):
    t = xo.memtable({"a": [1]})
    value = (
        xo.struct({"x": t.a, "y": "s"}) if typ.startswith("struct") else xo.array([t.a])
    )
    sql = xo.to_sql(t.select(v=value.cast(typ))).upper()

    assert ("ARROW_CAST" in sql) == uses_arrow_cast


@pytest.mark.parametrize(
    ("typ", "value", "arrow_type", "expected"),
    [
        pytest.param(
            "decimal(10, 2)",
            "1.5",
            "Decimal128(10, 2)",
            [decimal.Decimal("1.50"), None],
            id="decimal",
        ),
        pytest.param(
            "timestamp",
            "2024-01-02 03:04:05",
            "Timestamp(µs)",
            [datetime.datetime(2024, 1, 2, 3, 4, 5), None],
            id="timestamp",
        ),
        pytest.param(
            LargeString(), "1.5", "LargeUtf8", ["1.5", "x"], id="large-string"
        ),
    ],
)
def test_try_cast_arrow_types(
    typ: str | dt.DataType, value: str, arrow_type: str, expected: list
) -> None:
    t = xo.memtable({"s": [value, "x"]})
    cast = t.s.try_cast(typ)
    result = t.select(value=cast, computed_type=cast.typeof()).to_pyarrow()

    assert result["value"].to_pylist() == expected
    assert set(result["computed_type"].to_pylist()) == {arrow_type}


@pytest.mark.parametrize(
    ("typ", "expected"),
    [
        pytest.param(
            dt.Array(dt.Decimal(10, 2)),
            [decimal.Decimal("1.50"), None],
            id="array-decimal",
        ),
        pytest.param(dt.Array(LargeString()), ["1.5", "x"], id="array-large-string"),
    ],
)
def test_try_cast_arrow_types_nested(typ: dt.DataType, expected: list) -> None:
    t = xo.memtable({"a": [["1.5", "x"]]})
    result = t.select(v=t.a.try_cast(typ)).to_pyarrow()

    assert result["v"].to_pylist() == [expected]
