import ibis
import pytest
from packaging import version

from xorq.common.utils.ibis_utils import from_ibis


@pytest.mark.parametrize(
    "sort_key",
    (
        lambda t: ibis.asc("value"),
        lambda t: t.value.asc(),
        lambda t: ibis.desc("value"),
        lambda t: t.value.desc(),
    ),
)
def test_sort_keys(t, sort_key):
    t = ibis.table({"id": "int32", "value": "float64"}, name="T")
    expr = t.order_by(sort_key(t))
    assert from_ibis(expr) is not None


def test_sort_keys_nulls_first():
    t = ibis.table({"id": "int32", "value": "float64"}, name="T")
    expr = t.order_by(ibis.asc(t.value, nulls_first=True))
    assert from_ibis(expr) is not None


def test_cases():
    t = ibis.table({"int_col": "int32", "string_col": "string"}, name="T")

    args = [(1, "abcd"), (2, "ABCD")]
    kwargs = {"else_": "dabc"}
    if version.parse(ibis.__version__) <= version.parse("9.5.0"):
        args = [[(1, "abcd"), (2, "ABCD")]]
        kwargs = {"default": "dabc"}

    expr = t.int_col.cases(*args, **kwargs)
    assert from_ibis(expr) is not None


def test_isin_positional():
    t = ibis.table({"int_col": "int32", "string_col": "string"}, name="T")
    expr = t.filter(t.int_col.isin([1, 2, 3]))
    assert from_ibis(expr) is not None


def test_notin_positional():
    t = ibis.table({"int_col": "int32", "string_col": "string"}, name="T")
    expr = t.filter(t.int_col.notin([1, 2, 3]))
    assert from_ibis(expr) is not None


@pytest.mark.parametrize(
    "window",
    (
        lambda t: ibis.window(group_by=t.category, order_by=t.id),
        lambda t: ibis.window(
            group_by=t.category, order_by=t.id, preceding=2, following=0
        ),
        lambda t: None,
    ),
)
def test_window_with_rows(window):
    t = ibis.table({"id": "int32", "value": "float64", "category": "string"}, name="T")

    expr = t.value.mean().over(window(t))

    assert from_ibis(expr) is not None


def test_string_as_date():
    s = ibis.literal("2024-01-15")

    if version.parse(ibis.__version__) >= version.parse("11.0.0"):
        expr = s.as_date("%Y-%m-%d")
    else:
        expr = s.to_date("%Y-%m-%d")

    assert from_ibis(expr) is not None


def test_string_as_timestamp():
    s = ibis.literal("2024-01-15 10:30:00")

    if version.parse(ibis.__version__) >= version.parse("11.0.0"):
        expr = s.as_timestamp("%Y-%m-%d %H:%M:%S")
    else:
        expr = s.to_timestamp("%Y-%m-%d %H:%M:%S")

    assert from_ibis(expr) is not None


def test_integer_as_interval():
    i = ibis.literal(5)

    if version.parse(ibis.__version__) >= version.parse("10.0.0"):
        expr = i.as_interval("day")
    else:
        expr = i.to_interval(unit="day")

    assert from_ibis(expr) is not None


def test_integer_as_timestamp():
    i = ibis.literal(1705315800)

    if version.parse(ibis.__version__) >= version.parse("10.0.0"):
        expr = i.as_timestamp("s")
    else:
        expr = i.to_timestamp(unit="s")

    assert from_ibis(expr) is not None


def test_table_unpack():
    t = ibis.table({"id": "int32", "data": "struct<x: int32, y: string>"}, name="T")

    if version.parse(ibis.__version__) >= version.parse("11.0.0"):
        expr = t.unpack("data")
    else:
        expr = t.select(t.id, *t.data.destructure())

    assert from_ibis(expr) is not None


def test_limit_positional():
    t = ibis.table({"id": "int32", "value": "float64"}, name="T")
    expr = t.limit(10)
    assert from_ibis(expr) is not None


def test_limit_offset_keyword():
    t = ibis.table({"id": "int32", "value": "float64"}, name="T")
    expr = t.limit(10, offset=5)
    assert from_ibis(expr) is not None


def test_identical_to_positional():
    t = ibis.table({"id": "int32", "value": "float64"}, name="T")
    expr = t.filter(t.id.identical_to(1))
    assert from_ibis(expr) is not None
