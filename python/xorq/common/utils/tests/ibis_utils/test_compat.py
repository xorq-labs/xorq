import ibis
import pandas as pd
import pytest
from packaging import version

from xorq.common.utils.ibis_utils import from_ibis
from xorq.tests.util import assert_frame_equal, assert_series_equal


@pytest.mark.parametrize(
    "make_sort_key",
    (
        lambda t: ibis.asc("value"),
        lambda t: t.value.asc(),
        lambda t: ibis.desc("value"),
        lambda t: t.value.desc(),
        lambda t: ibis.asc(t.value, nulls_first=True),
    ),
)
def test_sort_keys(make_sort_key):
    data = pd.DataFrame({"id": [1, 2, 3, 4, 5], "value": [3.5, None, 4.8, 2.1, None]})
    t = ibis.memtable(data)
    expr = t.order_by(make_sort_key(t))

    xorq_expr = from_ibis(expr)

    assert_frame_equal(expr.execute(), xorq_expr.execute())


def test_cases():
    data = pd.DataFrame(
        {"int_col": [1, 2, 3, 1, 2], "string_col": ["a", "b", "c", "d", "e"]}
    )
    t = ibis.memtable(data)

    args = [(1, "abcd"), (2, "ABCD")]
    kwargs = {"else_": "dabc"}
    if version.parse(ibis.__version__) <= version.parse("9.5.0"):
        args = [[(1, "abcd"), (2, "ABCD")]]
        kwargs = {"default": "dabc"}

    expr = t.int_col.cases(*args, **kwargs)

    xorq_expr = from_ibis(expr)

    assert_series_equal(expr.execute(), xorq_expr.execute())


@pytest.mark.parametrize(
    "make_filter",
    (lambda t: t.int_col.isin((1, 2, 3)), lambda t: t.int_col.notin((1, 2, 3))),
)
def test_positional(make_filter):
    data = pd.DataFrame(
        {"int_col": [1, 2, 3, 4, 5], "string_col": ["a", "b", "c", "d", "e"]}
    )
    t = ibis.memtable(data)
    expr = t.filter(make_filter(t))

    xorq_expr = from_ibis(expr)

    assert_frame_equal(expr.execute(), xorq_expr.execute())


@pytest.mark.parametrize(
    "make_window",
    (
        lambda t: ibis.window(group_by=t.category, order_by=t.id),
        lambda t: ibis.window(
            group_by=t.category, order_by=t.id, preceding=2, following=0
        ),
        lambda t: None,
    ),
)
def test_window_with_rows(make_window):
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "category": ["A", "B", "A", "B", "A", "B"],
        }
    )
    t = ibis.memtable(data)

    expr = t.mutate(average=t.value.mean().over(make_window(t))).order_by("id")

    xorq_expr = from_ibis(expr)

    assert_frame_equal(expr.execute(), xorq_expr.execute())


def test_string_as_date():
    s = ibis.literal("2024-01-15")

    if version.parse(ibis.__version__) >= version.parse("11.0.0"):
        expr = s.as_date("%Y-%m-%d")
    else:
        expr = s.to_date("%Y-%m-%d")

    xorq_expr = from_ibis(expr)

    assert expr.execute() == pd.to_datetime(xorq_expr.execute())


def test_string_as_timestamp():
    s = ibis.literal("2024-01-15 10:30:00")

    if version.parse(ibis.__version__) >= version.parse("11.0.0"):
        expr = s.as_timestamp("%Y-%m-%d %H:%M:%S")
    else:
        expr = s.to_timestamp("%Y-%m-%d %H:%M:%S")

    xorq_expr = from_ibis(expr)

    assert expr.execute() == xorq_expr.execute()


def test_integer_as_interval():
    i = ibis.literal(5)

    if version.parse(ibis.__version__) >= version.parse("10.0.0"):
        expr = i.as_interval("day")
    else:
        expr = i.to_interval(unit="day")

    xorq_expr = from_ibis(expr)

    assert expr.execute() == xorq_expr.execute()


def test_integer_as_timestamp():
    i = ibis.literal(1705315800)

    if version.parse(ibis.__version__) >= version.parse("10.0.0"):
        expr = i.as_timestamp("s")
    else:
        expr = i.to_timestamp(unit="s")

    xorq_expr = from_ibis(expr)

    assert expr.execute() == xorq_expr.execute()


def test_table_unpack():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "data": [{"x": 10, "y": "a"}, {"x": 20, "y": "b"}, {"x": 30, "y": "c"}],
        }
    )
    t = ibis.memtable(data)

    if version.parse(ibis.__version__) >= version.parse("11.0.0"):
        expr = t.unpack("data")
    else:
        expr = t.select(t.id, *t.data.destructure())

    xorq_expr = from_ibis(expr)

    assert_frame_equal(expr.execute(), xorq_expr.execute())


def test_limit_positional():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        }
    )
    t = ibis.memtable(data)
    expr = t.limit(10)

    xorq_expr = from_ibis(expr)

    assert_frame_equal(expr.execute(), xorq_expr.execute())


def test_limit_offset_keyword():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        }
    )
    t = ibis.memtable(data)
    expr = t.limit(10, offset=5)

    xorq_expr = from_ibis(expr)

    assert_frame_equal(expr.execute(), xorq_expr.execute())


def test_identical_to_positional():
    data = pd.DataFrame({"id": [1, 2, None, 4, 1], "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    t = ibis.memtable(data)
    expr = t.filter(t.id.identical_to(1))

    xorq_expr = from_ibis(expr)

    assert_frame_equal(expr.execute(), xorq_expr.execute())
