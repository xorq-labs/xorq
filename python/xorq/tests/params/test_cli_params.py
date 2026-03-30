"""Tests for CLI --params flag and coercion helpers."""

import datetime

import click
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.cli import _click_type_for_dtype, _ClickDate, _parse_cli_params
from xorq.expr.api import bind_params


# --- _ClickDate ---


def test_click_date_valid():
    ct = _ClickDate()
    assert ct.convert("2024-06-01", param=None, ctx=None) == datetime.date(2024, 6, 1)


def test_click_date_invalid_raises():
    ct = _ClickDate()
    with pytest.raises(click.exceptions.BadParameter, match="not a valid date"):
        ct.convert("not-a-date", param=None, ctx=None)


def test_click_date_name():
    assert _ClickDate().name == "date"


# --- _click_type_for_dtype ---


@pytest.mark.parametrize(
    "value,dtype,expected",
    [
        ("1.5", dt.float64(), 1.5),
        ("-3.14", dt.float32(), -3.14),
        ("3", dt.int64(), 3),
        ("-1", dt.int32(), -1),
        ("0", dt.int16(), 0),
        ("127", dt.int8(), 127),
        ("hello", dt.string(), "hello"),
        ("", dt.string(), ""),
        ("2024-06-01", dt.date(), datetime.date(2024, 6, 1)),
        (
            "2024-06-01T12:00:00",
            dt.timestamp(),
            datetime.datetime(2024, 6, 1, 12, 0, 0),
        ),
        ("true", dt.boolean(), True),
        ("false", dt.boolean(), False),
    ],
)
def test_click_type_coercion(value, dtype, expected):
    click_type = _click_type_for_dtype(dtype)
    assert click_type.convert(value, param=None, ctx=None) == expected


def test_click_type_unsupported_dtype_raises():
    with pytest.raises(click.BadParameter, match="Unsupported parameter dtype"):
        _click_type_for_dtype(dt.Array(dt.int64()))


@pytest.mark.parametrize(
    "dtype",
    [
        dt.float64(),
        dt.float32(),
        dt.int64(),
        dt.int32(),
        dt.int16(),
        dt.int8(),
        dt.string(),
        dt.boolean(),
        dt.date(),
        dt.timestamp(),
    ],
)
def test_click_type_for_dtype_returns_click_param_type(dtype):
    result = _click_type_for_dtype(dtype)
    assert isinstance(result, click.ParamType)


def test_click_type_float_rejects_non_numeric():
    click_type = _click_type_for_dtype(dt.float64())
    with pytest.raises(click.exceptions.BadParameter):
        click_type.convert("abc", param=None, ctx=None)


def test_click_type_int_rejects_float_string():
    click_type = _click_type_for_dtype(dt.int64())
    with pytest.raises(click.exceptions.BadParameter):
        click_type.convert("1.5", param=None, ctx=None)


# --- _parse_cli_params ---


def _make_expr():
    threshold = xo.param("threshold", "float64")
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    return t.filter(t.x > threshold)


def _make_multi_param_expr():
    threshold = xo.param("threshold", "float64")
    label = xo.param("label", "string")
    t = xo.memtable({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "c"]})
    return t.filter(t.x > threshold).filter(t.y == label)


def test_parse_cli_params_empty_returns_empty_dict():
    expr = _make_expr()
    assert _parse_cli_params(expr, ()) == {}


def test_parse_cli_params_returns_name_value_dict():
    expr = _make_expr()
    params = _parse_cli_params(expr, ("threshold=1.5",))
    assert params == {"threshold": 1.5}


def test_parse_cli_params_multiple_params():
    expr = _make_multi_param_expr()
    params = _parse_cli_params(expr, ("threshold=2.0", "label=b"))
    assert params == {"threshold": 2.0, "label": "b"}


def test_parse_cli_params_value_with_equals():
    """Values containing '=' should be preserved (only first '=' splits)."""
    label = xo.param("label", "string")
    t = xo.memtable({"x": ["a"]})
    expr = t.filter(t.x == label)
    params = _parse_cli_params(expr, ("label=a=b",))
    assert params == {"label": "a=b"}


def test_parse_cli_params_bad_format_raises():
    with pytest.raises(click.BadParameter, match="Expected key=value"):
        _parse_cli_params(_make_expr(), ("threshold",))


def test_parse_cli_params_unknown_param_raises():
    with pytest.raises(click.BadParameter, match="Unknown parameter"):
        _parse_cli_params(_make_expr(), ("unknown=1.0",))


def test_parse_cli_params_type_error_propagates():
    """Passing a non-numeric value for a float param should raise."""
    with pytest.raises(click.exceptions.BadParameter):
        _parse_cli_params(_make_expr(), ("threshold=abc",))


def test_parse_cli_params_result_works_with_bind_params():
    """End-to-end: parsed params can be passed to bind_params and executed."""
    expr = _make_expr()
    params = _parse_cli_params(expr, ("threshold=1.5",))
    bound = bind_params(expr, params)
    result = bound.execute()
    assert list(result["x"]) == [2.0, 3.0]
