"""Tests for CLI --params flag and coercion helpers."""

import datetime

import click
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.cli import _coerce_param, _parse_cli_params
from xorq.expr.operations import NamedScalarParameter


# --- _coerce_param ---


@pytest.mark.parametrize(
    "value,dtype,expected",
    [
        ("1.5", dt.float64(), 1.5),
        ("3", dt.int64(), 3),
        ("hello", dt.string(), "hello"),
        ("2024-06-01", dt.date(), datetime.date(2024, 6, 1)),
        (
            "2024-06-01T12:00:00",
            dt.timestamp(),
            datetime.datetime(2024, 6, 1, 12, 0, 0),
        ),
        ("true", dt.boolean(), True),
        ("false", dt.boolean(), False),
        ("1", dt.boolean(), True),
    ],
)
def test_coerce_param(value, dtype, expected):
    assert _coerce_param(value, dtype) == expected


def test_coerce_param_unknown_dtype_raises():
    with pytest.raises(click.BadParameter):
        _coerce_param("x", dt.Array(dt.int64()))


# --- _parse_cli_params ---


def _make_expr():
    threshold = xo.param("threshold", "float64")
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    return t.filter(t.x > threshold)


def test_parse_cli_params_empty_returns_empty_dict():
    expr = _make_expr()
    assert _parse_cli_params(expr, ()) == {}


def test_parse_cli_params_returns_param_expr_dict():
    expr = _make_expr()
    params = _parse_cli_params(expr, ("threshold=1.5",))
    assert len(params) == 1
    (param_expr, value) = next(iter(params.items()))
    assert isinstance(param_expr.op(), NamedScalarParameter)
    assert param_expr.op().label == "threshold"
    assert value == 1.5


def test_parse_cli_params_executes_correctly():
    expr = _make_expr()
    params = _parse_cli_params(expr, ("threshold=1.5",))
    assert list(expr.execute(params=params)["x"]) == [2.0, 3.0]


def test_parse_cli_params_bad_format_raises():
    with pytest.raises(click.BadParameter, match="Expected key=value"):
        _parse_cli_params(_make_expr(), ("threshold",))


def test_parse_cli_params_unknown_param_raises():
    with pytest.raises(click.BadParameter, match="Unknown parameter"):
        _parse_cli_params(_make_expr(), ("unknown=1.0",))
