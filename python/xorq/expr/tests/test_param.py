"""Tests for xorq.param public API."""

from datetime import date

import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.types as ir
from xorq.expr.operations import NamedScalarParameter


def test_param_returns_scalar():
    p = xo.param("threshold", "float64")
    assert isinstance(p, ir.Expr)


def test_param_op_is_named_scalar_parameter():
    p = xo.param("cutoff", "date")
    assert isinstance(p.op(), NamedScalarParameter)


def test_param_label():
    p = xo.param("region", "string")
    assert p.op().label == "region"
    assert p.op().name == "region"


def test_param_dtype_string():
    p = xo.param("limit", "int64")
    assert str(p.op().dtype) == "int64"


def test_param_dtype_object():
    p = xo.param("start", dt.timestamp())
    assert str(p.op().dtype) == "timestamp"


def test_param_used_in_filter_executes():
    threshold = xo.param("threshold", "float64")
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)
    result = expr.execute(params={threshold: 1.5})
    assert list(result["x"]) == [2.0, 3.0]


def test_param_used_in_filter_different_values():
    cutoff = xo.param("cutoff", "date")
    t = xo.memtable({"d": [date(2024, 1, 1), date(2024, 6, 1), date(2024, 12, 1)]})
    expr = t.filter(t.d >= cutoff)

    result_early = expr.execute(params={cutoff: date(2024, 6, 1)})
    assert len(result_early) == 2

    result_late = expr.execute(params={cutoff: date(2024, 12, 1)})
    assert len(result_late) == 1


@pytest.mark.parametrize(
    "type_str",
    ["float64", "int64", "string", "date", "timestamp", "boolean"],
)
def test_param_various_types(type_str):
    p = xo.param("p", type_str)
    assert str(p.op().dtype) == type_str
    assert p.op().label == "p"


def test_param_default_none_when_not_supplied():
    p = xo.param("cutoff", "date")
    assert p.op().default is None


def test_param_default_value_stored():
    p = xo.param("threshold", "float64", default=0.5)
    assert p.op().default == 0.5


def test_param_default_integer():
    p = xo.param("limit", "int64", default=10)
    assert p.op().default == 10


def test_param_execute_with_default():
    threshold = xo.param("threshold", "float64", default=1.5)
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)
    result = expr.execute(params={threshold: threshold.op().default})
    assert list(result["x"]) == [2.0, 3.0]
