import pytest

import xorq.expr.datatypes as dt
from xorq.expr.operations import NamedScalarParameter
from xorq.vendor.ibis.expr.operations.generic import ScalarParameter


def test_named_scalar_parameter_creation():
    op = NamedScalarParameter(dtype=dt.float64(), label="threshold")
    assert op.label == "threshold"
    assert op.dtype == dt.float64()
    assert op.counter is not None


def test_named_scalar_parameter_name_returns_label():
    op = NamedScalarParameter(dtype=dt.float64(), label="threshold")
    assert op.name == "threshold"


def test_named_scalar_parameter_counter_explicit():
    op = NamedScalarParameter(dtype=dt.float64(), label="x", counter=99)
    assert op.counter == 99


def test_named_scalar_parameter_counter_auto_increments():
    op1 = NamedScalarParameter(dtype=dt.float64(), label="a")
    op2 = NamedScalarParameter(dtype=dt.float64(), label="b")
    assert op1.counter != op2.counter


def test_named_scalar_parameter_is_scalar_parameter():
    op = NamedScalarParameter(dtype=dt.date(), label="cutoff")
    assert isinstance(op, ScalarParameter)


def test_named_scalar_parameter_to_expr():
    op = NamedScalarParameter(dtype=dt.timestamp(), label="start")
    expr = op.to_expr()
    assert expr is not None
    assert expr.op() is op


@pytest.mark.parametrize(
    "type_str,label",
    [
        ("float64", "threshold"),
        ("date", "cutoff"),
        ("timestamp", "start"),
        ("string", "region"),
        ("int64", "limit"),
    ],
)
def test_named_scalar_parameter_various_types(type_str, label):
    op = NamedScalarParameter(dtype=dt.dtype(type_str), label=label)
    assert op.label == label
    assert op.name == label
    assert str(op.dtype) == type_str
