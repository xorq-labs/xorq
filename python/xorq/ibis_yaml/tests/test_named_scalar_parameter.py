"""Round-trip serialization tests for NamedScalarParameter."""

import json

import pytest

import xorq.expr.datatypes as dt
import xorq.vendor.ibis as ibis
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.operations import NamedScalarParameter
from xorq.ibis_yaml.compiler import YamlExpressionTranslator


@pytest.fixture
def compiler():
    return YamlExpressionTranslator


@pytest.fixture
def t():
    return ibis.table({"x": "float64", "y": "int64", "s": "string"}, name="t")


def test_named_scalar_parameter_serializes(compiler):
    op = NamedScalarParameter(dtype=dt.float64(), label="threshold")
    param_expr = op.to_expr()

    t = ibis.table({"x": "float64"}, name="t")
    expr = t.filter(t.x > param_expr)

    yaml_dict = compiler.to_yaml(expr)
    assert yaml_dict is not None


def test_named_scalar_parameter_round_trips_label(compiler, t):
    threshold = NamedScalarParameter(dtype=dt.float64(), label="threshold").to_expr()
    expr = t.filter(t.x > threshold)

    yaml_dict = compiler.to_yaml(expr)
    restored = compiler.from_yaml(yaml_dict)

    params = walk_nodes(NamedScalarParameter, restored)
    assert len(params) == 1
    assert params[0].label == "threshold"


def test_named_scalar_parameter_round_trips_dtype(compiler):
    cutoff = NamedScalarParameter(dtype=dt.date(), label="cutoff").to_expr()
    t2 = ibis.table({"d": "date", "v": "float64"}, name="t2")
    expr2 = t2.filter(t2.d > cutoff)

    yaml_dict = compiler.to_yaml(expr2)
    restored = compiler.from_yaml(yaml_dict)

    params = walk_nodes(NamedScalarParameter, restored)
    assert len(params) == 1
    assert params[0].dtype == dt.date()


def test_named_scalar_parameter_op_key_in_yaml(compiler, t):
    threshold = NamedScalarParameter(dtype=dt.float64(), label="my_param").to_expr()
    expr = t.filter(t.x > threshold)

    yaml_dict = compiler.to_yaml(expr)
    assert "NamedScalarParameter" in json.dumps(yaml_dict)


def test_named_scalar_parameter_multiple_params_round_trip(compiler):
    t = ibis.table({"x": "float64", "y": "float64"}, name="t")
    lo = NamedScalarParameter(dtype="float64", label="lo").to_expr()
    hi = NamedScalarParameter(dtype="float64", label="hi").to_expr()
    expr = t.filter((t.x > lo) & (t.x < hi))

    yaml_dict = compiler.to_yaml(expr)
    restored = compiler.from_yaml(yaml_dict)

    params = walk_nodes(NamedScalarParameter, restored)
    labels = {p.label for p in params}
    assert labels == {"lo", "hi"}


def test_named_scalar_parameter_default_numeric_round_trips(compiler):
    threshold = NamedScalarParameter(
        dtype=dt.float64(), label="threshold", default=0.5
    ).to_expr()
    t = ibis.table({"x": "float64"}, name="t")
    expr = t.filter(t.x > threshold)

    yaml_dict = compiler.to_yaml(expr)
    restored = compiler.from_yaml(yaml_dict)

    params = walk_nodes(NamedScalarParameter, restored)
    assert len(params) == 1
    assert params[0].default == 0.5


def test_named_scalar_parameter_default_none_round_trips(compiler, t):
    p = NamedScalarParameter(dtype=dt.float64(), label="p").to_expr()
    expr = t.filter(t.x > p)

    yaml_dict = compiler.to_yaml(expr)
    restored = compiler.from_yaml(yaml_dict)

    params = walk_nodes(NamedScalarParameter, restored)
    assert params[0].default is None


def test_named_scalar_parameter_default_not_in_yaml_when_none(compiler, t):
    p = NamedScalarParameter(dtype=dt.float64(), label="p").to_expr()
    expr = t.filter(t.x > p)

    yaml_dict = compiler.to_yaml(expr)
    assert "default" not in json.dumps(yaml_dict)
