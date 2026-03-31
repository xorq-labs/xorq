"""Tests for ExprMetadata.params field."""

import pytest

import xorq.vendor.ibis as ibis
from xorq.expr.operations import NamedScalarParameter
from xorq.ibis_yaml.compiler import build_expr
from xorq.vendor.ibis.expr.types.core import ExprMetadata


def test_params_empty_when_no_named_params():
    t = ibis.memtable({"x": [1.0, 2.0]})
    assert ExprMetadata.from_expr(t).params == ()


def test_params_single():
    threshold = NamedScalarParameter(dtype="float64", label="threshold").to_expr()
    t = ibis.memtable({"x": [1.0, 2.0]})
    expr = t.filter(t.x > threshold)
    assert ExprMetadata.from_expr(expr).params == (
        {"param_name": "threshold", "type": "float64"},
    )


def test_params_multiple():
    lo = NamedScalarParameter(dtype="float64", label="lo").to_expr()
    hi = NamedScalarParameter(dtype="float64", label="hi").to_expr()
    t = ibis.memtable({"x": [1.0, 2.0]})
    expr = t.filter((t.x > lo) & (t.x < hi))
    labels = {p["param_name"] for p in ExprMetadata.from_expr(expr).params}
    assert labels == {"lo", "hi"}


def test_params_includes_default():
    p = NamedScalarParameter(dtype="float64", label="threshold", default=0.5).to_expr()
    t = ibis.memtable({"x": [1.0, 2.0]})
    expr = t.filter(t.x > p)
    assert ExprMetadata.from_expr(expr).params == (
        {"param_name": "threshold", "type": "float64", "default": 0.5},
    )


def test_params_omits_default_key_when_none():
    p = NamedScalarParameter(dtype="float64", label="threshold").to_expr()
    t = ibis.memtable({"x": [1.0, 2.0]})
    expr = t.filter(t.x > p)
    assert "default" not in ExprMetadata.from_expr(expr).params[0]


def test_to_dict_includes_params():
    cutoff = NamedScalarParameter(dtype="date", label="cutoff").to_expr()
    t = ibis.table({"d": "date", "v": "float64"}, name="t")
    expr = t.filter(t.d > cutoff)
    d = ExprMetadata.from_expr(expr).to_dict()
    assert "params" in d
    assert d["params"] == ({"param_name": "cutoff", "type": "date"},)


def test_to_dict_omits_params_when_empty():
    t = ibis.memtable({"x": [1.0]})
    assert "params" not in ExprMetadata.from_expr(t).to_dict()


def test_conflicting_param_dtypes_raises():
    p_float = NamedScalarParameter(dtype="float64", label="x").to_expr()
    p_int = NamedScalarParameter(dtype="int64", label="x").to_expr()
    t = ibis.memtable({"x": [1.0, 2.0]})
    expr = t.filter((t.x > p_float) & (t.x < p_int))
    with pytest.raises(TypeError, match="conflicting dtypes"):
        ExprMetadata.from_expr(expr)


def test_conflicting_param_dtypes_reports_all_conflicts():
    """All conflicting labels are reported in a single TypeError, not just the first."""
    px_float = NamedScalarParameter(dtype="float64", label="x").to_expr()
    px_int = NamedScalarParameter(dtype="int64", label="x").to_expr()
    py_str = NamedScalarParameter(dtype="string", label="y").to_expr()
    py_bool = NamedScalarParameter(dtype="boolean", label="y").to_expr()
    t = ibis.memtable({"x": [1.0], "y": ["a"]})
    expr = t.filter((t.x > px_float) & (t.x < px_int) & (t.y == py_str)).filter(py_bool)
    with pytest.raises(TypeError, match="'x'") as exc_info:
        ExprMetadata.from_expr(expr)
    assert "'y'" in str(exc_info.value)


def test_conflicting_param_dtypes_raises_on_build(tmp_path):
    p_float = NamedScalarParameter(dtype="float64", label="x").to_expr()
    p_int = NamedScalarParameter(dtype="int64", label="x").to_expr()
    t = ibis.memtable({"x": [1.0, 2.0]})
    expr = t.filter((t.x > p_float) & (t.x < p_int))
    with pytest.raises(TypeError, match="conflicting dtypes"):
        build_expr(expr, builds_dir=tmp_path)
