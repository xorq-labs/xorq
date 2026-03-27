"""Tests for NamedScalarParameter replacement via bind_params."""

import pytest

import xorq.api as xo
import xorq.vendor.ibis as ibis
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.operations import NamedScalarParameter


def test_bound_param_replaced_with_literal():
    p = xo.param("x", "float64")
    t = ibis.table({"v": "float64"}, name="t")
    expr = t.filter(t.v > p)
    bound = xo.bind_params(expr, {"x": 1.5})
    assert walk_nodes(NamedScalarParameter, bound) == ()


def test_unbound_param_passes_through():
    p = xo.param("x", "float64", default=0.0)
    t = ibis.table({"v": "float64"}, name="t")
    expr = t.filter(t.v > p)
    # bind with empty dict — default is applied, param is still replaced
    bound = xo.bind_params(expr, {})
    assert walk_nodes(NamedScalarParameter, bound) == ()


def test_unbound_required_param_raises():
    xo.param("x", "float64")
    t = ibis.table({"v": "float64"}, name="t")
    p = xo.param("x", "float64")
    expr = t.filter(t.v > p)
    with pytest.raises(ValueError, match="Missing required parameters"):
        xo.bind_params(expr, {})


def test_partial_binding_leaves_unbound_intact():
    lo = xo.param("lo", "float64")
    hi = xo.param("hi", "float64", default=10.0)
    t = ibis.table({"v": "float64"}, name="t")
    expr = t.filter((t.v > lo) & (t.v < hi))
    bound = xo.bind_params(expr, {"lo": 0.0})
    assert walk_nodes(NamedScalarParameter, bound) == ()


def test_same_param_used_multiple_times_all_replaced():
    p = xo.param("threshold", "float64")
    q = xo.param("threshold", "float64")
    t = ibis.memtable({"v": [1.0, 2.0, 3.0, 4.0]})
    expr = t.filter((t.v > p) & (t.v < q + 2))
    bound = xo.bind_params(expr, {"threshold": 1.5})
    assert walk_nodes(NamedScalarParameter, bound) == ()
    assert bound.execute()["v"].tolist() == [2.0, 3.0]
