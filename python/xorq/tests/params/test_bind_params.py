"""Tests for bind_params name-based parameter binding."""

from datetime import date

import pytest

import xorq.api as xo
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.operations import NamedScalarParameter


def test_bind_replaces_all_params():
    threshold = xo.param("threshold", "float64")
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)
    bound = xo.bind_params(expr, {"threshold": 1.5})
    assert walk_nodes(NamedScalarParameter, bound) == ()


def test_bind_produces_correct_results():
    threshold = xo.param("threshold", "float64")
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)
    result = xo.bind_params(expr, {"threshold": 1.5}).execute()
    assert list(result["x"]) == [2.0, 3.0]


def test_bind_applies_default_for_omitted_param():
    threshold = xo.param("threshold", "float64", default=1.5)
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)
    result = xo.bind_params(expr, {}).execute()
    assert list(result["x"]) == [2.0, 3.0]


def test_bind_explicit_value_overrides_default():
    threshold = xo.param("threshold", "float64", default=0.0)
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)
    result = xo.bind_params(expr, {"threshold": 2.5}).execute()
    assert list(result["x"]) == [3.0]


def test_bind_raises_on_missing_required_param():
    threshold = xo.param("threshold", "float64")
    t = xo.memtable({"x": [1.0]})
    expr = t.filter(t.x > threshold)
    with pytest.raises(ValueError, match="Missing required parameters"):
        xo.bind_params(expr, {})


def test_bind_multiple_params():
    lo = xo.param("lo", "float64")
    hi = xo.param("hi", "float64")
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter((t.x > lo) & (t.x < hi))
    result = xo.bind_params(expr, {"lo": 1.0, "hi": 3.0}).execute()
    assert list(result["x"]) == [2.0]


def test_bind_date_param():
    cutoff = xo.param("cutoff", "date")
    t = xo.memtable({"d": [date(2024, 1, 1), date(2024, 6, 1), date(2024, 12, 1)]})
    expr = t.filter(t.d >= cutoff)
    result = xo.bind_params(expr, {"cutoff": date(2024, 6, 1)}).execute()
    assert len(result) == 2
