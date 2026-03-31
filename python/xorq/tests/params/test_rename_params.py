"""Tests for rename_params() utility."""

import xorq.api as xo
import xorq.vendor.ibis as ibis
from xorq.common.utils.graph_utils import rename_params, walk_nodes
from xorq.expr.operations import NamedScalarParameter


def test_rename_single_param():
    p = xo.param("threshold", "float64")
    t = ibis.table({"v": "float64"}, name="t")
    expr = t.filter(t.v > p)
    renamed = rename_params(expr, {"threshold": "cutoff"})
    nodes = walk_nodes(NamedScalarParameter, renamed)
    assert len(nodes) == 1
    assert nodes[0].label == "cutoff"
    assert str(nodes[0].dtype) == "float64"


def test_rename_preserves_default():
    p = xo.param("threshold", "float64", default=1.5)
    t = ibis.table({"v": "float64"}, name="t")
    expr = t.filter(t.v > p)
    renamed = rename_params(expr, {"threshold": "cutoff"})
    nodes = walk_nodes(NamedScalarParameter, renamed)
    assert len(nodes) == 1
    assert nodes[0].label == "cutoff"
    assert nodes[0].default == 1.5


def test_rename_nonexistent_label_is_noop():
    p = xo.param("threshold", "float64")
    t = ibis.table({"v": "float64"}, name="t")
    expr = t.filter(t.v > p)
    renamed = rename_params(expr, {"nonexistent": "something"})
    nodes = walk_nodes(NamedScalarParameter, renamed)
    assert len(nodes) == 1
    assert nodes[0].label == "threshold"


def test_rename_multiple_params():
    lo = xo.param("lo", "float64")
    hi = xo.param("hi", "float64")
    t = ibis.table({"v": "float64"}, name="t")
    expr = t.filter((t.v > lo) & (t.v < hi))
    renamed = rename_params(expr, {"lo": "low", "hi": "high"})
    nodes = walk_nodes(NamedScalarParameter, renamed)
    labels = {n.label for n in nodes}
    assert labels == {"low", "high"}


def test_rename_preserves_dtype():
    p = xo.param("cutoff", "date")
    t = ibis.table({"d": "date"}, name="t")
    expr = t.filter(t.d >= p)
    renamed = rename_params(expr, {"cutoff": "start_date"})
    nodes = walk_nodes(NamedScalarParameter, renamed)
    assert len(nodes) == 1
    assert nodes[0].label == "start_date"
    assert str(nodes[0].dtype) == "date"
