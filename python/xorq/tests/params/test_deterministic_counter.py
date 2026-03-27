"""Tests for deterministic counter derivation in NamedScalarParameter."""

import pytest

import xorq.api as xo
from xorq.common.utils.name_utils import tokenize_to_int
from xorq.expr.operations import NamedScalarParameter


def test_counter_is_deterministic_for_same_label():
    a = NamedScalarParameter(dtype="float64", label="threshold")
    b = NamedScalarParameter(dtype="float64", label="threshold")
    assert a.counter == b.counter


def test_counter_differs_for_different_labels():
    a = NamedScalarParameter(dtype="float64", label="lo")
    b = NamedScalarParameter(dtype="float64", label="hi")
    assert a.counter != b.counter


def test_counter_differs_for_same_label_different_dtype():
    a = NamedScalarParameter(dtype="float64", label="x")
    b = NamedScalarParameter(dtype="int64", label="x")
    assert a.counter != b.counter


def test_counter_matches_tokenize_to_int():
    label = "cutoff"
    p = NamedScalarParameter(dtype="date", label=label)
    assert p.counter == tokenize_to_int(label, p.dtype)


def test_explicit_counter_is_respected():
    p = NamedScalarParameter(dtype="float64", label="x", counter=42)
    assert p.counter == 42


def test_param_api_counter_is_deterministic():
    a = xo.param("threshold", "float64")
    b = xo.param("threshold", "float64")
    assert a.op().counter == b.op().counter


def test_compatible_default_accepted():
    p = NamedScalarParameter(
        dtype="float64", label="x", default=1
    )  # int castable to float64
    assert p.default == 1


def test_incompatible_default_raises():
    with pytest.raises(TypeError, match="not compatible with dtype"):
        NamedScalarParameter(dtype="float64", label="x", default="not_a_float")


def test_execute_string_param():
    """String-typed named params execute correctly with default and custom values."""
    p = xo.param("prefix", "string", default="a")
    t = xo.memtable({"name": ["alice", "bob", "anna"]})
    expr = t.filter(t.name.startswith(p))

    assert sorted(expr.execute()["name"].tolist()) == ["alice", "anna"]
    assert sorted(expr.execute(params={p: "b"})["name"].tolist()) == ["bob"]
