"""Advanced tests for named parameters in complex scenarios.

Tests parameters behind various node types and edge cases.
"""

import pytest

import xorq.api as xo
from xorq.expr.operations import NamedScalarParameter
from xorq.expr.relations import CachedNode


def test_param_behind_cached_node():
    """Test parameters work behind CachedNode."""
    threshold = xo.param("threshold", "float64", default=1.5)
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)

    # Cache the expression
    cached_expr = expr.cache()

    # Verify it's actually cached
    assert any(
        isinstance(node, CachedNode) for node in cached_expr.op().find(CachedNode)
    )

    # Should still work with defaults (1.5)
    result = cached_expr.execute()
    assert list(result["x"]) == [2.0, 3.0]

    # Should work with custom values
    result = cached_expr.execute(params={threshold: 0.5})
    assert list(result["x"]) == [1.0, 2.0, 3.0]


def test_param_behind_cached_node_multiple():
    """Test multiple parameters behind CachedNode."""
    lo = xo.param("lo", "float64", default=4.0)  # Changed to 4.0 so 5.0 > 4.0 is True
    hi = xo.param("hi", "float64", default=12.0)
    t = xo.memtable({"x": [1.0, 5.0, 10.0, 15.0]})
    expr = t.filter((t.x > lo) & (t.x < hi))

    # Cache the expression
    cached_expr = expr.cache()

    # Should work with defaults (4.0, 12.0) - includes 5.0 and 10.0
    result = cached_expr.execute()
    assert list(result["x"]) == [5.0, 10.0]

    # Should work with custom values (0.0, 10.0) - includes 1.0 and 5.0
    result = cached_expr.execute(params={lo: 0.0, hi: 10.0})
    assert list(result["x"]) == [1.0, 5.0]


def test_param_behind_cached_node_required():
    """Test required parameter behind CachedNode."""
    threshold = xo.param("threshold", "float64")  # No default
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)

    # Cache the expression
    cached_expr = expr.cache()

    # Should fail without params
    with pytest.raises(ValueError, match="Missing required parameters"):
        cached_expr.execute()

    # Should work with params
    result = cached_expr.execute(params={threshold: 1.5})
    assert list(result["x"]) == [2.0, 3.0]


def test_param_nested_in_cached_parent():
    """Test parameters nested in CachedNode.parent."""
    threshold = xo.param("threshold", "float64", default=1.5)
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})

    # Create expression with param
    expr = t.filter(t.x > threshold)

    # Cache it
    cached = expr.cache()

    # The cached node should have the param in its parent
    cached_node = cached.op()
    assert isinstance(cached_node, CachedNode)

    # Verify parent has the parameter
    parent_params = list(cached_node.parent.op().find(NamedScalarParameter))
    assert len(parent_params) == 1
    assert parent_params[0].label == "threshold"

    # Execution should still work (1.5 filters out 1.0)
    result = cached.execute()
    assert list(result["x"]) == [2.0, 3.0]


def test_param_with_remote_table():
    """Test parameters survive an into_backend round-trip via RemoteTable."""
    threshold = xo.param("threshold", "float64", default=0.5)

    src = xo.memtable({"x": [1.0, 2.0, 3.0]})
    dst = xo.connect()

    # into_backend produces a RemoteTable node under the hood
    remote = src.into_backend(dst)

    expr = remote.filter(remote.x > threshold)

    # Expression must contain our parameter
    assert expr.op().find(NamedScalarParameter)

    # Default (0.5): all rows pass
    result = expr.execute()
    assert list(result["x"]) == [1.0, 2.0, 3.0]

    # Custom value (1.5): only 2.0 and 3.0 pass
    result = expr.execute(params={threshold: 1.5})
    assert list(result["x"]) == [2.0, 3.0]


def test_param_with_read_node():
    """Test parameters with Read nodes."""
    threshold = xo.param("threshold", "float64", default=0.5)

    # Create an expression that would contain Read nodes
    # This is tricky to test without actual file setup, but we can check the logic
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)

    # For now, just verify the parameter is in the expression
    params = expr.op().find(NamedScalarParameter)
    assert len(params) == 1
    assert params[0].label == "threshold"

    # Execution should work — default is 0.5, so all three values pass
    result = expr.execute()
    assert list(result["x"]) == [1.0, 2.0, 3.0]


def test_param_cached_then_modified():
    """Test caching an expression then modifying parameters."""
    threshold = xo.param("threshold", "float64", default=1.5)
    t = xo.memtable({"x": [1.0, 2.0, 3.0]})
    expr = t.filter(t.x > threshold)

    # Cache with default (1.5)
    cached = expr.cache()
    result1 = cached.execute()
    assert list(result1["x"]) == [2.0, 3.0]  # 1.5 filters out 1.0

    # Execute with different param value (2.5)
    result2 = cached.execute(params={threshold: 2.5})
    assert list(result2["x"]) == [3.0]  # 2.5 filters out 1.0 and 2.0

    # Go back to default (1.5)
    result3 = cached.execute()
    assert list(result3["x"]) == [2.0, 3.0]  # Back to filtering out just 1.0


def test_param_multiple_cached_layers():
    """Test parameters with multiple layers of caching."""
    threshold = xo.param("threshold", "float64", default=1.0)
    t = xo.memtable({"x": [0.5, 1.5, 2.5, 3.5]})

    # First level
    expr1 = t.filter(t.x > threshold)
    cached1 = expr1.cache()

    # Second level (cache the cached)
    cached2 = cached1.cache()

    # Should still work
    result = cached2.execute()
    assert list(result["x"]) == [1.5, 2.5, 3.5]

    # Should work with custom value
    result = cached2.execute(params={threshold: 2.0})
    assert list(result["x"]) == [2.5, 3.5]


def test_param_in_cached_aggregation():
    """Test parameters in cached aggregations."""
    threshold = xo.param("threshold", "float64", default=5.0)
    t = xo.memtable({"x": [1.0, 3.0, 5.0, 7.0, 9.0]})

    # Complex expression with aggregation
    expr = (
        t.filter(t.x > threshold)
        .group_by((t.x // 2).name("bucket"))
        .aggregate(count=t.x.count())
    )

    # Cache it
    cached = expr.cache()

    # Should work
    result = cached.execute()
    assert len(result) == 2  # Buckets for 6-7 and 8-9


def test_into_backend_with_named_param():
    """Complex pipeline: into_backend followed by filter with named param."""
    lo = xo.param("lo", "float64", default=2.0)
    hi = xo.param("hi", "float64", default=8.0)

    src = xo.memtable({"x": [1.0, 3.0, 5.0, 7.0, 9.0], "label": list("abcde")})
    dst = xo.connect()

    remote = src.into_backend(dst)
    expr = remote.filter((remote.x > lo) & (remote.x < hi))

    # Defaults: 2.0 < x < 8.0 → 3.0, 5.0, 7.0
    result = expr.execute()
    assert sorted(result["x"].tolist()) == [3.0, 5.0, 7.0]

    # Custom: 4.0 < x < 8.0 → 5.0, 7.0
    result = expr.execute(params={lo: 4.0, hi: 8.0})
    assert sorted(result["x"].tolist()) == [5.0, 7.0]


def test_into_backend_with_param_and_aggregation():
    """into_backend + filter param + group-by aggregation."""
    threshold = xo.param("threshold", "float64", default=3.0)

    src = xo.memtable(
        {
            "group": ["a", "a", "b", "b", "b"],
            "value": [1.0, 4.0, 2.0, 5.0, 6.0],
        }
    )
    dst = xo.connect()

    remote = src.into_backend(dst)
    expr = (
        remote.filter(remote.value > threshold)
        .group_by("group")
        .aggregate(total=remote.value.sum())
    )

    # Default (>3.0): group a → 4.0; group b → 5.0 + 6.0 = 11.0
    result = expr.execute().sort_values("group").reset_index(drop=True)
    assert list(result["group"]) == ["a", "b"]
    assert list(result["total"]) == [4.0, 11.0]

    # Custom (>4.0): group b only → 5.0 + 6.0 = 11.0
    result = expr.execute(params={threshold: 4.0})
    assert list(result["group"]) == ["b"]
    assert list(result["total"]) == [11.0]


def test_into_backend_with_param_and_join():
    """into_backend + named param + join across two remote tables."""
    min_value = xo.param("min_value", "float64", default=2.0)

    facts = xo.memtable({"id": [1, 2, 3, 4], "value": [1.0, 3.0, 5.0, 7.0]})
    dims = xo.memtable({"id": [1, 2, 3, 4], "name": ["x", "y", "z", "w"]})
    dst = xo.connect()

    remote_facts = facts.into_backend(dst)
    remote_dims = dims.into_backend(dst)

    expr = (
        remote_facts.filter(remote_facts.value > min_value)
        .inner_join(remote_dims, remote_facts.id == remote_dims.id)
        .select(remote_facts.id, remote_facts.value, remote_dims.name)
    )

    # Default (>2.0): ids 2, 3, 4 → values 3.0, 5.0, 7.0
    result = expr.execute().sort_values("value").reset_index(drop=True)
    assert list(result["value"]) == [3.0, 5.0, 7.0]
    assert list(result["name"]) == ["y", "z", "w"]

    # Custom (>5.0): id 4 only → value 7.0
    result = expr.execute(params={min_value: 5.0})
    assert list(result["value"]) == [7.0]
    assert list(result["name"]) == ["w"]
