import pytest

import xorq.api as xo
from xorq.caching.strategy import SnapshotStrategy
from xorq.common.utils.node_utils import (
    compute_expr_hash,
    find_by_expr_hash,
)
from xorq.expr.relations import HashingTag, Tag
from xorq.ibis_yaml.common import RefEnum, RegistryEnum
from xorq.ibis_yaml.compiler import YamlExpressionTranslator
from xorq.vendor import ibis


@pytest.fixture
def t():
    return ibis.table(
        {"a": "int64", "b": "string"},
        name="test_table",
    )


@pytest.fixture
def compiler():
    return YamlExpressionTranslator()


def test_different_hashing_tag_metadata_produces_distinct_hashes(t):
    """Two expressions with same structure but different HashingTag dimensions produce distinct hashes."""
    expr_v1 = t.hashing_tag("v1")
    expr_v2 = t.hashing_tag("v2")

    strategy = SnapshotStrategy()
    hash_v1 = compute_expr_hash(expr_v1, strategy=strategy)
    hash_v2 = compute_expr_hash(expr_v2, strategy=strategy)

    assert hash_v1 != hash_v2


def test_hashing_tag_differs_from_plain_tag(t):
    """expr.hashing_tag('v1') hashes differently from expr.tag('v1')."""
    tagged = t.tag("v1")
    hashing_tagged = t.hashing_tag("v1")

    strategy = SnapshotStrategy()
    hash_tag = compute_expr_hash(tagged, strategy=strategy)
    hash_hashing_tag = compute_expr_hash(hashing_tagged, strategy=strategy)

    assert hash_tag != hash_hashing_tag


def test_plain_tag_hash_unchanged(t):
    """Plain tag hash is same as untagged (backward compat)."""
    tagged = t.tag("v1")

    strategy = SnapshotStrategy()
    hash_tagged = compute_expr_hash(tagged, strategy=strategy)
    hash_plain = compute_expr_hash(t, strategy=strategy)

    assert hash_tagged == hash_plain


def test_hashing_tag_is_tag_subclass(t):
    """HashingTag IS-A Tag."""
    expr = t.hashing_tag("v1")
    node = expr.op()
    assert isinstance(node, HashingTag)
    assert isinstance(node, Tag)


def test_find_by_expr_hash_works_for_hashing_tag():
    """find_by_expr_hash can locate HashingTag nodes."""
    batting = xo.examples.batting.fetch()
    ht = batting.hashing_tag("v1")

    strategy = SnapshotStrategy()
    ht_hash = compute_expr_hash(ht, strategy=strategy)

    found = find_by_expr_hash(ht, ht_hash, typs=(HashingTag,))
    assert found is not None
    assert isinstance(found, HashingTag)


def test_yaml_roundtrip_preserves_hashing_tag(t, compiler):
    """YAML roundtrip preserves HashingTag."""
    expr = t.hashing_tag("v1", extra="data")
    yaml_dict = compiler.to_yaml(expr)

    # Check the op type in the YAML
    node_ref = yaml_dict["expression"][RefEnum.node_ref]
    expression = yaml_dict["definitions"][RegistryEnum.nodes][node_ref]
    assert expression["op"] == "HashingTag"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert isinstance(roundtrip_expr.op(), HashingTag)
    assert roundtrip_expr.op().metadata == expr.op().metadata
    assert roundtrip_expr.schema() == expr.schema()


def test_yaml_snapshot_hash_stored(t, compiler):
    """snapshot_hash is stored after YAML serialization for HashingTag."""
    expr = t.hashing_tag("v1")
    yaml_dict = compiler.to_yaml(expr)

    node_ref = yaml_dict["expression"][RefEnum.node_ref]
    expression = yaml_dict["definitions"][RegistryEnum.nodes][node_ref]

    assert "snapshot_hash" in expression


def test_nested_tag_outer_stripped_inner_preserved(t):
    """expr.hashing_tag('a').tag('b') — outer Tag stripped, inner HashingTag preserved."""
    inner = t.hashing_tag("a")
    outer = inner.tag("b")

    strategy = SnapshotStrategy()
    # outer tag is stripped, so hash should equal inner hashing_tag hash
    hash_outer = compute_expr_hash(outer, strategy=strategy)
    hash_inner = compute_expr_hash(inner, strategy=strategy)

    assert hash_outer == hash_inner


def test_nested_hashing_tags_both_preserved(t):
    """expr.hashing_tag('a').hashing_tag('b') — both are preserved, distinct from single."""
    single = t.hashing_tag("a")
    double = t.hashing_tag("a").hashing_tag("b")

    strategy = SnapshotStrategy()
    hash_single = compute_expr_hash(single, strategy=strategy)
    hash_double = compute_expr_hash(double, strategy=strategy)

    assert hash_single != hash_double
