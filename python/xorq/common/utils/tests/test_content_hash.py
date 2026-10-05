from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import xorq.api as xo
import xorq.expr.api as api
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching.strategy import SnapshotStrategy
from xorq.common.utils.content_hash import (
    PLACEHOLDER_PREFIX,
    ContentHasher,
    content_hash,
)
from xorq.common.utils.graph_utils import bfs
from xorq.common.utils.node_utils import walk_nodes
from xorq.expr.relations import HashingTag, Read, Tag
from xorq.ibis_yaml.compiler import YamlExpressionTranslator
from xorq.vendor import ibis
from xorq.vendor.ibis.expr.schema import Schema


def _yaml_snapshot_hashes(expr: ibis.Expr) -> set[str]:
    """The set of ``snapshot_hash`` values ibis_yaml writes into expr.yaml."""
    yaml_dict = YamlExpressionTranslator().to_yaml(expr)
    return {
        node["snapshot_hash"]
        for node in yaml_dict["definitions"]["nodes"].values()
        if "snapshot_hash" in node
    }


@pytest.fixture
def t() -> ibis.Expr:
    return ibis.table({"a": "int64", "b": "string"}, name="test_table")


# --- golden identity + ibis_yaml wiring --------------------------------------
#
# Golden byte values, NOT `content_hash(node) in yaml_hashes`: the latter is
# tautological now that register_node writes snapshot_hash = content_hash(node)
# (both sides are the same function, so it passes for any implementation).
# Hardcoding the expected hash pins the actual bytes -- a change to content_hash
# fails here even if the serializer changes in lockstep -- and asserting the
# same byte appears in expr.yaml still proves ibis_yaml is wired to this helper.
# Nodes here are filesystem-independent so the golden values are stable across
# machines (Read embeds an absolute path; it is covered structurally below).


@pytest.mark.parametrize(
    ("build", "node_type", "expected"),
    [
        pytest.param(
            lambda t: t.tag("v1", extra="x"),
            Tag,
            {"885320479f53e24984215603ffcc7951"},
            id="tag",
        ),
        pytest.param(
            lambda t: t.hashing_tag("v1"),
            HashingTag,
            {"0871cbb8ecd40bc0280f8f1bc80ca78b"},
            id="hashing_tag",
        ),
        pytest.param(
            lambda t: t.filter(t.a > 1),
            ops.Filter,
            {"c212583de31a5b64c32d031b0e08a0ad"},
            id="filter-default",
        ),
    ],
)
def test_golden_hash_matches_and_is_serialized(
    t: ibis.Expr, build: Any, node_type: Any, expected: set[str]
) -> None:
    expr = build(t)
    nodes = list(walk_nodes(node_type, expr))
    assert nodes
    assert {content_hash(node) for node in nodes} == expected
    # ibis_yaml must emit the same bytes (proves the serializer is wired here).
    assert expected <= _yaml_snapshot_hashes(expr)


def test_golden_join_reference_hash_and_is_serialized() -> None:
    t1 = ibis.table({"a": "int64", "k": "int64"}, name="t1")
    t2 = ibis.table({"b": "int64", "k": "int64"}, name="t2")
    expr = t1.join(t2, [("k", "k")])
    expected = {
        "39c58d8de1af85ded58718f177ba6a6d",
        "a1deb7a2c301bc469849c3cd9e22d29e",
    }
    nodes = list(walk_nodes(ops.JoinReference, expr))
    assert {content_hash(node) for node in nodes} == expected
    assert expected <= _yaml_snapshot_hashes(expr)


def test_read_hash_is_serialized(tmp_path: Path) -> None:
    """Read hash embeds an absolute path, so pin it structurally, not by value."""
    path = tmp_path / "x.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(path)
    con = xo.connect()
    expr = xo.deferred_read_parquet(path, con, table_name="x").filter(lambda t: t.a > 1)
    (read,) = walk_nodes(Read, expr)
    assert content_hash(read) in _yaml_snapshot_hashes(expr)


# --- per-branch contract -----------------------------------------------------


def test_schema_hashes_directly(t: ibis.Expr) -> None:
    schema = t.op().schema
    assert isinstance(schema, Schema)
    # Schema has no to_expr(); it is hashed directly to a stable golden value.
    assert content_hash(schema) == "91bef2a71ad5c557d2712bfe058463c9"
    # A different schema must hash differently (guards against a constant hash).
    other = ibis.table({"a": "int64", "b": "float64"}, name="test_table").op().schema
    assert content_hash(schema) != content_hash(other)


def test_plain_tag_and_hashing_tag_hash_differently(t: ibis.Expr) -> None:
    """A plain Tag and a HashingTag over the same parent are distinct nodes."""
    plain = t.tag("v1").op()
    hashing = t.hashing_tag("v1").op()
    assert content_hash(plain) != content_hash(hashing)


def test_tag_metadata_changes_hash(t: ibis.Expr) -> None:
    assert content_hash(t.tag("v1").op()) != content_hash(t.tag("v2").op())


def test_read_name_distinguishes_identical_content(tmp_path: Path) -> None:
    """Two Reads with identical content but different names hash differently."""
    path = tmp_path / "x.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(path)
    con = xo.connect()
    (r1,) = walk_nodes(Read, xo.deferred_read_parquet(path, con, table_name="one"))
    (r2,) = walk_nodes(Read, xo.deferred_read_parquet(path, con, table_name="two"))
    assert content_hash(r1) != content_hash(r2)


def test_content_hash_is_deterministic(t: ibis.Expr) -> None:
    expr = t.filter(t.a > 1)
    node = expr.op()
    assert content_hash(node) == content_hash(node)


# --- incremental hashing (#2351) ---------------------------------------------


def _chain(n: int) -> ibis.Expr:
    t = ibis.table({"a": "int64"}, name="t")
    for i in range(n):
        t = t.mutate(**{f"m{i}": t.a + i}) if i % 2 == 0 else t.filter(t.a > -i)
    return t


def _stack_depth() -> int:
    frame, depth = sys._getframe(1), 0
    while frame is not None:
        frame, depth = frame.f_back, depth + 1
    return depth


def test_plain_tag_is_transparent_to_its_parents_hash(t: ibis.Expr) -> None:
    """A plain Tag under a Filter does not change the Filter's hash (untagged semantics)."""
    tagged = t.tag("v1").filter(lambda s: s.a > 1).op()
    plain = t.filter(t.a > 1).op()
    assert content_hash(tagged) == content_hash(plain)
    # ...but a HashingTag does.
    hashing = t.hashing_tag("v1").filter(lambda s: s.a > 1).op()
    assert content_hash(hashing) != content_hash(plain)


def test_content_hasher_agrees_with_standalone_calls() -> None:
    """One memoized hasher over the root hashes every relation exactly as a fresh call does."""
    expr = _chain(12)
    hasher = ContentHasher()
    root_hash = hasher(expr.op())
    relations = tuple(walk_nodes(ops.Relation, expr))
    assert set(relations) <= set(hasher.hashes)
    assert hasher.hashes[expr.op()] == root_hash
    assert all(hasher.hashes[node] == content_hash(node) for node in relations)


def _filter_chain(n: int) -> ibis.Expr:
    # filters only: a mutate chain re-lists every column, so its own graph
    # (not just the compiled subtree) grows with position in the chain
    t = ibis.table({"a": "int64", "k": "int64"}, name="t")
    for i in range(n):
        t = t.filter(t.a > i)
    return t


def _joined_filter_chain(n: int) -> ibis.Expr:
    u = ibis.table({"b": "int64", "k": "int64"}, name="u")
    return _filter_chain(n).join(u, "k")


@pytest.mark.parametrize("build", [_filter_chain, _joined_filter_chain])
def test_content_hash_compiles_one_level_of_sql_per_relation(
    monkeypatch, build
) -> None:
    """The op graph handed to the SQL compiler per relation stays bounded as the chain grows.

    Before #2351 each relation compiled its whole subtree, so hashing all N
    relations of a chain cost O(N^2) SQL compiles. A join reaches its inputs
    through ``JoinReference`` wrappers, which must not reopen the subtree.
    """
    sizes: list[int] = []
    original = api.to_sql

    def counting_to_sql(expr, *args, **kwargs):
        sizes.append(len(bfs(expr.op())))
        return original(expr, *args, **kwargs)

    monkeypatch.setattr(api, "to_sql", counting_to_sql)

    def max_compiled_size(n: int) -> int:
        sizes.clear()
        hasher = ContentHasher()
        for node in walk_nodes(ops.Relation, build(n)):
            hasher(node)
        return max(sizes)

    assert max_compiled_size(10) == max_compiled_size(40)


_SEEN_DATABASETABLES: list[str] = []
_ORIGINAL_NORMALIZE_DATABASETABLE = SnapshotStrategy.normalize_databasetable


def _recording_normalize_databasetable(dt):
    # module-level on purpose: dasher refuses closures as rule normalizers
    _SEEN_DATABASETABLES.append(dt.name)
    return _ORIGINAL_NORMALIZE_DATABASETABLE(dt)


def test_placeholders_skip_databasetable_normalization(monkeypatch) -> None:
    """A placeholder names an already-hashed child; it must not be normalized
    like a real backend table (Redshift resolves an unqualified table's schema
    with a query per table, which would be one round trip per relation)."""
    monkeypatch.setattr(
        SnapshotStrategy,
        "normalize_databasetable",
        staticmethod(_recording_normalize_databasetable),
    )
    _SEEN_DATABASETABLES.clear()
    con = xo.connect()
    t = con.create_table("real_table", pd.DataFrame({"a": [1, 2, 3]}))
    for i in range(5):
        t = t.filter(t.a > i)
    ContentHasher()(t.op())
    assert "real_table" in _SEEN_DATABASETABLES
    assert not [n for n in _SEEN_DATABASETABLES if n.startswith(PLACEHOLDER_PREFIX)]


def test_content_hash_depth_does_not_grow_with_chain_length() -> None:
    expr = _chain(80)
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(_stack_depth() + 150)
    try:
        hashed = content_hash(expr.op())
    finally:
        sys.setrecursionlimit(limit)
    assert hashed == content_hash(expr.op())
