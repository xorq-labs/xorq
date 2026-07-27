from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import xorq.api as xo
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.content_hash import content_hash
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
            {"563943016a2d36c008f1d7129a84ae75"},
            id="tag",
        ),
        pytest.param(
            lambda t: t.hashing_tag("v1"),
            HashingTag,
            {"d3a1cf68f4aaf63eba3c263ad5c8f5b7"},
            id="hashing_tag",
        ),
        pytest.param(
            lambda t: t.filter(t.a > 1),
            ops.Filter,
            {"b82aab366f891829b5c84e042a24ea85"},
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
        "18af0a10f0e57ec2eb54430e77f1566c",
        "7d9e90cea69511544f334134ca3d9716",
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
