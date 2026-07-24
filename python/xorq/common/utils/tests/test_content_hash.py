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
    nodes = dict(yaml_dict["definitions"]["nodes"])
    return {
        dict(v)["snapshot_hash"] for v in nodes.values() if "snapshot_hash" in dict(v)
    }


@pytest.fixture
def t() -> ibis.Expr:
    return ibis.table({"a": "int64", "b": "string"}, name="test_table")


# --- cross-caller agreement (the shared-identity contract) -------------------


@pytest.mark.parametrize(
    ("build", "node_type"),
    [
        pytest.param(lambda t: t.tag("v1", extra="x"), Tag, id="tag"),
        pytest.param(lambda t: t.hashing_tag("v1"), HashingTag, id="hashing_tag"),
        pytest.param(lambda t: t.filter(t.a > 1), ops.Filter, id="filter-default"),
    ],
)
def test_content_hash_agrees_across_callers(
    t: ibis.Expr, build: Any, node_type: Any
) -> None:
    """A node keys identically via ibis_yaml (expr.yaml) and content_hash.

    The module is the single source of truth for both callers, so every node's
    content_hash must appear as a snapshot_hash in the serialized artifact.
    """
    expr = build(t)
    yaml_hashes = _yaml_snapshot_hashes(expr)
    nodes = list(walk_nodes(node_type, expr))
    assert nodes
    for node in nodes:
        assert content_hash(node) in yaml_hashes


def test_join_reference_agrees_across_callers() -> None:
    t1 = ibis.table({"a": "int64", "k": "int64"}, name="t1")
    t2 = ibis.table({"b": "int64", "k": "int64"}, name="t2")
    expr = t1.join(t2, [("k", "k")])
    yaml_hashes = _yaml_snapshot_hashes(expr)
    nodes = list(walk_nodes(ops.JoinReference, expr))
    assert nodes
    for node in nodes:
        assert content_hash(node) in yaml_hashes


def test_read_agrees_across_callers(tmp_path: Path) -> None:
    path = tmp_path / "x.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(path)
    con = xo.connect()
    expr = xo.deferred_read_parquet(path, con, table_name="x").filter(lambda t: t.a > 1)
    yaml_hashes = _yaml_snapshot_hashes(expr)
    reads = list(walk_nodes(Read, expr))
    assert reads
    for node in reads:
        assert content_hash(node) in yaml_hashes


# --- per-branch contract -----------------------------------------------------


def test_schema_hashes_directly(t: ibis.Expr) -> None:
    schema = t.op().schema
    assert isinstance(schema, Schema)
    # Schema has no to_expr(); it is hashed directly and deterministically.
    assert content_hash(schema) == content_hash(schema)


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
