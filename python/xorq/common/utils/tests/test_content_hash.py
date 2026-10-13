from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from toolz import identity

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching import ParquetCache
from xorq.common.utils.content_hash import content_hash
from xorq.common.utils.node_utils import walk_nodes
from xorq.expr import api, udf
from xorq.expr.relations import CacheTag, HashingTag, Read, Tag
from xorq.ibis_yaml.common import TranslationContext
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
            {"c607b67f7696ab0e702d634506d5a7f3"},
            id="tag",
        ),
        pytest.param(
            lambda t: t.hashing_tag("v1"),
            HashingTag,
            {"9c7fbf834563e34fecb7a507c0101816"},
            id="hashing_tag",
        ),
        pytest.param(
            lambda t: t.filter(t.a > 1),
            ops.Filter,
            {"9f545334703480278326142203c7fb80"},
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
        "95360b4fa7c26c0fd2495b68be0faada",
        "f04458fa3d925cf54baf3ce48da6fafb",
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


def _filter_chain(n: int) -> ibis.Expr:
    t = ibis.table({"a": "int64", "k": "int64"}, name="t")
    for i in range(n):
        t = t.filter(t.a > i)
    return t


def test_tag_is_part_of_its_parents_hash(t: ibis.Expr) -> None:
    # a tagged relation and its untagged twin are different nodes with
    # different serialized forms, so they get different refs
    assert content_hash(t.tag("v1").filter(lambda s: s.a > 1).op()) != content_hash(
        t.filter(t.a > 1).op()
    )


def test_content_hash_compiles_no_sql(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("content_hash must not compile SQL")

    monkeypatch.setattr(api, "to_sql", refuse)
    expr = _filter_chain(20)
    assert all(content_hash(node) for node in walk_nodes(ops.Relation, expr))


def test_shared_context_agrees_with_standalone_calls() -> None:
    expr = _filter_chain(12)
    context = TranslationContext()
    assert all(
        content_hash(node, context) == content_hash(node)
        for node in walk_nodes(ops.Relation, expr)
    )
    assert content_hash(expr.op(), context) in {
        content_hash(node) for node in walk_nodes(ops.Relation, expr)
    }


def test_any_node_can_be_hashed_first() -> None:
    # a JoinLink is serialized inline, not registered; it hashes like any node
    con = xo.connect()
    t = con.create_table("t", pd.DataFrame({"x": [1.0]}))
    u = con.create_table("u", pd.DataFrame({"x": [1.0]}))
    link = next(iter(t.join(u, "x").op().rest))
    assert content_hash(link) == content_hash(link)


def test_cache_tag_hashes_the_same_for_expr_and_node_payloads(tmp_path: Path) -> None:
    con = xo.connect()
    t = con.create_table("t", pd.DataFrame({"a": [1]}))
    cache = ParquetCache.from_kwargs(relative_path=tmp_path, source=con)
    as_expr = CacheTag(schema=t.schema(), parent=t.op(), uncached=t, cache=cache)
    as_node = CacheTag(schema=t.schema(), parent=t.op(), uncached=t.op(), cache=cache)
    assert content_hash(as_expr) == content_hash(as_node)


def test_scalars_without_a_dasher_rule_hash(t: ibis.Expr) -> None:
    dated = ibis.table({"d": "date"}, name="d")
    march = dated.filter(dated.d > ibis.date(2020, 3, 1)).op()
    april = dated.filter(dated.d > ibis.date(2020, 4, 1)).op()
    assert content_hash(march) != content_hash(april)
    by_day = dated.mutate(w=dated.d.truncate("D")).op()
    by_month = dated.mutate(w=dated.d.truncate("M")).op()
    assert content_hash(by_day) != content_hash(by_month)


def test_expr_scalar_udf_hashes_its_computed_input() -> None:
    base = xo.memtable(pd.DataFrame({"x": [1.0, 2.0]}), name="base")

    def pipeline(values: list[float]) -> ibis.Expr:
        other = xo.memtable(pd.DataFrame({"x": values}), name="other")

        @udf.agg.pandas_df(schema=other.schema(), return_type=dt.float64, name="my_sum")
        def my_sum(frame):
            return frame["x"].astype(float).sum()

        predict = udf.make_pandas_expr_udf(
            computed_kwargs_expr=my_sum.on_expr(other).name("my_sum").as_table(),
            fn=lambda value, frame, **kw: frame["x"] + float(value),
            schema=base.schema(),
            name="add_sum",
            return_type=dt.float64,
            post_process_fn=identity,
        )
        return base.mutate(out=predict.on_expr(base)).as_table()

    assert content_hash(pipeline([1.0, 2.0]).op()) != content_hash(
        pipeline([10.0, 20.0]).op()
    )


def test_builtin_udfs_hash_the_sql_function_name() -> None:
    t = ibis.table({"x": "float64"}, name="t")

    def stub(x: float) -> float: ...

    as_sqrt = t.mutate(y=udf.scalar.builtin(stub, name="sqrt")(t.x)).op()
    as_abs = t.mutate(y=udf.scalar.builtin(stub, name="abs")(t.x)).op()
    assert content_hash(as_sqrt) != content_hash(as_abs)
