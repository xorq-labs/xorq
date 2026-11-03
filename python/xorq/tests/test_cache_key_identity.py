"""Every site that derives a cache key for a ``CachedNode`` must agree.

A ``CachedNode`` is keyed at five independent sites: execution (the artifact
``set_default`` writes), the ``ls`` accessors, the public ``Cache.exists`` /
``Cache.get`` surface, the pin helpers, and the build projection recorded in
``expr_metadata.json``. Nothing else checks that they derive the same key, so
this module pins their agreement per node and the distinctness of keys across
nodes. It asserts agreement, never key literals: key stability is the fix's
close evidence, not this harness's.

GH #2382: a stack of one cache on an equal one (``t.cache(c).cache(c)``) holds
the same rows at every level, so every site keys it as its innermost node and
it writes one artifact, in memory and after a build round trip, where the two
levels hold equal but distinct cache objects. On main the sites disagree
(``Cache.exists`` and ``Cache.get`` key the outer node one level shallower than
execution and the ``ls`` accessors) and a stacked ``SourceCache`` writes its
table twice. Those cases are ``xfail(strict=True)`` while the probe below sees
the bug, and run as ordinary tests once it is gone, so this file is
byte-identical on the fix branch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

import xorq.api as xo
import xorq.vendor.ibis.expr.types as ir
from xorq.caching import (
    ParquetDummySnapshotCache,
    ParquetSnapshotCache,
    SourceCache,
)
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import (
    CachedNode,
    CacheTag,
    _cached_node_materialized,
    pin_cache,
)
from xorq.ibis_yaml.compiler import build_expr, load_expr
from xorq.ibis_yaml.enums import DumpFiles
from xorq.vendor.ibis.backends import BaseBackend


ORIGINAL_ROWS = pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})

CaseBuilder = Callable[[ir.Table, BaseBackend, Path, pytest.FixtureRequest], ir.Table]


def _same_cache_stacking_is_broken() -> bool:
    # Probe for GH #2382; remove it (and the conditional xfail) once the fix
    # lands. No storage path is needed: the dummy cache never
    # touches disk, and the two sites compared here are pure key derivation.
    con = xo.connect()
    t = con.create_table("probe", ORIGINAL_ROWS)
    cache = ParquetDummySnapshotCache.from_kwargs()
    outer = t.cache(cache).cache(cache)
    return cache.calc_key(outer) != outer.ls.get_key()


SAME_CACHE_STACKING_BROKEN = _same_cache_stacking_is_broken()

xfail_same_cache_stacking = pytest.mark.xfail(
    condition=SAME_CACHE_STACKING_BROKEN,
    reason="GH #2382: keying sites disagree on a same-cache stack",
    strict=True,
)


def _make_cache(
    con: BaseBackend, tmp_path: Path, relative_path: str
) -> ParquetSnapshotCache:
    return ParquetSnapshotCache.from_kwargs(
        source=con, base_path=tmp_path / "cache", relative_path=relative_path
    )


def _single(
    t: ir.Table, con: BaseBackend, tmp_path: Path, request: pytest.FixtureRequest
) -> ir.Table:
    return t.cache(_make_cache(con, tmp_path, "parquet"))


def _stacked_same_cache(
    t: ir.Table, con: BaseBackend, tmp_path: Path, request: pytest.FixtureRequest
) -> ir.Table:
    cache = _make_cache(con, tmp_path, "parquet")
    return t.cache(cache).cache(cache)


def _stacked_different_caches(
    t: ir.Table, con: BaseBackend, tmp_path: Path, request: pytest.FixtureRequest
) -> ir.Table:
    inner = _make_cache(con, tmp_path, "inner")
    outer = _make_cache(con, tmp_path, "outer")
    return t.cache(inner).cache(outer)


def _cross_source_parent(
    t: ir.Table, con: BaseBackend, tmp_path: Path, request: pytest.FixtureRequest
) -> ir.Table:
    pg = request.getfixturevalue("pg")
    return t.into_backend(pg).cache(_make_cache(con, tmp_path, "parquet"))


@pytest.fixture
def expr(request: pytest.FixtureRequest, tmp_path: Path) -> ir.Table:
    con = xo.connect()
    t = con.create_table("t", ORIGINAL_ROWS)
    builder: CaseBuilder = request.param
    return builder(t, con, tmp_path, request)


@pytest.fixture
def observed_set_default_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[ParquetSnapshotCache, str]]:
    """Capture every key ``set_default`` computes during an execute."""
    observed = []
    original = ParquetSnapshotCache.set_default

    def capturing(
        self: ParquetSnapshotCache,
        expr: ir.Expr,
        default: object,
        parquet_metadata: dict | None = None,
    ) -> object:
        observed.append((self, self.calc_key(expr)))
        return original(self, expr, default, parquet_metadata=parquet_metadata)

    monkeypatch.setattr(ParquetSnapshotCache, "set_default", capturing)
    return observed


def _artifact_keys(cache: ParquetSnapshotCache) -> frozenset[str]:
    return frozenset(path.stem for path in cache.storage.path.glob("*.parquet"))


def _pinned_key(node: CachedNode) -> str:
    # pin_cache freezes the whole subtree; the root of the result is the tag
    # for ``node`` itself, whose ``parent`` is the read of its artifact
    tag = pin_cache(node.to_expr()).op()
    assert isinstance(tag, CacheTag)
    return tag.parent.name


def _keys_by_site(node: CachedNode) -> dict[str, str]:
    """One key per keying site for ``node``; they must all be the same string."""
    cache = node.cache
    node_expr = node.to_expr()
    return {
        "cache.calc_key(node)": cache.calc_key(node_expr),
        "ls.get_key": node_expr.ls.get_key(),
        "ls.get_cache_path": node_expr.ls.get_cache_path().stem,
        "pin_cache tag.parent.name": _pinned_key(node),
    }


def _parametrize_cases(
    same_cache_marks: tuple[pytest.MarkDecorator, ...] = (),
) -> pytest.MarkDecorator:
    # each case with the number of distinct keys its nodes hold: a same-cache
    # stack is one key, different caches one each
    return pytest.mark.parametrize(
        "expr,key_count",
        (
            pytest.param(_single, 1, id="single"),
            pytest.param(
                _stacked_same_cache,
                1,
                id="stacked_same_cache",
                marks=same_cache_marks,
            ),
            pytest.param(_stacked_different_caches, 2, id="stacked_different_caches"),
            pytest.param(
                _cross_source_parent,
                1,
                id="cross_source_parent",
                marks=pytest.mark.postgres,
            ),
        ),
        indirect=["expr"],
    )


@_parametrize_cases(same_cache_marks=(xfail_same_cache_stacking,))
def test_keying_sites_agree_per_node(
    expr: ir.Table,
    key_count: int,
    observed_set_default_keys: list[tuple[ParquetSnapshotCache, str]],
) -> None:
    nodes = expr.ls.cached_nodes
    assert nodes

    expr.execute()

    # every site keys each node identically
    expected_keys = tuple(node.cache.calc_key(node.to_expr()) for node in nodes)
    for node, expected in zip(nodes, expected_keys):
        node_expr = node.to_expr()
        by_site = _keys_by_site(node)
        assert set(by_site.values()) == {expected}, by_site
        assert node.cache.get(node_expr).name == expected
        assert node_expr.ls.cache_exists() is True
        assert node_expr.ls.get_cache_path().exists()
        assert node.cache.exists(node_expr) is True
        assert _cached_node_materialized(node) is True

    # nodes share a key only within a same-cache stack
    assert len(set(expected_keys)) == key_count

    # execution wrote exactly the artifacts the nodes key to, per cache, and
    # computed no other key on the way
    for cache in {node.cache for node in nodes}:
        keys_for_cache = {
            expected
            for node, expected in zip(nodes, expected_keys)
            if node.cache == cache
        }
        assert _artifact_keys(cache) == keys_for_cache
        observed_for_cache = {
            key
            for observed_cache, key in observed_set_default_keys
            if observed_cache == cache
        }
        assert observed_for_cache == keys_for_cache

    # pinning the whole expression yields one tag per node reading its artifact
    tags = walk_nodes((CacheTag,), pin_cache(expr))
    assert tuple(tag.parent.name for tag in tags) == expected_keys
    assert pin_cache(expr).execute().equals(expr.execute())


# Not xfailed for the same-cache stack: the projection and ``calc_key`` are one
# function, so they agree on main too, and an XPASS would be the wrong signal.
@_parametrize_cases()
def test_build_projection_agrees_with_root_node(
    expr: ir.Table, key_count: int, tmp_path: Path
) -> None:
    (root, *_) = expr.ls.cached_nodes
    expected = root.cache.calc_key(root.to_expr())

    assert expr.ls.metadata.projected_cache_key.key == expected

    build_path = build_expr(
        expr, builds_dir=tmp_path / "builds", cache_dir=root.cache.storage.base_path
    )
    recorded = json.loads((build_path / DumpFiles.expr_metadata).read_text())
    assert recorded["cache_keys"]["key"] == expected


@xfail_same_cache_stacking
def test_stacked_same_cache_writes_one_artifact_after_a_build(tmp_path: Path) -> None:
    """A build round trip gives each level its own, equal, cache object; the
    stack still writes one artifact, the one the build recorded."""
    con = xo.connect()
    t = con.create_table("t", ORIGINAL_ROWS)
    cache = _make_cache(con, tmp_path, "parquet")
    build_path = build_expr(t.cache(cache).cache(cache), builds_dir=tmp_path / "builds")
    loaded = load_expr(build_path, cache_dir=cache.storage.base_path)

    assert loaded.execute().equals(ORIGINAL_ROWS)
    recorded = json.loads((build_path / DumpFiles.expr_metadata).read_text())
    assert _artifact_keys(cache) == {recorded["cache_keys"]["key"]}


@xfail_same_cache_stacking
def test_source_cache_stacked_on_itself_executes() -> None:
    con = xo.connect()
    t = con.create_table("t", ORIGINAL_ROWS)
    cache = SourceCache.from_kwargs(source=xo.duckdb.connect())
    result = t.cache(cache).cache(cache).execute()
    assert result.sort_values("x").reset_index(drop=True).equals(ORIGINAL_ROWS)
