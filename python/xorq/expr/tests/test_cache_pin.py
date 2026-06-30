from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.common.exceptions import IntegrityError
from xorq.common.utils.graph_utils import find_all_sources, walk_nodes
from xorq.common.utils.name_utils import get_uid_prefix
from xorq.common.utils.provenance_utils import get_expr_hash
from xorq.expr.relations import CachedNode, CacheTag, Read, RemoteTable
from xorq.ibis_yaml.compiler import canonicalize_expr
from xorq.vendor.ibis.expr.operations.relations import InMemoryTable


def make_cached(tmp_path: Path) -> tuple:
    con = xo.connect()
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    t = con.register(df, "tbl")
    cache = ParquetCache.from_kwargs(relative_path=tmp_path, source=con)
    return t.cache(cache), cache


def test_pin_replaces_cached_node_with_cache_tag(tmp_path: Path) -> None:
    cached, _ = make_cached(tmp_path)
    cached.execute()

    pinned = cached.ls.pin()

    assert walk_nodes((CacheTag,), pinned)
    assert not walk_nodes((CachedNode,), pinned)
    assert walk_nodes((Read,), pinned)
    assert pinned.execute().equals(cached.execute())


def test_pin_requires_materialized_cache(tmp_path: Path) -> None:
    cached, _ = make_cached(tmp_path)

    with pytest.raises(IntegrityError, match="unmaterialized"):
        cached.ls.pin()


def test_pinned_reads_directly_and_does_not_recompute(tmp_path: Path) -> None:
    cached, cache = make_cached(tmp_path)
    cached.execute()
    pinned = cached.ls.pin()

    # remove the cache artifact: a pinned expr reads the file directly, so it
    # must fail, whereas the unpinned form would simply recompute.
    cache.drop(cached.ls.uncached_one)

    with pytest.raises(ValueError):
        pinned.execute()

    # unpin restores the recompute-capable form
    assert (
        pinned.ls.unpin()
        .execute()
        .equals(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    )


def test_unpin_is_inverse_of_pin(tmp_path: Path) -> None:
    cached, _ = make_cached(tmp_path)
    cached.execute()

    unpinned = cached.ls.pin().ls.unpin()

    assert walk_nodes((CachedNode,), unpinned)
    assert not walk_nodes((CacheTag,), unpinned)
    # structural equivalence (CachedNode == is unreliable: it stores parent as
    # an Any-typed Expr, so compare via tokenized + structural op equality)
    assert unpinned.ls.tokenized == cached.ls.tokenized
    assert unpinned.op().parent.op() == cached.op().parent.op()


def test_pin_changes_build_hash_but_not_result(tmp_path: Path) -> None:
    cached, _ = make_cached(tmp_path)
    cached.execute()
    pinned = cached.ls.pin()

    # freeze-time pinning is intentionally NOT cache-hash-neutral
    assert get_expr_hash(pinned) != get_expr_hash(cached)
    assert pinned.execute().equals(cached.execute())


def test_pin_is_noop_without_cached_nodes() -> None:
    con = xo.connect()
    t = con.register(pd.DataFrame({"a": [1, 2]}), "tbl")
    expr = t.filter(t.a > 1)

    assert expr.ls.pin().ls.tokenized == expr.ls.tokenized


def test_pin_unpin_multi_engine_remote_table(tmp_path: Path) -> None:
    # Cache across engines: the CachedNode.parent is a RemoteTable, so pinning
    # must take the `remote_expr` branch when locating the materialized cache.
    con = xo.connect()
    con2 = xo.connect()
    t = con.register(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), "tbl")
    cache = ParquetCache.from_kwargs(relative_path=tmp_path, source=con2)
    cached = t.into_backend(con2).cache(cache)
    # guard: this test is only meaningful if the cached parent is a RemoteTable
    assert isinstance(cached.op().parent.op(), RemoteTable)
    cached.execute()

    pinned = cached.ls.pin()
    assert walk_nodes((CacheTag,), pinned)
    assert not walk_nodes((CachedNode,), pinned)
    assert pinned.execute().equals(cached.execute())

    unpinned = pinned.ls.unpin()
    assert walk_nodes((CachedNode,), unpinned)
    assert not walk_nodes((CacheTag,), unpinned)
    # tokenized is the reliable structural comparison here: RemoteTable.__eq__
    # is unreliable (it stores remote_expr as an Any-typed Expr), so a direct
    # `parent.op() ==` check on a RemoteTable parent gives false negatives.
    assert unpinned.ls.tokenized == cached.ls.tokenized


def test_pin_unpin_nested_caches(tmp_path: Path) -> None:
    # A cache stacked on a cache: the inner CachedNode lives under the outer's
    # opaque `uncached` payload, so pin/unpin must descend into it.
    con = xo.connect()
    t = con.register(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), "tbl")
    inner_cache = ParquetCache.from_kwargs(relative_path=tmp_path / "inner", source=con)
    outer_cache = ParquetCache.from_kwargs(relative_path=tmp_path / "outer", source=con)
    cached = t.cache(inner_cache).filter(xo._.a > 1).cache(outer_cache)
    # executing the outer cache materializes the inner one along the way
    cached.execute()

    pinned = cached.ls.pin()
    assert len(walk_nodes((CacheTag,), pinned)) == 2
    assert not walk_nodes((CachedNode,), pinned)
    assert pinned.execute().equals(cached.execute())

    unpinned = pinned.ls.unpin()
    assert len(walk_nodes((CachedNode,), unpinned)) == 2
    assert not walk_nodes((CacheTag,), unpinned)
    assert unpinned.ls.tokenized == cached.ls.tokenized


def test_pinned_expr_is_a_frozen_read_not_a_cache(tmp_path: Path) -> None:
    # Model A contract: a pin is a frozen read, so cache-introspection accessors
    # deliberately do not see it. unpin() is the gate back to cache semantics.
    cached, _ = make_cached(tmp_path)
    cached.execute()
    assert cached.ls.is_cached
    assert cached.ls.has_cached

    pinned = cached.ls.pin()

    assert not pinned.ls.is_cached
    assert not pinned.ls.has_cached
    assert pinned.ls.cached_nodes == ()
    # .ls.uncached strips live caches; a pin has none, so it is a no-op
    assert pinned.ls.uncached.ls.tokenized == pinned.ls.tokenized

    # unpin restores cache semantics
    assert pinned.ls.unpin().ls.is_cached


def test_cache_tag_rejects_non_expr_uncached() -> None:
    # `parent` is type-enforced by the inherited Tag annotation, but `uncached`
    # is Any; the __init__ guard catches a swapped/garbage upstream at
    # construction instead of failing obscurely later.
    t = xo.memtable(pd.DataFrame({"a": [1]}))
    schema = t.schema()

    with pytest.raises(IntegrityError, match="uncached must be an Expr or Node"):
        CacheTag(schema=schema, parent=t.op(), uncached=42, cache=None)

    # an Expr is accepted, a Node is accepted, and None (unset) is accepted
    assert CacheTag(schema=schema, parent=t.op(), uncached=t, cache=None) is not None
    assert (
        CacheTag(schema=schema, parent=t.op(), uncached=t.op(), cache=None) is not None
    )
    assert CacheTag(schema=schema, parent=t.op(), uncached=None, cache=None) is not None


def test_find_all_sources_execution_only_prunes_uncached_backend(
    tmp_path: Path,
) -> None:
    # Regression: a pinned read executes only through its frozen `parent`, but
    # its `uncached` payload still references the original upstream backend. The
    # default walk reports both (serialization needs the uncached profiles);
    # connection-selection consumers must pass execution_only=True to avoid
    # reporting backends that aren't actually required to run the pinned expr.
    con = xo.connect()
    con2 = xo.connect()
    t = con.register(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), "tbl")
    cache = ParquetCache.from_kwargs(relative_path=tmp_path, source=con2)
    cached = t.into_backend(con2).cache(cache)
    # guard: the uncached payload must carry a distinct second backend
    assert isinstance(cached.op().parent.op(), RemoteTable)
    cached.execute()
    pinned = cached.ls.pin()

    full = find_all_sources(pinned)
    execution = find_all_sources(pinned, execution_only=True)

    # default walk descends uncached and sees both backends
    assert len(full) == 2
    # execution path runs only through the frozen cache read on con2
    assert tuple(map(id, execution)) == (id(con2),)


def test_pin_does_not_freeze_dag_shared_upstream_leaf(tmp_path: Path) -> None:
    # Regression: a pin's `uncached` payload holds the original upstream. If a
    # leaf there is also referenced by a live, non-pinned part of the same
    # expression (DAG sharing), name canonicalization must NOT skip it as a
    # pinned leaf -- otherwise the live reference keeps its session-random UID
    # name and leaks nondeterminism into the outer expression's build hash.
    con = xo.connect()
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # memtable (not register) so the upstream leaf is a UID-prefixed
    # InMemoryTable -- exactly the name that must be sanitized.
    t = xo.memtable(df, name=None)
    cache = ParquetCache.from_kwargs(relative_path=tmp_path, source=con)
    cached = t.cache(cache)
    cached.execute()
    pinned = cached.ls.pin()

    # `t` is shared: it is both the pin's uncached upstream and the live left
    # side of the join.
    expr = t.join(pinned, "a")[["a"]]
    assert walk_nodes((CacheTag,), expr)

    canonical = canonicalize_expr(expr)
    # No live leaf may retain its auto-generated UID prefix after canonicalization.
    leftover = [
        n.name
        for n in walk_nodes((InMemoryTable, Read), canonical)
        if get_uid_prefix(n.name)
    ]
    assert leftover == [], f"unsanitized UID-prefixed leaves: {leftover}"
