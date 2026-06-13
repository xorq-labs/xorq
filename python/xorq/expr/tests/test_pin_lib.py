from __future__ import annotations

import shutil

import pandas as pd
import pytest

import xorq.api as xo
from xorq.caching import ParquetSnapshotCache, SourceCache
from xorq.common.utils.dasher import tokenize
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.pin_lib import (
    PinInfo,
    _pin_caches,
    pin_caches,
    pin_infos,
    pinned_tag_nodes,
    unpin,
    verify_pinned,
)
from xorq.expr.relations import CachedNode, Read
from xorq.ibis_yaml.compiler import build_expr, load_expr


@pytest.fixture(scope="module")
def data_path(tmp_path_factory):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    path = tmp_path_factory.mktemp("data") / "data.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def cache(tmp_path):
    return ParquetSnapshotCache.from_kwargs(base_path=tmp_path / "cache")


@pytest.fixture
def cached_expr(data_path, cache):
    con = xo.connect()
    t = deferred_read_parquet(data_path, con, "data")
    return t.mutate(c=t.a + t.b).cache(cache=cache)


def pinned_read(expr):
    (tag,) = pinned_tag_nodes(expr)
    read = tag.parent
    assert isinstance(read, Read)
    return read


def test_pin_caches_results_match(cached_expr):
    expected = cached_expr.execute()
    pinned = pin_caches(cached_expr)
    pd.testing.assert_frame_equal(expected, pinned.execute())
    # the cache (and its compute) are gone from the pinned expression
    assert not walk_nodes(CachedNode, pinned)
    read = pinned_read(pinned)
    assert dict(read.read_kwargs).get("relocate")


def test_pin_no_relocate(cached_expr):
    expected = cached_expr.execute()
    pinned = pin_caches(cached_expr, relocate=False)
    read = pinned_read(pinned)
    assert "relocate" not in dict(read.read_kwargs)
    pd.testing.assert_frame_equal(expected, pinned.execute())


def test_pin_keys_subset(data_path, cache):
    con = xo.connect()
    t = deferred_read_parquet(data_path, con, "data")
    expr = (
        t.mutate(z=t.a)
        .cache(cache=cache)
        .union(t.mutate(z=t.b.cast("int64")).cache(cache=cache))
    )
    nodes = walk_nodes(CachedNode, expr)
    assert len(nodes) == 2
    (key, _) = (node.cache.calc_key(node.to_expr()) for node in nodes)
    pinned = pin_caches(expr, keys=(key,))
    assert len(pinned_tag_nodes(pinned)) == 1
    assert len(walk_nodes(CachedNode, pinned)) == 1
    (info,) = pin_infos(pinned)
    assert info.cache_key == key


def test_pin_unknown_key_raises(cached_expr):
    with pytest.raises(ValueError, match="no cached node matches"):
        pin_caches(cached_expr, keys=("no-such-key",))


def test_pin_skips_source_storage(data_path):
    con = xo.connect()
    t = deferred_read_parquet(data_path, con, "data")
    expr = t.mutate(z=t.a).cache(cache=SourceCache.from_kwargs(source=con))
    pinned = pin_caches(expr)
    # a non-parquet cache is left in place by default
    assert not pinned_tag_nodes(pinned)
    assert len(walk_nodes(CachedNode, pinned)) == 1


def test_pin_keyed_source_storage_raises(data_path):
    con = xo.connect()
    t = deferred_read_parquet(data_path, con, "data")
    expr = t.mutate(z=t.a).cache(cache=SourceCache.from_kwargs(source=con))
    (node,) = walk_nodes(CachedNode, expr)
    key = node.cache.calc_key(node.to_expr())
    with pytest.raises(ValueError, match="not a local"):
        pin_caches(expr, keys=(key,))


def test_pin_idempotent(cached_expr):
    pinned = pin_caches(cached_expr)
    repinned = pin_caches(pinned)
    assert pin_infos(pinned) == pin_infos(repinned)
    assert tokenize(pinned) == tokenize(repinned)


def test_unpin_restores_cache(cached_expr):
    expected = cached_expr.execute()
    pinned = pin_caches(cached_expr)
    unpinned = unpin(pinned)
    # the pin tag is gone and the cached compute is back
    assert not pinned_tag_nodes(unpinned)
    assert len(walk_nodes(CachedNode, unpinned)) == 1
    pd.testing.assert_frame_equal(expected, unpinned.execute())


def test_unpin_noop_when_unpinned(cached_expr):
    assert unpin(cached_expr) is cached_expr


def test_unpin_survives_build_roundtrip(cached_expr, tmp_path):
    # the recipe is self-contained: a loaded build unpins without the source
    expected = cached_expr.execute()
    build_path = build_expr(pin_caches(cached_expr), builds_dir=tmp_path / "builds")
    loaded = load_expr(build_path)
    unpinned = unpin(loaded)
    assert not pinned_tag_nodes(unpinned)
    assert len(walk_nodes(CachedNode, unpinned)) == 1
    pd.testing.assert_frame_equal(expected, unpinned.execute())


def test_ls_unpin(cached_expr):
    pinned = cached_expr.ls.pin_caches()
    assert not pinned.ls.unpin().ls.pinned_tags


def _nested_caches(data_path, cache):
    con = xo.connect()
    t = deferred_read_parquet(data_path, con, "data")
    inner = t.mutate(z=t.a + t.b).cache(cache=cache)
    outer = inner.mutate(w=inner.z + 1).cache(cache=cache)
    keys = [n.cache.calc_key(n.to_expr()) for n in walk_nodes(CachedNode, outer)]
    return outer, keys  # keys[0] outer, keys[1] inner


def test_pin_nested_keys_no_spurious_error(data_path, cache):
    outer, keys = _nested_caches(data_path, cache)
    # selecting both an outer cache and a cache nested inside its parent must
    # not raise: pinning the outer (which runs before its parent subtree is
    # traversed) absorbs the inner
    pinned = pin_caches(outer, keys=keys)
    assert not walk_nodes(CachedNode, pinned)
    assert len(pinned_tag_nodes(pinned)) == 1


def test_pin_count_with_absorption(data_path, cache):
    outer, keys = _nested_caches(data_path, cache)
    # pin only the inner cache, leaving the outer live with a nested pin tag
    partial = pin_caches(outer, keys=[keys[1]])
    assert len(pinned_tag_nodes(partial)) == 1
    assert len(walk_nodes(CachedNode, partial)) == 1
    # pinning the outer absorbs the inner pin tag and adds one of its own: a
    # before/after pin-tag tally reads 0 new, but one cache was truly pinned
    full, n_new = _pin_caches(partial)
    assert n_new == 1
    assert not walk_nodes(CachedNode, full)


def test_pin_reports_new_count(cached_expr):
    pinned, n_new = _pin_caches(cached_expr)
    assert n_new == 1
    _, n_again = _pin_caches(pinned)
    assert n_again == 0


def test_pin_already_pinned_key_message(cached_expr):
    (node,) = walk_nodes(CachedNode, cached_expr)
    key = node.cache.calc_key(node.to_expr())
    pinned = pin_caches(cached_expr, keys=[key])
    # the cache is now a pin tag (no live CachedNode); re-requesting its key
    # reports it as already pinned rather than "no cached node matches"
    with pytest.raises(ValueError, match="already pinned"):
        pin_caches(pinned, keys=[key])


def test_pin_shared_cache_counts_once(data_path, cache):
    con = xo.connect()
    t = deferred_read_parquet(data_path, con, "data")
    cached = t.mutate(z=t.a).cache(cache=cache)
    expr = cached.union(cached)
    expected = expr.execute()
    pinned = pin_caches(expr)
    # one shared cache -> one pin, one verification
    assert len(pin_infos(pinned)) == 1
    (verification,) = verify_pinned(pinned)
    assert verification.ok
    pd.testing.assert_frame_equal(expected, pinned.execute())


def test_verify_pinned_detects_corruption(cached_expr, tmp_path):
    pinned = pin_caches(cached_expr)
    (before,) = verify_pinned(pinned)
    assert before.ok
    (info,) = pin_infos(pinned)
    (cache_file,) = (tmp_path / "cache").rglob(f"{info.cache_key}.parquet")
    corrupted = pd.DataFrame({"a": [9], "b": [9.0], "c": [9.0]})
    corrupted.to_parquet(cache_file)
    (after,) = verify_pinned(pinned)
    assert not after.ok
    assert after.expected_token == info.content_token


def test_pin_build_roundtrip(cached_expr, tmp_path):
    expected = cached_expr.execute()
    pinned = pin_caches(cached_expr)
    build_path = build_expr(pinned, builds_dir=tmp_path / "builds")
    # the pinned parquet is packed inside the artifact, byte-for-byte
    # identical to the cache file it came from (copied, not re-encoded)
    (packed_file,) = (build_path / "reads").glob("*.parquet")
    (info,) = pin_infos(pinned)
    (cache_file,) = (tmp_path / "cache").rglob(f"{info.cache_key}.parquet")
    assert packed_file.read_bytes() == cache_file.read_bytes()

    loaded = load_expr(build_path)
    assert len(loaded.ls.pinned_tags) == 1
    pd.testing.assert_frame_equal(expected, loaded.execute())
    # a loaded build verifies against its packed reads/ dir
    (verification,) = verify_pinned(loaded, reads_dir=build_path / "reads")
    assert verification.ok
    # the packed file is content-addressed by the pinned digest
    assert packed_file.name == f"{info.content_token}.parquet"


def test_pin_build_survives_source_removal(cached_expr, tmp_path):
    expected = cached_expr.execute()
    pinned = pin_caches(cached_expr)
    build_path = build_expr(pinned, builds_dir=tmp_path / "builds")
    # the build must not depend on the cache dir the pin read through
    (info,) = pin_infos(pinned)
    for cached in tmp_path.rglob(f"{info.cache_key}.parquet"):
        if build_path not in cached.parents:
            cached.unlink()
    moved = tmp_path / "moved"
    shutil.move(build_path, moved)
    loaded = load_expr(moved)
    pd.testing.assert_frame_equal(expected, loaded.execute())


def test_pin_build_deterministic(cached_expr, tmp_path):
    first = build_expr(pin_caches(cached_expr), builds_dir=tmp_path / "builds")
    second = build_expr(pin_caches(cached_expr), builds_dir=tmp_path / "builds")
    assert first == second


def test_ls_accessor(cached_expr):
    assert not cached_expr.ls.pinned_tags
    assert not cached_expr.ls.pinned_caches
    pinned = cached_expr.ls.pin_caches()
    assert len(pinned.ls.pinned_tags) == 1
    (info,) = pinned.ls.pinned_caches
    assert isinstance(info, PinInfo)
    (verification,) = pinned.ls.verify_pinned()
    assert verification.ok
