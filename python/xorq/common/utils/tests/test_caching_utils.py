from pathlib import Path

import xorq.api as xo
from xorq.caching import ParquetSnapshotCache
from xorq.common.utils.caching_utils import CacheKey, get_xorq_cache_dir


def test_default_caching_dir():
    actual_dir = get_xorq_cache_dir()
    assert actual_dir is not None
    assert isinstance(actual_dir, Path)

    assert actual_dir.match("**/.cache/xorq/")


def test_cache_keys_stores_key_and_relative_path(catalog, tmp_path):
    """CacheKey in the sidecar carries both the hash key and the relative_path
    so paths can be reconstructed without loading the expression."""
    relative = "my_cache"
    cache = ParquetSnapshotCache.from_kwargs(relative_path=relative)
    expr = xo.memtable({"x": [1, 2, 3]}).cache(cache=cache)
    entry = catalog.add(expr)

    assert len(entry.cache_keys) == 1
    ck = entry.cache_keys[0]
    assert isinstance(ck, CacheKey)
    assert ck.relative_path == relative
    assert ck.key  # non-empty hash string


def test_cache_keys_paths_relocatable(catalog, tmp_path, monkeypatch):
    """cache_keys_paths uses get_xorq_cache_dir() + relative_path at access time;
    no expression loading is needed, so the result tracks the current cache dir."""
    cache_dir_A = tmp_path / "cache_A"
    cache_dir_B = tmp_path / "cache_B"
    relative = "my_cache"

    monkeypatch.setattr("xorq.caching.storage.get_xorq_cache_dir", lambda: cache_dir_A)
    cache = ParquetSnapshotCache.from_kwargs(relative_path=relative)
    expr = xo.memtable({"x": [1, 2, 3]}).cache(cache=cache)
    entry = catalog.add(expr)

    paths_at_A = entry.cache_keys_paths
    assert str(cache_dir_A) in paths_at_A[0]

    monkeypatch.setattr("xorq.caching.storage.get_xorq_cache_dir", lambda: cache_dir_B)
    paths_at_B = entry.cache_keys_paths
    assert str(cache_dir_B) in paths_at_B[0]
    assert str(cache_dir_A) not in paths_at_B[0]

    # Same filename (hash key) in both dirs
    assert Path(paths_at_A[0]).name == Path(paths_at_B[0]).name


def test_base_path_is_silently_dropped_through_catalog_round_trip(
    catalog, tmp_path, monkeypatch
):
    """base_path is not serialized in ibis_yaml — only relative_path survives.
    load_cache_from_yaml always calls from_kwargs without base_path, so the
    resulting CacheKey.relative_path is correct but base_path is always None.

    Consequence (good): cache_keys_paths always uses get_xorq_cache_dir().
    Consequence (bad): setting base_path on an expression going through catalog
    is silently ignored — no warning is raised anywhere.
    """
    cache_dir_explicit = tmp_path / "explicit_base"
    cache_dir_xorq = tmp_path / "xorq_cache"

    cache = ParquetSnapshotCache.from_kwargs(
        relative_path="my_cache", base_path=cache_dir_explicit
    )
    expr = xo.memtable({"x": [1, 2, 3]}).cache(cache=cache)
    entry = catalog.add(expr)

    monkeypatch.setattr(
        "xorq.caching.storage.get_xorq_cache_dir", lambda: cache_dir_xorq
    )
    paths = entry.cache_keys_paths

    assert paths
    assert str(cache_dir_xorq) in paths[0]
    assert str(cache_dir_explicit) not in paths[0]
