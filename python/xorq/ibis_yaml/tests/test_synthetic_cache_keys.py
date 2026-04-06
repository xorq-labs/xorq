import xorq.vendor.ibis as ibis
from xorq.caching import ParquetSnapshotCache
from xorq.caching.storage import resolve_parquet_cache_path
from xorq.common.utils.caching_utils import CacheKey
from xorq.vendor.ibis.expr.types.core import ExprMetadata


def test_synthetic_key_always_present_for_uncached_expr():
    t = ibis.memtable({"x": [1, 2, 3]})
    (key,) = ExprMetadata.from_expr(t).parquet_snapshot_cache_keys
    assert isinstance(key, CacheKey)


def test_synthetic_key_matches_parquet_snapshot_cache_key():
    t = ibis.memtable({"x": [1, 2, 3]})
    synthetic_key, *_ = ExprMetadata.from_expr(t).parquet_snapshot_cache_keys
    real_key = ParquetSnapshotCache.from_kwargs().calc_key(t)
    assert synthetic_key.key == real_key


def test_to_dict_always_includes_cache_keys():
    d = ExprMetadata.from_expr(ibis.memtable({"x": [1.0]})).to_dict()
    assert "cache_keys" in d
    (ck,) = d["cache_keys"]
    assert set(ck) == {"key", "relative_path"}


def test_parquet_file_locatable_from_metadata_cache_key():
    t = ibis.memtable({"x": [1, 2, 3]})

    ck = ExprMetadata.from_expr(t).parquet_snapshot_cache_keys[0]

    cached_expr = t.cache(cache=ParquetSnapshotCache.from_kwargs())
    cached_expr.execute()

    path = resolve_parquet_cache_path(ck.relative_path, ck.key)
    assert path.exists()
