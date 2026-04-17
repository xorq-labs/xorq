import xorq.vendor.ibis as ibis
from xorq.caching import ParquetSnapshotCache
from xorq.caching.storage import resolve_parquet_cache_path
from xorq.cli import run_cached_command
from xorq.common.utils.caching_utils import CacheKey
from xorq.ibis_yaml.compiler import build_expr, load_expr
from xorq.vendor.ibis.expr.types.core import ExprMetadata


def test_synthetic_key_always_present_for_uncached_expr():
    t = ibis.memtable({"x": [1, 2, 3]})
    key = ExprMetadata.from_expr(t).resolved_snapshot_cache_key
    assert isinstance(key, CacheKey)


def test_synthetic_key_matches_parquet_snapshot_cache_key():
    t = ibis.memtable({"x": [1, 2, 3]})
    synthetic_key = ExprMetadata.from_expr(t).resolved_snapshot_cache_key
    real_key = ParquetSnapshotCache.from_kwargs().calc_key(t)
    assert synthetic_key.key == real_key


def test_to_dict_always_includes_cache_keys():
    d = ExprMetadata.from_expr(ibis.memtable({"x": [1.0]})).to_dict()
    assert "cache_keys" in d
    ck = d["cache_keys"]
    assert set(ck) == {"key", "relative_path"}


def test_parquet_file_locatable_from_metadata_cache_key():
    t = ibis.memtable({"x": [1, 2, 3]})

    ck = ExprMetadata.from_expr(t).resolved_snapshot_cache_key

    cached_expr = t.cache(cache=ParquetSnapshotCache.from_kwargs())
    cached_expr.execute()

    path = resolve_parquet_cache_path(ck.relative_path, ck.key)
    assert path.exists()


def test_run_cached_creates_file_at_metadata_cache_key(tmp_path):
    t = ibis.memtable({"x": [1, 2, 3]})
    expr_path = build_expr(t, builds_dir=tmp_path / "builds")

    run_cached_command(expr_path, cache_type="snapshot")

    loaded_expr = load_expr(expr_path)
    ck = ExprMetadata.from_expr(loaded_expr).resolved_snapshot_cache_key

    path = resolve_parquet_cache_path(ck.relative_path, ck.key)
    assert path.exists()
