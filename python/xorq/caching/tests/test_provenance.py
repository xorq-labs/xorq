from __future__ import annotations

import datetime
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

import xorq.api as xo
from xorq.caching.provenance import (
    build_provenance_metadata,
    cache_to_entry_map,
    check_cache_valid,
    inject_metadata_into_schema,
    read_parquet_provenance,
)
from xorq.caching.storage import ParquetStorage, ParquetTTLStorage
from xorq.caching.strategy import ModificationTimeStrategy, SnapshotStrategy


def test_build_provenance_metadata():
    strategy = SnapshotStrategy()
    storage = ParquetStorage()
    meta = build_provenance_metadata("abc123", strategy, storage)

    assert meta[b"xorq:expr_hash"] == b"abc123"
    assert meta[b"xorq:cache_strategy"] == b"SnapshotStrategy"
    assert meta[b"xorq:cache_storage"] == b"ParquetStorage"
    assert b"xorq:cache_ttl_seconds" not in meta


def test_build_provenance_metadata_with_ttl():
    strategy = SnapshotStrategy()
    storage = ParquetTTLStorage(ttl=datetime.timedelta(hours=2))
    meta = build_provenance_metadata("key456", strategy, storage)

    assert meta[b"xorq:expr_hash"] == b"key456"
    assert meta[b"xorq:cache_strategy"] == b"SnapshotStrategy"
    assert meta[b"xorq:cache_storage"] == b"ParquetTTLStorage"
    assert meta[b"xorq:cache_ttl_seconds"] == b"7200"


def test_build_provenance_metadata_modification_time():
    strategy = ModificationTimeStrategy()
    storage = ParquetStorage()
    meta = build_provenance_metadata("hash789", strategy, storage)

    assert meta[b"xorq:cache_strategy"] == b"ModificationTimeStrategy"


def test_inject_metadata_into_schema():
    schema = pa.schema([("x", pa.int64())])
    meta = {b"xorq:expr_hash": b"test"}
    result = inject_metadata_into_schema(schema, meta)

    assert result.metadata[b"xorq:expr_hash"] == b"test"


def test_inject_metadata_preserves_existing():
    schema = pa.schema([("x", pa.int64())]).with_metadata({b"pandas": b"existing"})
    meta = {b"xorq:expr_hash": b"test"}
    result = inject_metadata_into_schema(schema, meta)

    assert result.metadata[b"pandas"] == b"existing"
    assert result.metadata[b"xorq:expr_hash"] == b"test"


def _write_test_parquet(path, metadata_dict=None):
    schema = pa.schema([("x", pa.int64())])
    if metadata_dict:
        schema = inject_metadata_into_schema(schema, metadata_dict)
    table = pa.table({"x": [1, 2, 3]}, schema=schema)
    pq.write_table(table, path)


def test_read_parquet_provenance():
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        meta = {b"xorq:expr_hash": b"abc", b"xorq:cache_strategy": b"Snapshot"}
        _write_test_parquet(path, meta)
        prov = read_parquet_provenance(path)
        assert prov == {
            "xorq:expr_hash": "abc",
            "xorq:cache_strategy": "Snapshot",
        }
    finally:
        os.unlink(path)


def test_read_parquet_provenance_none_without_metadata():
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        _write_test_parquet(path)
        assert read_parquet_provenance(path) is None
    finally:
        os.unlink(path)


def test_check_cache_valid_no_ttl():
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        meta = {b"xorq:expr_hash": b"abc"}
        _write_test_parquet(path, meta)
        assert check_cache_valid(path) is True
    finally:
        os.unlink(path)


def test_check_cache_valid_no_metadata():
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        _write_test_parquet(path)
        assert check_cache_valid(path) is True
    finally:
        os.unlink(path)


def test_check_cache_valid_ttl_not_expired():
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        meta = {b"xorq:expr_hash": b"abc", b"xorq:cache_ttl_seconds": b"3600"}
        _write_test_parquet(path, meta)
        # file was just written, so mtime is now — well within 3600s TTL
        assert check_cache_valid(path) is True
    finally:
        os.unlink(path)


def test_check_cache_valid_ttl_expired():
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        meta = {b"xorq:expr_hash": b"abc", b"xorq:cache_ttl_seconds": b"1"}
        _write_test_parquet(path, meta)
        # backdate the file mtime by 10 seconds
        old_time = (
            datetime.datetime.now() - datetime.timedelta(seconds=10)
        ).timestamp()
        os.utime(path, (old_time, old_time))
        assert check_cache_valid(path) is False
    finally:
        os.unlink(path)


def test_cache_to_entry_map():
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, expr_hash in [("a.parquet", "hash_a"), ("b.parquet", "hash_b")]:
            path = os.path.join(tmpdir, name)
            meta = {b"xorq:expr_hash": expr_hash.encode()}
            _write_test_parquet(path, meta)

        # file without provenance
        _write_test_parquet(os.path.join(tmpdir, "c.parquet"))

        result = cache_to_entry_map(tmpdir)
        assert result == {"a.parquet": "hash_a", "b.parquet": "hash_b"}


def test_parquet_storage_embeds_metadata():
    cache = xo.ParquetSnapshotCache.from_kwargs()
    t = xo.memtable({"x": [1, 2, 3]})
    expr = t.cache(cache=cache)
    expr.execute()

    cached_nodes = expr.ls.cached_nodes
    assert cached_nodes, "expression must have cached nodes"

    cn = cached_nodes[0]
    key = cn.cache.calc_key(cn.parent)
    path = cn.cache.storage.get_path(key)
    assert path.exists()

    prov = read_parquet_provenance(path)
    assert prov is not None
    assert prov["xorq:expr_hash"] == key
    assert prov["xorq:cache_strategy"] == "SnapshotStrategy"
    assert prov["xorq:cache_storage"] == "ParquetStorage"


def test_parquet_ttl_storage_embeds_ttl():
    cache = xo.ParquetTTLSnapshotCache(
        strategy=SnapshotStrategy(),
        storage=ParquetTTLStorage(ttl=datetime.timedelta(hours=6)),
    )
    t = xo.memtable({"y": [10, 20]})
    expr = t.cache(cache=cache)
    expr.execute()

    cached_nodes = expr.ls.cached_nodes
    assert cached_nodes

    cn = cached_nodes[0]
    key = cn.cache.calc_key(cn.parent)
    path = cn.cache.storage.get_path(key)
    assert path.exists()

    prov = read_parquet_provenance(path)
    assert prov is not None
    assert prov["xorq:cache_ttl_seconds"] == "21600"
