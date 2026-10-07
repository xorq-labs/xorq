from __future__ import annotations

import datetime

import pyarrow as pa
import pyarrow.parquet as pq

import xorq.api as xo
from xorq.caching.storage import ParquetStorage, ParquetTTLStorage
from xorq.caching.strategy import ModificationTimeStrategy, SnapshotStrategy
from xorq.common.enums import ProvenanceField
from xorq.common.utils.provenance_utils import (
    build_provenance_metadata,
    get_expr_hash,
    inject_metadata_into_schema,
    read_parquet_provenance,
)


def test_build_provenance_metadata():
    F = ProvenanceField
    strategy = SnapshotStrategy()
    storage = ParquetStorage()
    t = xo.memtable({"x": [1, 2, 3]})
    meta = build_provenance_metadata(t, strategy, storage)

    assert F.expr_hash.encode() in meta
    assert meta[F.expr_hash.encode()] == get_expr_hash(t).encode()
    assert meta[F.cache_strategy.encode()] == b"SnapshotStrategy"
    assert meta[F.cache_storage.encode()] == b"ParquetStorage"
    assert F.cache_ttl_seconds.encode() not in meta


def test_build_provenance_metadata_with_ttl():
    F = ProvenanceField
    strategy = SnapshotStrategy()
    storage = ParquetTTLStorage(ttl=datetime.timedelta(hours=2))
    t = xo.memtable({"y": [10, 20]})
    meta = build_provenance_metadata(t, strategy, storage)

    assert meta[F.expr_hash.encode()] == get_expr_hash(t).encode()
    assert meta[F.cache_strategy.encode()] == b"SnapshotStrategy"
    assert meta[F.cache_storage.encode()] == b"ParquetTTLStorage"
    assert meta[F.cache_ttl_seconds.encode()] == b"7200"


def test_build_provenance_metadata_modification_time():
    F = ProvenanceField
    strategy = ModificationTimeStrategy()
    storage = ParquetStorage()
    t = xo.memtable({"z": [1]})
    meta = build_provenance_metadata(t, strategy, storage)

    assert meta[F.cache_strategy.encode()] == b"ModificationTimeStrategy"


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
    if metadata_dict is not None:
        schema = inject_metadata_into_schema(schema, metadata_dict)
    table = pa.table({"x": [1, 2, 3]}, schema=schema)
    pq.write_table(table, path)


def test_read_parquet_provenance(tmp_path):
    F = ProvenanceField
    path = tmp_path / "test.parquet"
    meta = {F.expr_hash.encode(): b"abc", F.cache_strategy.encode(): b"Snapshot"}
    _write_test_parquet(path, meta)
    prov = read_parquet_provenance(path)
    assert prov == {
        F.expr_hash: "abc",
        F.cache_strategy: "Snapshot",
    }


def test_read_parquet_provenance_none_without_metadata(tmp_path):
    path = tmp_path / "test.parquet"
    _write_test_parquet(path)
    assert read_parquet_provenance(path) is None


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

    F = ProvenanceField
    prov = read_parquet_provenance(path)
    assert prov is not None
    assert prov[F.expr_hash] == get_expr_hash(expr)
    assert prov[F.cache_strategy] == "SnapshotStrategy"
    assert prov[F.cache_storage] == "ParquetStorage"


def test_parquet_ttl_storage_embeds_ttl():
    F = ProvenanceField
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
    assert prov[F.cache_ttl_seconds] == "21600"


def test_plain_tag_changes_build_hash():
    # ADR-0015: every op participates in the build hash, a plain Tag included.
    t = xo.memtable({"a": [1, 2, 3]})
    assert get_expr_hash(t.tag("v1")) != get_expr_hash(t), (
        "attaching a plain Tag must change the build hash"
    )
    assert get_expr_hash(t.tag("a")) != get_expr_hash(t.tag("b")), (
        "tag name must contribute to the build hash"
    )
    assert get_expr_hash(t.tag("a", x=1)) != get_expr_hash(t.tag("a", x=2)), (
        "tag kwargs must contribute to the build hash"
    )
    assert get_expr_hash(t.tag("a").tag("b")) != get_expr_hash(t.tag("a")), (
        "every tag in a chain must contribute to the build hash"
    )
    assert get_expr_hash(t.tag("v1")) == get_expr_hash(t.tag("v1"))


def test_plain_tag_stays_cache_hash_neutral():
    # The build-hash fold is gated to get_expr_hash; the cache key does not see it.
    t = xo.memtable({"a": [1, 2, 3]})
    assert t.tag("v1").ls.tokenized == t.ls.tokenized
    # get_key is None on an uncached expr, so key through a real cache.
    cache = xo.ParquetSnapshotCache.from_kwargs()
    key = t.cache(cache=cache).ls.get_key()
    assert key is not None
    assert t.tag("v1").cache(cache=cache).ls.get_key() == key
    assert t.hashing_tag("v1").cache(cache=cache).ls.get_key() != key


def test_plain_tag_and_hashing_tag_differ_in_build_hash():
    # Same metadata, different op: different DAGs must not share a build hash.
    t = xo.memtable({"a": [1, 2, 3]})
    assert get_expr_hash(t.tag("v1")) != get_expr_hash(t.hashing_tag("v1"))
