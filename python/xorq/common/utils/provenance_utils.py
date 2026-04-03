from __future__ import annotations

import enum

import pyarrow.parquet as pq


XORQ_METADATA_PREFIX = "xorq:"


class ProvenanceField(enum.StrEnum):
    expr_hash = f"{XORQ_METADATA_PREFIX}expr_hash"
    cache_strategy = f"{XORQ_METADATA_PREFIX}cache_strategy"
    cache_storage = f"{XORQ_METADATA_PREFIX}cache_storage"
    cache_ttl_seconds = f"{XORQ_METADATA_PREFIX}cache_ttl_seconds"


def get_expr_hash(expr):
    import dask.base  # noqa: PLC0415

    from xorq.caching.strategy import SnapshotStrategy  # noqa: PLC0415
    from xorq.ibis_yaml.compiler import canonicalize_expr  # noqa: PLC0415
    from xorq.ibis_yaml.config import config  # noqa: PLC0415

    expr = canonicalize_expr(expr)
    with SnapshotStrategy().normalization_context(expr):
        return dask.base.tokenize(expr)[: config.hash_length]


def build_provenance_metadata(expr, strategy, storage):
    F = ProvenanceField
    expr_hash = get_expr_hash(expr)
    metadata = {
        F.expr_hash.encode(): expr_hash.encode(),
        F.cache_strategy.encode(): type(strategy).__name__.encode(),
        F.cache_storage.encode(): type(storage).__name__.encode(),
    }
    if hasattr(storage, "ttl"):
        metadata[F.cache_ttl_seconds.encode()] = str(
            int(storage.ttl.total_seconds())
        ).encode()
    return metadata


def inject_metadata_into_schema(schema, metadata_dict):
    return schema.with_metadata((schema.metadata or {}) | metadata_dict)


def read_parquet_provenance(path, fs=None):
    if fs is not None:
        with fs.open(path, "rb") as fh:
            schema = pq.ParquetFile(fh).schema_arrow
    else:
        schema = pq.read_schema(path)
    raw = schema.metadata or {}
    prefix = XORQ_METADATA_PREFIX.encode()
    filtered = {k.decode(): v.decode() for k, v in raw.items() if k.startswith(prefix)}
    return filtered or None
