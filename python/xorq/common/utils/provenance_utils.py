from __future__ import annotations

import pyarrow.parquet as pq


XORQ_METADATA_PREFIX = "xorq:"


def get_expr_hash(expr):
    import dask.base  # noqa: PLC0415

    from xorq.caching.strategy import SnapshotStrategy  # noqa: PLC0415
    from xorq.ibis_yaml.compiler import canonicalize_expr  # noqa: PLC0415
    from xorq.ibis_yaml.config import config  # noqa: PLC0415

    expr = canonicalize_expr(expr)
    with SnapshotStrategy().normalization_context(expr):
        return dask.base.tokenize(expr)[: config.hash_length]


def build_provenance_metadata(expr, strategy, storage):
    expr_hash = get_expr_hash(expr)
    metadata = {
        b"xorq:expr_hash": expr_hash.encode(),
        b"xorq:cache_strategy": type(strategy).__name__.encode(),
        b"xorq:cache_storage": type(storage).__name__.encode(),
    }
    if hasattr(storage, "ttl"):
        metadata[b"xorq:cache_ttl_seconds"] = str(
            int(storage.ttl.total_seconds())
        ).encode()
    return metadata


def inject_metadata_into_schema(schema, metadata_dict):
    existing = schema.metadata or {}
    merged = {**existing, **metadata_dict}
    return schema.with_metadata(merged)


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
