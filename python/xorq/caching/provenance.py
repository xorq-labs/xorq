from __future__ import annotations

import datetime
from pathlib import Path

import pyarrow.parquet as pq


XORQ_METADATA_PREFIX = "xorq:"


def build_provenance_metadata(expr_hash, strategy, storage):
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


def check_cache_valid(path, fs=None):
    provenance = read_parquet_provenance(path, fs=fs)
    if provenance is None:
        return True
    ttl_str = provenance.get("xorq:cache_ttl_seconds")
    if not ttl_str:
        return True
    ttl = datetime.timedelta(seconds=int(ttl_str))
    mtime = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return (datetime.datetime.now() - mtime) < ttl


def cache_to_entry_map(directory):
    result = {}
    for p in Path(directory).glob("*.parquet"):
        provenance = read_parquet_provenance(p)
        if provenance and "xorq:expr_hash" in provenance:
            result[p.name] = provenance["xorq:expr_hash"]
    return result
