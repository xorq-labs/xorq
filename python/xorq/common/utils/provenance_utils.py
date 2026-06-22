from __future__ import annotations

from typing import TYPE_CHECKING

from xorq.common.enums import XORQ_METADATA_PREFIX, ProvenanceField


if TYPE_CHECKING:
    from xorq.vendor.ibis.expr.types.core import Expr


def get_expr_hash(expr: Expr) -> str:
    from xorq.caching.strategy import SnapshotStrategy  # noqa: PLC0415
    from xorq.common.utils.dasher._opaque import include_tee_nodes  # noqa: PLC0415
    from xorq.ibis_yaml.compiler import canonicalize_expr  # noqa: PLC0415
    from xorq.ibis_yaml.config import config  # noqa: PLC0415

    expr = canonicalize_expr(expr)
    with include_tee_nodes(), SnapshotStrategy().normalization_context(expr) as hasher:
        return hasher.tokenize(expr)[: config.hash_length]


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
    import pyarrow.parquet as pq  # noqa: PLC0415

    if fs is not None:
        with fs.open(path, "rb") as fh:
            schema = pq.ParquetFile(fh).schema_arrow
    else:
        schema = pq.read_schema(path)
    raw = schema.metadata or {}
    prefix = XORQ_METADATA_PREFIX.encode()
    filtered = {k.decode(): v.decode() for k, v in raw.items() if k.startswith(prefix)}
    return filtered or None
