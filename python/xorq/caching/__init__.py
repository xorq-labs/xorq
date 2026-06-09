from __future__ import annotations

import contextlib

from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)
from opentelemetry import trace
from public import public

from xorq.caching.storage import (
    CacheStorage,
    ParquetDummyStorage,
    ParquetStorage,
    ParquetTTLStorage,
    SourceStorage,
)
from xorq.caching.strategy import (
    CacheStrategy,
    ModificationTimeStrategy,
    SnapshotStrategy,
)
from xorq.common.exceptions import XorqError
from xorq.common.utils.otel_utils import tracer
from xorq.vendor.ibis.expr import types as ir


__all__ = [  # noqa: PLE0604
    "ParquetCache",
    "ParquetSnapshotCache",
    "ParquetTTLSnapshotCache",
    "SourceCache",
    "SourceSnapshotCache",
    "GCSCache",
]


@frozen
class Cache:
    strategy = field(validator=instance_of(CacheStrategy))
    storage = field(validator=instance_of(CacheStorage))

    strategy_typ = None
    storage_typ = None

    def __attrs_post_init__(self):
        if not isinstance(self.strategy, self.strategy_typ):
            raise TypeError(
                f"expected strategy of type {self.strategy_typ.__name__}, "
                f"got {type(self.strategy).__name__}"
            )
        if not isinstance(self.storage, self.storage_typ):
            raise TypeError(
                f"expected storage of type {self.storage_typ.__name__}, "
                f"got {type(self.storage).__name__}"
            )

    def calc_key(self, expr):
        # the order here matters: must check is_cached before calling maybe_prevent_cross_source_caching
        if expr.ls.is_cached and expr.ls.cache is self:
            expr = expr.ls.uncached_one
        expr = maybe_prevent_cross_source_caching(expr, self)
        return self.strategy.calc_key(expr)

    def key_exists(self, key):
        return self.storage.exists(key)

    def exists(self, expr):
        key = self.calc_key(expr)
        return self.storage.exists(key)

    def get(self, expr: ir.Expr):
        key = self.calc_key(expr)
        if not self.key_exists(key):
            raise KeyError(key)
        else:
            return self.storage.get(key)

    def put(self, expr: ir.Expr, value, parquet_metadata=None):
        key = self.calc_key(expr)
        if self.key_exists(key):
            raise ValueError(f"cache entry already exists for key {key!r}")
        else:
            return self.storage.put(key, value, parquet_metadata=parquet_metadata)

    @tracer.start_as_current_span("cache.set_default")
    def set_default(self, expr: ir.Expr, default, parquet_metadata=None):
        span = trace.get_current_span()
        key = self.calc_key(expr)
        if not self.key_exists(key):
            span.add_event("cache.miss", {"key": key})
            with tracer.start_as_current_span("cache.put") as child_span:
                child_span.add_event("cache.miss", {"key": key})
                return self.storage.put(key, default, parquet_metadata=parquet_metadata)
        else:
            span.add_event("cache.hit", {"key": key})
            return self.storage.get(key)

    def drop(self, expr: ir.Expr):
        key = self.calc_key(expr)
        if not self.key_exists(key):
            raise KeyError(key)
        else:
            self.storage.drop(key)

    def __dasher_tokenize__(self):
        return (
            "normalize_cache",
            self.strategy,
            self.storage,
            self.strategy.key_prefix,
        )

    @classmethod
    def from_kwargs(cls, **kwargs):
        strategy = cls.strategy_typ()
        storage = cls.storage_typ(**kwargs)
        return cls(strategy=strategy, storage=storage)


@public
@frozen
class ParquetCache(Cache):
    """Cache expression results as Parquet files, re-hashing when source data changes.

    Pairs ``ModificationTimeStrategy`` with ``ParquetStorage``: results are
    written as Parquet files on local disk, and the cache key folds in
    source-data metadata so the cache invalidates automatically when the
    upstream data changes. Build it with :meth:`from_kwargs`.

    Parameters
    ----------
    source : ibis.backends.BaseBackend, optional
        Backend used to write the file on a miss and read it back on a hit.
        Defaults to xorq's default backend.
    relative_path : Path, optional
        Subdirectory under the cache root. Defaults to
        ``xorq.config.options.cache.default_relative_path`` (``parquet``).
    base_path : Path, optional
        Cache root. Defaults to ``None``, which resolves to ``XORQ_CACHE_DIR``.
    """

    strategy_typ = ModificationTimeStrategy
    storage_typ = ParquetStorage


@public
@frozen
class ParquetSnapshotCache(Cache):
    """Cache expression results as Parquet files with a stable, snapshot key.

    Unlike :class:`ParquetCache` (which uses ``ModificationTimeStrategy`` and
    re-hashes when source data changes), this class pairs ``SnapshotStrategy``
    with ``ParquetStorage``: the cache key is computed from the expression
    structure only, so source-data changes do not invalidate cached results.
    Build it with :meth:`from_kwargs`.

    Parameters
    ----------
    source : ibis.backends.BaseBackend, optional
        Backend used to write the file on a miss and read it back on a hit.
        Defaults to xorq's default backend.
    relative_path : Path, optional
        Subdirectory under the cache root. Defaults to
        ``xorq.config.options.cache.default_relative_path`` (``parquet``).
    base_path : Path, optional
        Cache root. Defaults to ``None``, which resolves to ``XORQ_CACHE_DIR``.
    """

    strategy_typ = SnapshotStrategy
    storage_typ = ParquetStorage


@public
@frozen
class ParquetTTLSnapshotCache(Cache):
    """Cache expression results as Parquet files with a stable key and a time-to-live.

    Like :class:`ParquetSnapshotCache` (``SnapshotStrategy`` + Parquet storage)
    but backed by ``ParquetTTLStorage``: a cached file older than ``ttl`` is
    treated as expired and recomputed. Build it with :meth:`from_kwargs`.

    Parameters
    ----------
    source : ibis.backends.BaseBackend, optional
        Backend used to write the file on a miss and read it back on a hit.
        Defaults to xorq's default backend.
    relative_path : Path, optional
        Subdirectory under the cache root. Defaults to
        ``xorq.config.options.cache.default_relative_path`` (``parquet``).
    base_path : Path, optional
        Cache root. Defaults to ``None``, which resolves to ``XORQ_CACHE_DIR``.
    ttl : datetime.timedelta, optional
        How long a cached file stays valid. Defaults to one day.
    """

    strategy_typ = SnapshotStrategy
    storage_typ = ParquetTTLStorage


@public
@frozen
class ParquetDummySnapshotCache(ParquetSnapshotCache):
    storage_typ = ParquetDummyStorage


@public
@frozen
class SourceCache(Cache):
    """Cache expression results as a table in a source backend, with automatic invalidation.

    Pairs ``ModificationTimeStrategy`` with ``SourceStorage``: the result is
    stored as a table in the ``source`` backend, and the cache key folds in
    source-data metadata so the cache invalidates automatically when the
    upstream data changes. Build it with :meth:`from_kwargs`.

    Parameters
    ----------
    source : ibis.backends.BaseBackend, optional
        Backend the cache table lives in. Defaults to xorq's default backend.
    """

    strategy_typ = ModificationTimeStrategy
    storage_typ = SourceStorage


@public
@frozen
class SourceSnapshotCache(Cache):
    """Cache expression results as a table in a source backend, with a stable key.

    Pairs ``SnapshotStrategy`` with ``SourceStorage``: the result is stored as
    a table in the ``source`` backend, and the cache key is computed from the
    expression structure only, so source-data changes do not invalidate cached
    results. Build it with :meth:`from_kwargs`.

    Parameters
    ----------
    source : ibis.backends.BaseBackend, optional
        Backend the cache table lives in. Defaults to xorq's default backend.
    """

    strategy_typ = SnapshotStrategy
    storage_typ = SourceStorage


@public
@frozen
class GCSCache(Cache):
    """Cache expression results as Parquet files in a Google Cloud Storage bucket.

    Pairs ``ModificationTimeStrategy`` with ``GCStorage``: results are written
    as Parquet to a GCS bucket via ``gcsfs`` instead of local disk, and the
    cache key folds in source-data metadata so the cache invalidates when the
    upstream data changes. Build it with :meth:`from_kwargs`.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket to store cached Parquet files in.
    source : ibis.backends.BaseBackend
        Backend used to write the file on a miss and read it back on a hit.
    """

    strategy_typ = ModificationTimeStrategy
    storage_typ = None

    def __attrs_post_init__(self):
        from xorq.common.utils.gcloud_utils import GCStorage  # noqa: PLC0415

        if not isinstance(self.strategy, self.strategy_typ):
            raise TypeError(
                f"expected strategy of type {self.strategy_typ.__name__}, "
                f"got {type(self.strategy).__name__}"
            )
        if not isinstance(self.storage, GCStorage):
            raise TypeError(
                f"expected storage of type GCStorage, got {type(self.storage).__name__}"
            )

    @classmethod
    def from_kwargs(cls, bucket_name, source):
        from xorq.common.utils.gcloud_utils import GCStorage  # noqa: PLC0415

        strategy = cls.strategy_typ()
        storage = GCStorage(bucket_name=bucket_name, source=source)
        return cls(strategy=strategy, storage=storage)


def maybe_prevent_cross_source_caching(expr, storage):
    with contextlib.suppress(XorqError):
        if storage.storage.source != expr._find_backend(use_default=False):
            return expr.into_backend(storage.storage.source)
    return expr
