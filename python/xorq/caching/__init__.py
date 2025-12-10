from __future__ import annotations

import functools

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
    ParquetStorage,
    ParquetTTLStorage,
    SourceStorage,
)
from xorq.caching.strategy import (
    CacheStrategy,
    ModificationTimeStrategy,
    SnapshotStrategy,
)
from xorq.common.utils.otel_utils import tracer
from xorq.config import options
from xorq.vendor.ibis.expr import types as ir


@frozen
class Cache:
    strategy = field(validator=instance_of(CacheStrategy))
    storage = field(validator=instance_of(CacheStorage))
    key_prefix = field(
        validator=instance_of(str),
        factory=functools.partial(options.get, "cache.key_prefix"),
    )

    strategy_typ = None
    storage_typ = None

    def __attrs_post_init__(self):
        assert isinstance(self.strategy, self.strategy_typ)
        assert isinstance(self.storage, self.storage_typ)

    def calc_key(self, expr):
        # the order here matters: must check is_cached before calling maybe_prevent_cross_source_caching
        if expr.ls.is_cached and expr.ls.storage is self:
            expr = expr.ls.uncached_one
        expr = maybe_prevent_cross_source_caching(expr, self)
        # FIXME: let strategy solely determine key by giving it key_prefix
        return self.key_prefix + self.strategy.calc_key(expr)

    def key_exists(self, key):
        return self.storage.exists(key)

    def exists(self, expr):
        key = self.calc_key(expr)
        return self.storage.exists(key)

    def get(self, expr: ir.Expr):
        key = self.calc_key(expr)
        if not self.key_exists(key):
            raise KeyError
        else:
            return self.storage.get(key)

    def put(self, expr: ir.Expr, value):
        key = self.calc_key(expr)
        if self.key_exists(key):
            raise ValueError
        else:
            return self.storage.put(key, value)

    @tracer.start_as_current_span("cache.set_default")
    def set_default(self, expr: ir.Expr, default):
        span = trace.get_current_span()
        key = self.calc_key(expr)
        if not self.key_exists(key):
            span.add_event("cache.miss", {"key": key})
            with tracer.start_as_current_span("cache.put") as child_span:
                child_span.add_event("cache.miss", {"key": key})
                return self.storage.put(key, default)
        else:
            span.add_event("cache.hit", {"key": key})
            return self.storage.get(key)

    def drop(self, expr: ir.Expr):
        key = self.calc_key(expr)
        if not self.key_exists(key):
            raise KeyError
        else:
            self.storage.drop(key)

    @classmethod
    def from_kwargs(cls, **kwargs):
        strategy = cls.strategy_typ()
        storage = cls.storage_typ(**kwargs)
        return cls(strategy=strategy, storage=storage)


@public
@frozen
class ParquetCache(Cache):
    """Cache expressions as Parquet files using a snapshot invalidation strategy.

    This storage class saves intermediate results as Parquet files in a specified
    directory and uses a snapshot-based approach for cache invalidation.
    The snapshot strategy ensures cached data is only invalidated when the
    expression's definition changes, making it suitable for stable datasets.

    Parameters
    ----------
    source : ibis.backends.BaseBackend
        The backend to use for execution. Defaults to xorq's default backend.
    path : Path
        The directory where Parquet files will be stored. Defaults to
        xorq.config.options.cache.default_path.
    """

    strategy_typ = ModificationTimeStrategy
    storage_typ = ParquetStorage


@public
@frozen
class ParquetSnapshotCache(Cache):
    """Cache expressions as Parquet files using a snapshot invalidation strategy.

    This storage class saves intermediate results as Parquet files in a specified
    directory and uses a snapshot-based approach for cache invalidation.
    The snapshot strategy ensures cached data is only invalidated when the
    expression's definition changes, making it suitable for stable datasets.

    Parameters
    ----------
    source : ibis.backends.BaseBackend
        The backend to use for execution. Defaults to xorq's default backend.
    path : Path
        The directory where Parquet files will be stored. Defaults to
        xorq.config.options.cache.default_path.
    """

    strategy_typ = SnapshotStrategy
    storage_typ = ParquetStorage


@public
@frozen
class ParquetTTLSnapshotCache(Cache):
    strategy_typ = SnapshotStrategy
    storage_typ = ParquetTTLStorage


@public
@frozen
class SourceCache(Cache):
    strategy_typ = ModificationTimeStrategy
    storage_typ = SourceStorage


@public
@frozen
class SourceSnapshotCache(Cache):
    strategy_typ = SnapshotStrategy
    storage_typ = SourceStorage


@public
@frozen
class GCSCache(Cache):
    strategy_typ = ModificationTimeStrategy
    storage_typ = None

    def __attrs_post_init__(self):
        from xorq.common.utils.gcloud_utils import GCStorage

        assert isinstance(self.strategy, ModificationTimeStrategy)
        assert isinstance(self.storage, GCStorage)

    @classmethod
    def from_kwargs(cls, bucket_name, source):
        from xorq.common.utils.gcloud_utils import GCStorage

        strategy = cls.strategy_typ()
        storage = GCStorage(bucket_name=bucket_name, source=source)
        return cls(strategy=strategy, storage=storage)


def maybe_prevent_cross_source_caching(expr, storage):
    from xorq.expr.relations import (
        into_backend,
    )

    if storage.storage.source != expr._find_backend():
        expr = into_backend(expr, storage.storage.source)
    return expr
