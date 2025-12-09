from __future__ import annotations

import contextlib
import datetime
import functools
from abc import (
    abstractmethod,
)
from pathlib import (
    Path,
)

import dask
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)
from opentelemetry import trace
from public import public

import xorq.common.utils.dask_normalize  # noqa: F401
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.caching_utils import (
    get_xorq_cache_dir,
)
from xorq.common.utils.dask_normalize.dask_normalize_expr import (
    normalize_backend,
    normalize_read,
    normalize_remote_table,
)
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    patch_normalize_token,
)
from xorq.common.utils.defer_utils import (
    deferred_read_parquet,
)
from xorq.common.utils.func_utils import (
    if_not_none,
)
from xorq.common.utils.otel_utils import tracer
from xorq.config import _backend_init, options
from xorq.expr.relations import (
    Read,
    RemoteTable,
)
from xorq.vendor import ibis
from xorq.vendor.ibis.expr import types as ir


@frozen
class CacheStrategy:
    @abstractmethod
    def calc_key(self, expr):
        pass


@frozen
class CacheStorage:
    @abstractmethod
    def exists(self, key):
        pass

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def put(self, key, value):
        pass

    @abstractmethod
    def drop(self, key):
        pass


@frozen
class Cache:
    strategy = field(validator=instance_of(CacheStrategy))
    storage = field(validator=instance_of(CacheStorage))
    key_prefix = field(
        validator=instance_of(str),
        factory=functools.partial(options.get, "cache.key_prefix"),
    )

    def calc_key(self, expr):
        # the order here matters: must check is_cached before calling maybe_prevent_cross_source_caching
        if expr.ls.is_cached and expr.ls.storage.cache is self:
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


@frozen
class ModificationTimeStrategy(CacheStrategy):
    key_prefix = field(
        validator=instance_of(str),
        factory=functools.partial(options.get, "cache.key_prefix"),
    )

    def calc_key(self, expr: ir.Expr):
        return expr.ls.tokenized


@frozen
class SnapshotStrategy(CacheStrategy):
    def calc_key(self, expr: ir.Expr):
        # can we cache this?
        with self.normalization_context(expr):
            replaced = self.replace_remote_table(expr)
            tokenized = replaced.ls.tokenized
            return "-".join(("snapshot", tokenized))

    @contextlib.contextmanager
    def normalization_context(self, expr):
        typs = map(type, expr.ls.backends)
        with patch_normalize_token(*typs, f=self.normalize_backend):
            with patch_normalize_token(
                ops.DatabaseTable,
                f=self.normalize_databasetable,
            ):
                with patch_normalize_token(
                    Read,
                    f=self.cached_normalize_read,
                ):
                    yield

    @staticmethod
    @functools.cache
    def cached_normalize_read(op):
        return normalize_read(op)

    @staticmethod
    @functools.cache
    def cached_replace_remote_table(op):
        def rename_remote_table(node, _, **kwargs):
            if isinstance(node, RemoteTable):
                # FIXME: how to verify that we're always within a self.normalization_context?
                name = dask.base.tokenize(node)
                rt = RemoteTable(
                    name=name,
                    schema=node.schema,
                    source=node.source,
                    remote_expr=node.remote_expr,
                    namespace=node.namespace,
                )
                return rt
            else:
                return node.__recreate__(kwargs)

        return op.replace(rename_remote_table)

    def replace_remote_table(self, expr):
        """replace remote table with deterministic name ***strictly for key calculation***"""
        if expr.op().find(RemoteTable):
            expr = self.cached_replace_remote_table(expr.op()).to_expr()
        return expr

    @staticmethod
    def normalize_backend(con):
        name = con.name
        if name in ("pandas", "duckdb", "datafusion", "let"):
            return (name, None)
        else:
            return normalize_backend(con)

    @staticmethod
    def normalize_databasetable(dt):
        if isinstance(dt, RemoteTable):
            # one alternative is to explicitly iterate over the fields name, schema, source, namespace
            # but explicit is better than implicit, additionally the name is not a safe bet for caching
            # RemoteTable
            return normalize_remote_table(dt)
        else:
            keys = ["name", "schema", "source", "namespace"]
            return dask.tokenize._normalize_seq_func(
                (key, getattr(dt, key)) for key in keys
            )


@frozen
class ParquetStorage(CacheStorage):
    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=_backend_init,
    )
    relative_path = field(
        validator=instance_of(Path),
        factory=functools.partial(options.get, "cache.default_relative_path"),
        converter=Path,
    )
    base_path = field(
        validator=optional(instance_of(Path)),
        default=None,
        converter=if_not_none(Path),
    )

    def __attrs_post_init__(self):
        self.path.mkdir(exist_ok=True, parents=True)

    @property
    def path(self):
        return (self.base_path or get_xorq_cache_dir()).joinpath(self.relative_path)

    def get_path(self, key):
        return self.path.joinpath(key + ".parquet")

    def exists(self, key):
        return self.get_path(key).exists()

    def get(self, key):
        op = deferred_read_parquet(
            path=self.get_path(key),
            con=self.source,
            table_name=key,
        ).op()
        return op

    def put(self, key, value):
        path = self.get_path(key)
        # move from temp location upon success to prevent empty files on failure
        tmp_path = path.with_name(path.name + ".tmp")
        value.to_expr().to_parquet(tmp_path)
        tmp_path.rename(path)
        return self.get(key)

    def drop(self, key):
        path = self.get_path(key)
        path.unlink()


@frozen
class ParquetTTLStorage(ParquetStorage):
    ttl = field(
        validator=instance_of(datetime.timedelta), default=datetime.timedelta(days=1)
    )

    def exists(self, key):
        path = self.get_path(key)
        return path.exists() and self.satisfies_ttl(path)

    def satisfies_ttl(self, path):
        delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(
            path.stat().st_mtime
        )
        return delta < self.ttl


# named with underscore prefix until we swap out SourceStorage
@frozen
class SourceStorage(CacheStorage):
    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=_backend_init,
    )

    def exists(self, key):
        return key in self.source.tables

    def get(self, key):
        return self.source.table(key).op()

    def put(self, key, value):
        def is_remote(value):
            name = value.to_expr()._find_backend().name
            # FIXME: add pyiceberg, trino
            return name in ("postgres", "snowflake")

        def is_single_backend(storage, value):
            from xorq.common.utils.graph_utils import find_all_sources

            return (storage.source,) == find_all_sources(value.to_expr())

        if is_remote(value):
            if is_single_backend(self, value):
                from xorq.expr.api import _transform_expr

                # must transform for Read ops: create_table expects a vanilla ibis expr
                (transformed, _) = _transform_expr(value.to_expr())
                self.source.create_table(key, transformed)
            else:
                assert hasattr(self.source, "read_record_batches")
                # read_record_batches will create durable table in out-of-core fashion
                # works for snowflake and postgres
                self.source.read_record_batches(
                    value.to_expr().to_pyarrow_batches(),
                    key,
                )
        else:
            self.source.create_table(key, value.to_expr().to_pyarrow())
        return self.get(key)

    def drop(self, key):
        self.source.drop_table(key)


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

    def __attrs_post_init__(self):
        assert isinstance(self.strategy, SnapshotStrategy)
        assert isinstance(self.storage, ParquetStorage)

    @classmethod
    def from_kwargs(cls, **kwargs):
        strategy = SnapshotStrategy()
        storage = ParquetStorage(**kwargs)
        return cls(strategy=strategy, storage=storage)


@public
@frozen
class ParquetTTLSnapshotCache(Cache):
    def __attrs_post_init__(self):
        assert isinstance(self.strategy, SnapshotStrategy)
        assert isinstance(self.storage, ParquetTTLStorage)

    @classmethod
    def from_kwargs(cls, **kwargs):
        strategy = SnapshotStrategy()
        storage = ParquetTTLStorage(**kwargs)
        return cls(strategy=strategy, storage=storage)


@public
@frozen
class SourceCache:
    def __attrs_post_init__(self):
        assert isinstance(self.strategy, ModificationTimeStrategy)
        assert isinstance(self.storage, SourceStorage)

    @classmethod
    def from_kwargs(cls, key_prefix, source):
        strategy = ModificationTimeStrategy(key_prefix=key_prefix)
        storage = SourceStorage(source=source)
        return cls(strategy=strategy, storage=storage)


@public
@frozen
class SourceSnapshotCache:
    def __attrs_post_init__(self):
        assert isinstance(self.strategy, SnapshotStrategy)
        assert isinstance(self.storage, SourceStorage)

    @classmethod
    def from_kwargs(cls, source):
        strategy = SnapshotStrategy()
        storage = SourceStorage(source=source)
        return cls(strategy=strategy, storage=storage)


@public
@frozen
class GCSCache(Cache):
    def __attrs_post_init__(self):
        from xorq.common.utils.gcloud_utils import _GCStorage

        assert isinstance(self.strategy, ModificationTimeStrategy)
        assert isinstance(self.storage, _GCStorage)

    @classmethod
    def from_kwargs(cls, bucket_name, source):
        from xorq.common.utils.gcloud_utils import _GCStorage

        strategy = ModificationTimeStrategy()
        storage = _GCStorage(bucket_name=bucket_name, source=source)
        return cls(strategy=strategy, storage=storage)


def maybe_prevent_cross_source_caching(expr, storage):
    from xorq.expr.relations import (
        into_backend,
    )

    if storage.storage.source != expr._find_backend():
        expr = into_backend(expr, storage.storage.source)
    return expr
