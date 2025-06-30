from __future__ import annotations

import contextlib
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

import xorq as xo
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
from xorq.common.utils.otel_utils import tracer
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
    def key_exists(self, key):
        pass

    @abstractmethod
    def _get(self, expr):
        pass

    @abstractmethod
    def _put(self, expr):
        pass

    @abstractmethod
    def _drop(self, expr):
        pass


@frozen
class Cache:
    strategy = field(validator=instance_of(CacheStrategy))
    storage = field(validator=instance_of(CacheStorage))
    key_prefix = field(
        validator=instance_of(str),
        factory=functools.partial(xorq.options.get, "cache.key_prefix"),
    )

    def exists(self, expr):
        key = self.get_key(expr)
        return self.storage.key_exists(key)

    def key_exists(self, key):
        return self.storage.key_exists(key)

    def get_key(self, expr):
        # the order here matters: must check is_cached before calling maybe_prevent_cross_source_caching
        if expr.ls.is_cached and expr.ls.storage.cache is self:
            expr = expr.ls.uncached_one
        expr = maybe_prevent_cross_source_caching(expr, self)
        # FIXME: let strategy solely determine key by giving it key_prefix
        return self.key_prefix + self.strategy.get_key(expr)

    def get(self, expr: ir.Expr):
        key = self.get_key(expr)
        if not self.key_exists(key):
            raise KeyError
        else:
            return self.storage._get(key)

    def put(self, expr: ir.Expr, value):
        key = self.get_key(expr)
        if self.key_exists(key):
            raise ValueError
        else:
            key = self.get_key(expr)
            return self.storage._put(key, value)

    @tracer.start_as_current_span("cache.set_default")
    def set_default(self, expr: ir.Expr, default):
        span = trace.get_current_span()
        key = self.get_key(expr)
        if not self.key_exists(key):
            span.add_event("cache.miss", {"key": key})
            with tracer.start_as_current_span("cache._put") as child_span:
                child_span.add_event("cache.miss", {"key": key})
                return self.storage._put(key, default)
        else:
            span.add_event("cache.hit", {"key": key})
            return self.storage._get(key)

    def drop(self, expr: ir.Expr):
        key = self.get_key(expr)
        if not self.key_exists(key):
            raise KeyError
        else:
            self.storage._drop(key)


@frozen
class ModificationTimeStrategy(CacheStrategy):
    key_prefix = field(
        validator=instance_of(str),
        factory=functools.partial(xorq.options.get, "cache.key_prefix"),
    )

    def get_key(self, expr: ir.Expr):
        return dask.base.tokenize(expr)


@frozen
class SnapshotStrategy(CacheStrategy):
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

    def get_key(self, expr: ir.Expr):
        # can we cache this?
        with self.normalization_context(expr):
            replaced = self.replace_remote_table(expr)
            tokenized = dask.tokenize.tokenize(replaced)
            return "-".join(("snapshot", tokenized))

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
class _ParquetStorage(CacheStorage):
    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=xorq.config._backend_init,
    )
    relative_path = field(
        validator=instance_of(Path),
        factory=functools.partial(xorq.options.get, "cache.default_relative_path"),
    )
    base_path = field(
        validator=optional(instance_of(Path)),
        default=None,
    )

    @property
    def path(self):
        return (self.base_path or get_xorq_cache_dir()).joinpath(self.relative_path)

    def __attrs_post_init__(self):
        self.path.mkdir(exist_ok=True, parents=True)

    def get_loc(self, key):
        return self.path.joinpath(key + ".parquet")

    def key_exists(self, key):
        return self.get_loc(key).exists()

    def _get(self, key):
        op = self.source.read_parquet(self.get_loc(key), key).op()
        return op

    def _put(self, key, value):
        loc = self.get_loc(key)
        xo.to_parquet(value.to_expr(), loc)
        return self._get(key)

    def _drop(self, key):
        path = self.get_loc(key)
        path.unlink()
        # FIXME: what to do if table is not registered?
        self.source.drop_table(key)


# named with underscore prefix until we swap out SourceStorage
@frozen
class _SourceStorage(CacheStorage):
    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=xorq.config._backend_init,
    )

    def key_exists(self, key):
        return key in self.source.tables

    def _get(self, key):
        return self.source.table(key).op()

    def _put(self, key, value):
        if self.source.name == "postgres":
            self.source.read_record_batches(
                value.to_expr().to_pyarrow_batches(),
                key,
            )
        else:
            self.source.create_table(key, xo.to_pyarrow(value.to_expr()))
        return self._get(key)

    def _drop(self, key):
        self.source.drop_table(key)


###############
###############
# drop in replacements for previous versions


def chained_getattr(self, attr):
    if hasattr(self.cache, attr):
        return getattr(self.cache, attr)
    if hasattr(self.cache.storage, attr):
        return getattr(self.cache.storage, attr)
    if hasattr(self.cache.strategy, attr):
        return getattr(self.cache.strategy, attr)
    else:
        return object.__getattribute__(self, attr)


@public
@frozen
class ParquetSnapshotStorage:
    """Storage that caches expressions as Parquet files using a snapshot invalidation strategy.

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
        xorq.options.cache.default_path.
    """

    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=xorq.config._backend_init,
    )
    relative_path = field(
        validator=instance_of(Path),
        factory=functools.partial(xorq.options.get, "cache.default_relative_path"),
    )
    base_path = field(
        validator=optional(instance_of(Path)),
        default=None,
    )
    cache = field(validator=instance_of(Cache), init=False)

    def __attrs_post_init__(self):
        cache = Cache(
            strategy=SnapshotStrategy(),
            storage=_ParquetStorage(
                source=self.source,
                relative_path=self.relative_path,
                base_path=self.base_path,
            ),
        )
        object.__setattr__(self, "cache", cache)

    def exists(self, expr: ir.Expr) -> bool:
        """Check if the expression has been cached.

        Parameters
        ----------
        expr : ir.Expr
            The expression to check

        Returns
        -------
        bool
            True if the expression is cached, False otherwise
        """
        return self.cache.exists(expr)

    __getattr__ = chained_getattr


@public
@frozen
class ParquetStorage:
    """Storage that caches expressions as Parquet files using a modification time strategy.

    This storage class saves intermediate results as Parquet files in a specified
    directory and uses a modification time-based approach for cache invalidation.
    The cache is invalidated when the modification time of the source data changes,
    making it suitable for data that changes periodically.

    Parameters
    ----------
    source : ibis.backends.BaseBackend
        The backend to use for execution. Defaults to xorq's default backend.
    relative_path : Path
        The relative directory where Parquet files will be stored. Defaults to
        xorq.options.cache.default_path.
    base_path : Path
        The base path where Parquet files will be stored.
    """

    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=xorq.config._backend_init,
    )
    relative_path = field(
        validator=instance_of(Path),
        factory=functools.partial(xorq.options.get, "cache.default_relative_path"),
    )
    base_path = field(
        validator=optional(instance_of(Path)),
        default=None,
    )
    cache = field(validator=instance_of(Cache), init=False)

    def __attrs_post_init__(self):
        cache = Cache(
            strategy=ModificationTimeStrategy(),
            storage=_ParquetStorage(
                source=self.source,
                relative_path=self.relative_path,
                base_path=self.base_path,
            ),
        )
        object.__setattr__(self, "cache", cache)

    __getattr__ = chained_getattr


@public
@frozen
class SourceStorage:
    """Storage that caches expressions within the source backend using a modification time strategy.

    This storage class materializes intermediate results as tables within the source
    backend itself (e.g., as temporary tables in a database) and uses a modification
    time-based approach for cache invalidation. The cache is invalidated when the
    modification time of the source data changes.

    Parameters
    ----------
    source : ibis.backends.BaseBackend
        The backend to use for both execution and storage. Defaults to xorq's default backend.
    """

    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=xorq.config._backend_init,
    )
    cache = field(validator=instance_of(Cache), init=False)

    def __attrs_post_init__(self):
        cache = Cache(
            strategy=ModificationTimeStrategy(), storage=_SourceStorage(self.source)
        )
        object.__setattr__(self, "cache", cache)

    __getattr__ = chained_getattr


@public
@frozen
class SourceSnapshotStorage:
    """Storage that caches expressions within the source backend using a snapshot strategy.

    This storage class materializes intermediate results as tables within the source
    backend itself (e.g., as temporary tables in a database) and uses a snapshot-based
    approach for cache invalidation. The snapshot strategy ensures cached data is only
    invalidated when the expression's definition changes, making it suitable for
    stable datasets that are queried frequently.

    Parameters
    ----------
    source : ibis.backends.BaseBackend
        The backend to use for both execution and storage. Defaults to xorq's default backend.
    """

    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=xorq.config._backend_init,
    )
    cache = field(validator=instance_of(Cache), init=False)

    def __attrs_post_init__(self):
        cache = Cache(strategy=SnapshotStrategy(), storage=_SourceStorage(self.source))
        object.__setattr__(self, "cache", cache)

    __getattr__ = chained_getattr


@public
@frozen
class GCStorage:
    bucket_name = field(validator=instance_of(str))
    source = field(
        validator=instance_of(xo.vendor.ibis.backends.BaseBackend),
        factory=xo.config._backend_init,
    )
    cache = field(validator=instance_of(Cache), init=False)

    def __attrs_post_init__(self):
        from xorq.common.utils.gcloud_utils import _GCStorage

        cache = Cache(
            strategy=ModificationTimeStrategy(),
            storage=_GCStorage(
                bucket_name=self.bucket_name,
                source=self.source,
            ),
        )
        object.__setattr__(self, "cache", cache)

    def get_path(self, expr):
        return (
            Path(
                self.bucket_name,
                self.cache.get_key(expr),
            )
            .with_suffix(".parquet")
            .as_posix()
        )

    __getattr__ = chained_getattr


def maybe_prevent_cross_source_caching(expr, storage):
    from xorq.expr.relations import (
        into_backend,
    )

    if storage.storage.source != expr._find_backend():
        expr = into_backend(expr, storage.storage.source)
    return expr
