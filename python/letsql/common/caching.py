from __future__ import annotations

import functools
import operator
import pathlib
from abc import (
    abstractmethod,
)

import dask
import ibis
import ibis.expr.operations as ops
import toolz
from ibis.expr import types as ir
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)

import letsql as ls
import letsql.common.utils.dask_normalize  # noqa: F401
from letsql.common.utils.dask_normalize import (
    patch_normalize_token,
)
from letsql.common.utils.dask_normalize_expr import (
    normalize_backend,
)


abs_path_converter = toolz.compose(
    operator.methodcaller("expanduser"), operator.methodcaller("absolute"), pathlib.Path
)


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
        factory=functools.partial(letsql.options.get, "cache.key_prefix"),
    )

    def exists(self, expr):
        key = self.get_key(expr)
        return self.storage.key_exists(key)

    def key_exists(self, key):
        return self.storage.key_exists(key)

    def get_key(self, expr):
        # FIXME: let strategy solely determine key
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

    def set_default(self, expr: ir.Expr, default):
        key = self.get_key(expr)
        if not self.key_exists(key):
            return self.storage._put(key, default)
        else:
            return self.storage._get(key)

    def drop(self, expr: ir.Expr):
        key = self.get_key(expr)
        if not self.key_exists(key):
            raise KeyError
        else:
            self.storage._drop(key)


@frozen
class ModificationTimeStragegy(CacheStrategy):
    key_prefix = field(
        validator=instance_of(str),
        factory=functools.partial(letsql.options.get, "cache.key_prefix"),
    )

    def get_key(self, expr: ir.Expr):
        return self.key_prefix + dask.base.tokenize(expr)


@frozen
class SnapshotStrategy(CacheStrategy):
    def get_key(self, expr: ir.Expr):
        typs = map(type, expr.ls.backends)
        with patch_normalize_token(*typs, f=self.normalize_backend):
            with patch_normalize_token(
                ops.DatabaseTable, f=self.normalize_databasetable
            ):
                tokenized = dask.tokenize.tokenize(expr)
                return "-".join(("snapshot", tokenized))

    @staticmethod
    def normalize_backend(con):
        name = con.name
        if name in ("pandas", "duckdb", "datafusion"):
            return (name, None)
        else:
            return normalize_backend(con)

    @staticmethod
    def normalize_databasetable(dt):
        keys = ["name", "schema", "source", "namespace"]
        return dask.tokenize._normalize_seq_func(
            (key, getattr(dt, key)) for key in keys
        )


@frozen
class ParquetStorage(CacheStorage):
    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=letsql.config._backend_init,
    )
    path = field(
        validator=instance_of(pathlib.Path),
        converter=abs_path_converter,
        factory=functools.partial(letsql.options.get, "cache.default_path"),
    )

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
        ls.to_parquet(value.to_expr(), loc)
        return self._get(key)

    def _drop(self, key):
        path = self.get_loc(key)
        path.unlink()
        # FIXME: what to do if table is not registered?
        self.source.drop_table(key)


# named with underscore prefix until we swap out SourceStorage
@frozen
class _SourceStorage(CacheStorage):
    _source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=letsql.config._backend_init,
    )

    def key_exists(self, key):
        return key in self._source.tables

    def _get(self, key):
        return self._source.table(key).op()

    def _put(self, key, value):
        expr = value.to_expr()
        backends, _ = expr._find_backends()
        # FIXME what happens when the backend is LETSQL, to_pyarrow won't work
        if (
            len(backends) == 1
            and backends[0].name != "pandas"
            and backends[0] is self._source
        ):
            self._source.create_table(key, expr)
        else:
            self._source.create_table(key, ls.to_pyarrow(expr))
        return self._get(key)

    def _drop(self, key):
        self._source.drop_table(key)


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


@frozen
class ParquetSnapshot:
    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=letsql.config._backend_init,
    )
    path = field(
        validator=instance_of(pathlib.Path),
        converter=abs_path_converter,
        factory=functools.partial(letsql.options.get, "cache.default_path"),
    )
    cache = field(validator=instance_of(Cache), init=False)

    def __attrs_post_init__(self):
        cache = Cache(
            strategy=SnapshotStrategy(),
            storage=ParquetStorage(
                self.source,
                self.path,
            ),
        )
        object.__setattr__(self, "cache", cache)

    __getattr__ = chained_getattr


@frozen
class ParquetCacheStorage:
    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=letsql.config._backend_init,
    )
    path = field(
        validator=instance_of(pathlib.Path),
        converter=abs_path_converter,
        factory=functools.partial(letsql.options.get, "cache.default_path"),
    )
    cache = field(validator=instance_of(Cache), init=False)

    def __attrs_post_init__(self):
        cache = Cache(
            strategy=ModificationTimeStragegy(),
            storage=ParquetStorage(
                self.source,
                self.path,
            ),
        )
        object.__setattr__(self, "cache", cache)

    __getattr__ = chained_getattr


@frozen
class SourceStorage:
    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=letsql.config._backend_init,
    )
    cache = field(validator=instance_of(Cache), init=False)

    def __attrs_post_init__(self):
        cache = Cache(
            strategy=ModificationTimeStragegy(), storage=_SourceStorage(self.source)
        )
        object.__setattr__(self, "cache", cache)

    __getattr__ = chained_getattr


@frozen
class SnapshotStorage:
    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=letsql.config._backend_init,
    )
    cache = field(validator=instance_of(Cache), init=False)

    def __attrs_post_init__(self):
        cache = Cache(strategy=SnapshotStrategy(), storage=_SourceStorage(self.source))
        object.__setattr__(self, "cache", cache)

    __getattr__ = chained_getattr
