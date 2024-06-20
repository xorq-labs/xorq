from __future__ import annotations

import functools
import operator
import pathlib
from abc import (
    ABC,
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
from cloudpickle import (
    dump as _dump,
)
from cloudpickle import (
    load as _load,
)

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


KEY_PREFIX = "letsql_cache-"


def dump(obj, path):
    with path.open("wb") as fh:
        _dump(obj, fh)


def load(path):
    with path.open("rb") as fh:
        return _load(fh)


class CacheStorage(ABC):
    def exists(self, expr: ir.Expr):
        key = self.get_key(expr)
        return self.key_exists(key)

    @abstractmethod
    def key_exists(self, key):
        pass

    def get_key(self, expr: ir.Expr):
        return KEY_PREFIX + dask.base.tokenize(expr)

    def get(self, expr: ir.Expr):
        key = self.get_key(expr)
        if not self.key_exists(key):
            raise KeyError
        else:
            return self._get(key)

    def set_default(self, expr: ir.Expr, default):
        key = self.get_key(expr)
        if not self.key_exists(key):
            return self._put(key, default)
        else:
            return self._get(key)

    @abstractmethod
    def _get(self, key):
        pass

    def put(self, expr: ir.Expr, value):
        key = self.get_key(expr)
        if self.key_exists(key):
            raise ValueError
        else:
            key = self.get_key(expr)
            return self._put(key, value)

    @abstractmethod
    def _put(self, key, value):
        pass

    def drop(self, expr: ir.Expr):
        key = self.get_key(expr)
        if not self.key_exists(key):
            raise KeyError
        else:
            self._drop(key)

    @abstractmethod
    def _drop(self, key):
        pass


@frozen
class ParquetCacheStorage(CacheStorage):
    source = field(validator=instance_of(ibis.backends.BaseBackend))
    path = field(
        validator=instance_of(pathlib.Path),
        converter=abs_path_converter,
        factory=functools.partial(letsql.options.get, "cache_default_path"),
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
        value.to_expr().to_parquet(loc)
        return self._get(key)

    def _drop(self, key):
        path = self.get_loc(key)
        path.unlink()
        # FIXME: what to do if table is not registered?
        self.source.drop_table(key)


@frozen
class SourceStorage(CacheStorage):
    source = field(validator=instance_of(ibis.backends.BaseBackend))

    def key_exists(self, key):
        return key in self.source.tables

    def _get(self, key):
        return self.source.table(key).op()

    def _put(self, key, value):
        expr = value.to_expr()
        backends, _ = expr._find_backends()
        if set(backends) == set((self.source,)):
            self.source.create_table(key, expr)
        else:
            self.source.create_table(key, expr.to_pyarrow())
        return self._get(key)

    def _drop(self, key):
        self.source.drop_table(key)


@frozen
class SnapshotStorage(SourceStorage):
    def get_key(self, expr: ir.Expr):
        typs = map(type, expr.ls.backends)
        with patch_normalize_token(*typs, f=self.normalize_backend):
            with patch_normalize_token(
                ops.DatabaseTable, f=self.normalize_databasetable
            ):
                tokenized = dask.base.tokenize(expr)
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
        return dask.base.normalize_token(
            {
                argname: getattr(dt, argname)
                # argnames: name, schema, source, namespace
                for argname in dt.argnames
            }
        )
