from __future__ import annotations

import contextlib
import functools
import pathlib
from abc import (
    abstractmethod,
)

import dask
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)

import xorq.common.utils.dask_normalize  # noqa: F401
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.dask_normalize.dask_normalize_expr import (
    normalize_backend,
    normalize_remote_table,
)
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_seq_with_caller,
    patch_normalize_op_caching,
    patch_normalize_token,
)
from xorq.config import options
from xorq.expr.relations import (
    Read,
    RemoteTable,
)
from xorq.vendor.ibis.expr import types as ir


def snapshot_normalize_read(read):
    """Normalize Read for snapshot caching using path identity only, not file modification stats."""
    read_kwargs = dict(read.read_kwargs)
    # Materialized build-bundle reads carry a content-hash-named read_path that is
    # stable across environments. Their hash_path is an absolute tmpdir path that
    # changes every run, so prefer read_path when available.
    path = read_kwargs.get("read_path") or read_kwargs["hash_path"]
    match path:
        case list() | tuple() if len(path) == 1:
            tpls = (("path", str(path[0])),)
        case list() | tuple():
            tpls = (("paths", tuple(str(p) for p in path)),)
        case str() | pathlib.Path():
            tpls = (("path", str(path)),)
        case _:
            raise NotImplementedError(f'Don\'t know how to deal with path "{path}"')
    tpls += tuple(
        (k, v) for k, v in read.read_kwargs if k in ("mode", "schema", "temporary")
    )
    return normalize_seq_with_caller(
        read.schema,
        tpls,
        caller="snapshot_normalize_read",
    )


def _rename_remote_table(node, kwargs):
    if isinstance(node, RemoteTable):
        # FIXME: how to verify that we're always within a normalization_context?
        name = dask.base.tokenize(node)
        return RemoteTable(
            name=name,
            schema=node.schema,
            source=node.source,
            remote_expr=node.remote_expr,
            namespace=node.namespace,
        )
    # kwargs is None when no children were rewritten (graph.py convention)
    return node.__recreate__(kwargs) if kwargs else node


@frozen
class CacheStrategy:
    key_prefix = field(
        validator=instance_of(str),
        factory=functools.partial(options.get, "cache.key_prefix"),
    )

    @abstractmethod
    def calc_key(self, expr):
        pass

    def __dask_tokenize__(self):
        return (type(self).__name__,)


@frozen
class ModificationTimeStrategy(CacheStrategy):
    def calc_key(self, expr: ir.Expr):
        return self.key_prefix + expr.ls.tokenized


@frozen
class SnapshotStrategy(CacheStrategy):
    def calc_key(self, expr: ir.Expr):
        with self.normalization_context(expr):
            replaced = self.replace_remote_table(expr)
            tokenized = replaced.ls.tokenized
            return self.key_prefix + "-".join(("snapshot", tokenized))

    @contextlib.contextmanager
    def normalization_context(self, expr):
        # patch_normalize_op_caching memoizes normalize_op by (op, compiler).
        # Without it, tokenizing depth-n pipeline expressions is O(2^n) because
        # normalize_remote_table recursively tokenizes remote_expr and
        # normalize_scalar_udf recursively tokenizes computed_kwargs_expr, both
        # of which share sub-expressions that get re-tokenized without caching.
        typs = map(type, expr.ls.backends)
        with patch_normalize_op_caching():
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

    def replace_remote_table(self, expr):
        """replace remote table with deterministic name ***strictly for key calculation***"""
        if expr.op().find(RemoteTable):
            expr = self.cached_replace_remote_table(expr.op()).to_expr()
        return expr

    @staticmethod
    @functools.cache
    def cached_normalize_read(op):
        return snapshot_normalize_read(op)

    @staticmethod
    @functools.cache
    def cached_replace_remote_table(op):
        return op.replace(_rename_remote_table)

    @staticmethod
    def normalize_backend(con):
        name = con.name
        if name in ("pandas", "duckdb", "datafusion", "xorq_datafusion"):
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
