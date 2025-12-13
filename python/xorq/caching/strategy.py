from __future__ import annotations

import contextlib
import functools
from abc import (
    abstractmethod,
)

import dask
from attr import (
    frozen,
)

import xorq.common.utils.dask_normalize  # noqa: F401
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.dask_normalize.dask_normalize_expr import (
    normalize_backend,
    normalize_read,
    normalize_remote_table,
)
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    patch_normalize_token,
)
from xorq.expr.relations import (
    Read,
    RemoteTable,
)
from xorq.vendor.ibis.expr import types as ir


@frozen
class CacheStrategy:
    @abstractmethod
    def calc_key(self, expr):
        pass


@frozen
class ModificationTimeStrategy(CacheStrategy):
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
        ### hack: begin: deal with patch_normalize_token side effect
        import numpy as np

        dask.base.tokenize(np.dtypes.Float64DType())
        ### hack: end: deal with patch_normalize_token side effect

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
