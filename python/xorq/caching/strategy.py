from __future__ import annotations

import contextlib
import functools
import pathlib
from abc import (
    abstractmethod,
)

from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)
from xorq_dasher.rules.expr import normalize_remote_table

import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.dasher import (
    HASHER,
    fqn,
    snapshot_hasher,
    tokenize,
)
from xorq.config import options
from xorq.expr.relations import (
    Read,
    RemoteTable,
)
from xorq.vendor import ibis
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
    return ("snapshot_normalize_read", read.schema, tpls)


@frozen
class CacheStrategy:
    key_prefix = field(
        validator=instance_of(str),
        factory=functools.partial(options.get, "cache.key_prefix"),
    )

    @abstractmethod
    def calc_key(self, expr):
        pass

    def __dasher_tokenize__(self):
        return (type(self).__name__, self.key_prefix)


@frozen
class ModificationTimeStrategy(CacheStrategy):
    def calc_key(self, expr: ir.Expr):
        return self.key_prefix + expr.ls.tokenized


@frozen
class SnapshotStrategy(CacheStrategy):
    def calc_key(self, expr: ir.Expr):
        with self.normalization_context(expr) as local:
            replaced = self._replace_remote_table(expr, local)
            tokenized = local.tokenize(replaced)
            return self.key_prefix + "-".join(("snapshot", tokenized))

    @contextlib.contextmanager
    def normalization_context(self, expr):
        """Yield a snapshot-flavored Hasher; callers tokenize through it.

        Replaces the previous dask-monkeypatching context manager: instead of
        swapping global normalizers, we hand out a per-call hasher whose rules
        override DatabaseTable/Read/backend normalization.
        """
        yield self._build_hasher(expr)

    def _build_hasher(self, expr):
        extra = [
            (fqn(ibis.backends.BaseBackend), self.normalize_backend),
            (fqn(ops.DatabaseTable), self.normalize_databasetable),
            (fqn(Read), snapshot_normalize_read),
        ]
        # Each concrete backend subclass on the expression also needs the
        # snapshot backend rule registered against its concrete FQN, otherwise
        # the MRO lookup picks the more-specific subclass and bypasses our
        # override on BaseBackend.
        for backend in expr.ls.backends:
            extra.append((fqn(type(backend)), self.normalize_backend))
        return snapshot_hasher(*extra)

    def _replace_remote_table(self, expr, local_hasher):
        if expr.op().find(RemoteTable):

            def rename(node, kwargs):
                if isinstance(node, RemoteTable):
                    return RemoteTable(
                        name=local_hasher.tokenize(node),
                        schema=node.schema,
                        source=node.source,
                        remote_expr=node.remote_expr,
                        namespace=node.namespace,
                    )
                return node.__recreate__(kwargs) if kwargs else node

            return expr.op().replace(rename).to_expr()
        return expr

    @staticmethod
    def normalize_backend(con):
        name = con.name
        if name in ("pandas", "duckdb", "datafusion", "xorq_datafusion"):
            return (name, None)
        return HASHER.normalize(con)

    @staticmethod
    def normalize_databasetable(dt):
        from xorq_dasher.rules.expr import normalize_cached_node  # noqa: PLC0415

        from xorq.expr.relations import CachedNode  # noqa: PLC0415

        # Read and CachedNode are subclasses of DatabaseTable. Dasher's
        # earliest-match-wins MRO lookup picks this DatabaseTable rule over
        # the more specific Read/CachedNode rules, so we must isinstance-
        # dispatch here or those subclasses get a wrong (path/parent-blind)
        # normalization.
        if isinstance(dt, Read):
            return snapshot_normalize_read(dt)
        if isinstance(dt, CachedNode):
            return normalize_cached_node(dt)
        if isinstance(dt, RemoteTable):
            return normalize_remote_table(dt)
        keys = ("name", "schema", "source", "namespace")
        return tuple((k, getattr(dt, k)) for k in keys)


__all__ = [
    "CacheStrategy",
    "ModificationTimeStrategy",
    "SnapshotStrategy",
    "snapshot_normalize_read",
    "tokenize",
]
