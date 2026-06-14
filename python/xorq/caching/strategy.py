from __future__ import annotations

import contextlib
import contextvars
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
from xorq_dasher.rules.expr import (
    normalize_cached_node,
    normalize_remote_table,
)

import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.dasher import (
    HASHER,
    fqn,
    snapshot_hasher,
    with_caches,
)
from xorq.config import options
from xorq.expr.relations import (
    CachedNode,
    Read,
    RemoteTable,
)
from xorq.vendor import ibis
from xorq.vendor.ibis.expr import types as ir


# Per-outer-call memo for ``SnapshotStrategy.normalize_databasetable``.
# Mirrors ``_dt_normalize_memo`` in ``_relations.py`` but kept separate so
# snapshot-flavored DT results don't alias on the same ``dt`` key used by
# the global-hasher dispatcher.
_snapshot_dt_normalize_memo: contextvars.ContextVar[dict | None] = (
    contextvars.ContextVar("_xorq_snapshot_dt_normalize_memo", default=None)
)


def snapshot_normalize_read(read):
    """Normalize Read for snapshot caching using path identity only, not file modification stats."""
    read_kwargs = dict(read.read_kwargs)
    # Materialized build-bundle reads carry a content-hash-named read_path that is
    # stable across environments. Their hash_path is an absolute tmpdir path that
    # changes every run, so prefer read_path when available.
    read_path = read_kwargs.get("read_path")
    path = read_path if read_path is not None else read_kwargs["hash_path"]
    match path:
        case list() | tuple() if len(path) == 1:
            tpls = (("path", str(path[0])),)
        case list() | tuple():
            tpls = (("paths", tuple(str(p) for p in path)),)
        case str() | pathlib.Path():
            tpls = (("path", str(path)),)
        case _:
            raise NotImplementedError(f'Don\'t know how to deal with path "{path}"')
    tpls += tuple((k, v) for k, v in read.read_kwargs if k in Read.IDENTITY_KEYS)
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
    @with_caches  # must be inner so memos span the full with-block
    def normalization_context(self, expr):
        """Yield a snapshot-flavored Hasher; callers tokenize through it.

        Replaces the previous dask-monkeypatching context manager: instead of
        swapping global normalizers, we hand out a per-call hasher whose rules
        override DatabaseTable/Read/backend normalization. The hasher is also
        installed in ``_current_hasher`` so transitive tokenize calls inside
        the opaque-placeholder replacer (``_parent_token``) propagate the
        snapshot-flavored rules instead of falling back to the data-sensitive
        global HASHER.  Per-call memos are installed alongside so repeated
        visits of the same nodes under deeply nested into_backend chains
        normalize once.
        """
        from xorq.common.utils.dasher import _current_hasher  # noqa: PLC0415

        local = self._build_hasher(expr)
        hasher_token = _current_hasher.set(local)
        memo_token = (
            _snapshot_dt_normalize_memo.set({})
            if _snapshot_dt_normalize_memo.get() is None
            else None
        )
        try:
            yield local
        finally:
            _current_hasher.reset(hasher_token)
            if memo_token is not None:
                _snapshot_dt_normalize_memo.reset(memo_token)

    def _build_hasher(self, expr):
        extra = [
            (fqn(ibis.backends.BaseBackend), self.normalize_backend),
            (fqn(ops.DatabaseTable), self.normalize_databasetable),
            (fqn(Read), snapshot_normalize_read),
            # Each concrete backend subclass on the expression also needs the
            # snapshot backend rule registered against its concrete FQN, otherwise
            # the MRO lookup picks the more-specific subclass and bypasses our
            # override on BaseBackend.
            *(
                (fqn(type(backend)), self.normalize_backend)
                for backend in expr.ls.backends
            ),
        ]
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
        # In-memory backends identified by name alone; remote backends
        # delegate to HASHER.normalize which raises if unregistered.
        name = con.name
        if name in ("pandas", "duckdb", "datafusion", "xorq_datafusion"):
            return (name, None)
        return HASHER.normalize(con)

    @staticmethod
    def normalize_databasetable(dt):
        # DatabaseTable subclasses (Read, CachedNode, RemoteTable) need
        # concrete-type dispatch here — dasher's MRO lookup would otherwise
        # pick this broader DatabaseTable rule over them.
        memo = _snapshot_dt_normalize_memo.get()
        if memo is not None and dt in memo:
            return memo[dt]
        match dt:
            case Read():
                result = snapshot_normalize_read(dt)
            case CachedNode():
                result = normalize_cached_node(dt)
            case RemoteTable():
                result = normalize_remote_table(dt)
            case _:
                keys = ("name", "schema", "source", "namespace")
                result = tuple((k, getattr(dt, k)) for k in keys)
        if memo is not None:
            memo[dt] = result
        return result


__all__ = [
    "CacheStrategy",
    "ModificationTimeStrategy",
    "SnapshotStrategy",
    "snapshot_normalize_read",
]
