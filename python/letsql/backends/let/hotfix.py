import ibis.formats.pyarrow
import ibis.expr.datatypes.core
import ibis.expr.operations as ops
import ibis.expr.types.core
import ibis.expr.types.relations
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)
from ibis.common.exceptions import (
    IbisError,
)

from letsql.common.caching import (
    SourceStorage,
    maybe_prevent_cross_source_caching,
)
from letsql.common.utils.caching_utils import find_backend
from letsql.common.utils.hotfix_utils import (
    hotfix,
    none_tokenized,
)
from letsql.config import (
    _backend_init,
)
from letsql.expr.relations import (
    CachedNode,
    Read,
    RemoteTable,
    replace_cache_table,
)


@frozen
class LETSQLAccessor:
    expr = field(validator=instance_of(ibis.expr.types.core.Expr))
    node_types = (ops.DatabaseTable, ops.SQLQueryResult)

    @property
    def op(self):
        return self.expr.op()

    @property
    def cached_nodes(self):
        def _find(node):
            cached = node.find((CachedNode, RemoteTable))
            for no in cached:
                if isinstance(no, RemoteTable):
                    yield from _find(no.remote_expr.op())
                else:
                    yield from _find(no.parent.op())
                    yield no

        op = self.expr.op()
        return tuple(_find(op))

    @property
    def storage(self):
        if self.is_cached:
            return self.op.storage
        else:
            return None

    @property
    def storages(self):
        return tuple(node.storage for node in self.cached_nodes)

    @property
    def backends(self):
        def _find_backends(expr):
            _backends, _ = expr._find_backends()
            _backends = set(_backends)
            if backend := expr._find_backend():
                _backends.add(backend)

            for node in expr.op().find_topmost(CachedNode):
                _backends.update(_find_backends(node.parent))

            for node in expr.op().find_topmost(RemoteTable):
                _backends.update(_find_backends(node.remote_expr))

            return _backends

        backends = _find_backends(self.expr)

        return tuple(backends)

    @property
    def is_multiengine(self):
        (_, *rest) = set(self.backends)
        return bool(rest)

    @property
    def dts(self):
        nodes = set(self.op.find(self.node_types))
        for node in self.cached_nodes:
            candidates = node.parent.op().find(self.node_types)
            nodes.update(
                candidate
                for candidate in candidates
                if not isinstance(candidate, RemoteTable)
            )

        return tuple(nodes)

    @property
    def is_cached(self):
        return isinstance(self.op, CachedNode)

    @property
    def has_cached(self):
        def _has_cached(node):
            if tuple(node.find_topmost(CachedNode)):
                return True
            elif tables := node.find_topmost(RemoteTable):
                return any(_has_cached(table.remote_expr.op()) for table in tables)
            else:
                return False

        return _has_cached(self.op)

    @property
    def uncached(self):
        if self.has_cached:
            op = self.expr.op()
            return op.map_clear(replace_cache_table).to_expr()
        else:
            return self.expr

    @property
    def uncached_one(self):
        if self.is_cached:
            op = self.expr.op()
            return op.parent
        else:
            return self.expr

    def get_key(self):
        if self.is_cached and (self.exists() or not self.uncached_one.ls.has_cached):
            return self.storage.get_key(self.uncached_one)
        else:
            return None

    def get_keys(self):
        if self.has_cached and self.cached_nodes[0].to_expr().ls.exists():
            # FIXME: yield storage with key
            return tuple(op.to_expr().ls.get_key() for op in self.cached_nodes)
        else:
            return None

    def exists(self):
        if self.is_cached:
            cn = self.op
            return cn.storage.exists(cn.parent)
        else:
            return None


@hotfix(
    ibis.expr.types.core.Expr,
    "_find_backend",
    "9e19dfcd3404a043987ff26dce7a40ad",
)
def _letsql_find_backend(self, *, use_default=True):
    # FIXME: push this into LETSQLAccessor
    try:
        if tuple(op.source for op in self.op().find((Read, CachedNode))):
            current_backend = _backend_init()
        else:
            current_backend = self._find_backend._original(
                self, use_default=use_default
            )
    except IbisError as e:
        if "Multiple backends found" in e.args[0]:
            current_backend = _backend_init()
        else:
            raise e
    return current_backend


@hotfix(
    ibis.expr.types.relations.Table,
    "cache",
    "654b574765abdd475264851b89112881",
)
def letsql_cache(self, storage=None):
    # FIXME: push this into LETSQLAccessor
    try:
        current_backend, _ = find_backend(self.op(), use_default=True)
    except IbisError as e:
        if "Multiple backends found" in e.args[0]:
            current_backend = _backend_init()
        else:
            raise e
    storage = storage or SourceStorage(source=current_backend)
    expr = maybe_prevent_cross_source_caching(self, storage)
    op = CachedNode(
        schema=expr.schema(),
        parent=expr,
        source=current_backend,
        storage=storage,
    )
    return op.to_expr()


@hotfix(
    ibis.expr.types.core.Expr,
    "ls",
    none_tokenized,
)
@property
def ls(self):
    return LETSQLAccessor(self)
