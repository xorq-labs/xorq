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

import letsql
from letsql.expr.relations import (
    CachedNode,
    replace_cache_table,
)
from letsql.common.utils.hotfix_utils import (
    maybe_hotfix,
    none_tokenized,
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
        return self.op.find(CachedNode)

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
    def _backends(self):
        (backends, _) = self.expr._find_backends()
        return backends

    @property
    def is_letsql(self):
        names = set(backend.name for backend in self._backends)
        if letsql.backends.let.Backend.name in names:
            if len(names) > 1:
                raise ValueError
            else:
                return True
        else:
            return False

    @property
    def ls_con(self):
        if self.is_letsql:
            (con,) = self._backends
            return con
        else:
            return None

    @property
    def backends(self):
        if self.is_letsql:
            return tuple(
                set((self.ls_con,) + tuple(dt.source for dt in self.native_dts))
            )
        else:
            return self._backends

    @property
    def is_multiengine(self):
        (_, *rest) = set(self.backends)
        return bool(rest)

    @property
    def dts(self):
        return self.op.find(self.node_types)

    @property
    def native_dts(self):
        return tuple(self.native_expr.op().find(self.node_types))

    @property
    def native_expr(self):
        native_expr = self.expr
        if self.is_letsql:
            _sources = self.ls_con._sources

            def replace_table(_node, _, **_kwargs):
                return _sources.get_table_or_op(_node, _node.__recreate__(_kwargs))

            native_expr = self.op.replace(replace_table).to_expr()
        return native_expr

    @property
    def is_cached(self):
        return isinstance(self.op, CachedNode)

    @property
    def has_cached(self):
        return bool(tuple(self.op.find_topmost(CachedNode)))

    @property
    def uncached(self):
        if self.has_cached:
            return self.op.map_clear(replace_cache_table).to_expr()
        else:
            return self.expr

    @property
    def uncached_one(self):
        if self.is_cached:
            return self.op.parent.to_expr()
        else:
            return self.expr

    def get_key(self):
        if self.is_cached:
            return self.storage.get_key(
                self.ls_con._register_and_transform_cache_tables(
                    self.native_expr.uncached_one
                )
            )
        else:
            return None

    def exists(self):
        if self.is_cached:
            con = self.ls_con
            return all(
                cn.storage.exists(
                    con._register_and_transform_cache_tables(cn.parent.to_expr())
                )
                for cn in self.native_expr.op().find(CachedNode)[::-1]
            )
        else:
            return None


@maybe_hotfix(
    ibis.expr.types.relations.Table,
    "cache",
    "86f5c6be5577d4d65b53025d151bdd94",
)
def letsql_cache(self, storage=None):
    current_backend = self._find_backend(use_default=True)
    return current_backend._cached(self, storage=storage)


@maybe_hotfix(
    ibis.expr.types.core.Expr,
    "ls",
    none_tokenized,
)
@property
def ls(self):
    return LETSQLAccessor(self)
