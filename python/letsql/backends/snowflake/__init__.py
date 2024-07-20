from typing import Mapping, Any

import ibis.expr.types as ir
from ibis.backends.snowflake import Backend as IbisSnowflakeBackend

from letsql.common.caching import (
    SourceStorage,
)
from letsql.expr.relations import CachedNode, replace_cache_table


class Backend(IbisSnowflakeBackend):
    @staticmethod
    def _register_and_transform_cache_tables(expr):
        """This function will sequentially execute any cache node that is not already cached"""

        def fn(node, _, **kwargs):
            node = node.__recreate__(kwargs)
            if isinstance(node, CachedNode):
                uncached, storage = node.parent, node.storage
                uncached_to_expr = uncached.to_expr()
                node = storage.set_default(uncached_to_expr, uncached)
            return node

        op = expr.op()
        out = op.replace(fn)

        return out.to_expr()

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping | None = None,
        limit: str | None = "default",
        **kwargs: Any,
    ) -> Any:
        expr = self._register_and_transform_cache_tables(expr)
        return super().execute(expr, params=params, limit=limit, **kwargs)

    def _to_sqlglot(
        self, expr: ir.Expr, *, limit: str | None = None, params=None, **_: Any
    ):
        op = expr.op()
        out = op.map_clear(replace_cache_table)

        return super()._to_sqlglot(out.to_expr(), limit=limit, params=params)

    def _cached(self, expr: ir.Table, storage=None):
        storage = storage or SourceStorage(self)
        op = CachedNode(
            schema=expr.schema(),
            parent=expr.op(),
            source=self,
            storage=storage,
        )
        return op.to_expr()
