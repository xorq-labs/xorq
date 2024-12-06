from typing import Mapping, Any

import ibis.expr.types as ir
from ibis.backends.snowflake import Backend as IbisSnowflakeBackend

from letsql.common.utils.graph_utils import replace_fix
from letsql.expr.relations import CachedNode, replace_cache_table


class Backend(IbisSnowflakeBackend):
    _top_level_methods = ("connect_env",)

    @classmethod
    def connect_env(cls, database="SNOWFLAKE_SAMPLE_DATA", schema="TPCH_SF1", **kwargs):
        from letsql.common.utils.snowflake_utils import make_connection

        return make_connection(database=database, schema=schema, **kwargs)

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
        out = op.replace(replace_fix(fn))

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
