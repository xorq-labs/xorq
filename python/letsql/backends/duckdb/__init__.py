from typing import Mapping, Any

from ibis.backends.duckdb import Backend as IbisDuckDBBackend
from ibis.expr import types as ir

from letsql.backends.duckdb.compiler import DuckDBCompiler
from letsql.expr.relations import RemoteTableReplacer


class Backend(IbisDuckDBBackend):
    compiler = DuckDBCompiler()

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping | None = None,
        limit: str | None = "default",
        **_: Any,
    ) -> Any:
        expr = expr.op().replace(RemoteTableReplacer()).to_expr()
        return super().execute(expr, params=params, limit=limit)
