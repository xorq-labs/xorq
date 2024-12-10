from contextlib import contextmanager
from typing import Mapping, Any

from ibis.backends.duckdb import Backend as IbisDuckDBBackend
from ibis.expr import types as ir

from letsql.backends.duckdb.compiler import DuckDBCompiler
from letsql.expr.relations import register_and_transform_remote_tables


@contextmanager
def _transform_expr(expr):
    expr, created = register_and_transform_remote_tables(expr)
    yield expr
    for table, con in created.items():
        try:
            con.drop_table(table)
        except Exception:
            con.drop_view(table)


class Backend(IbisDuckDBBackend):
    compiler = DuckDBCompiler()

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping | None = None,
        limit: str | None = "default",
        **_: Any,
    ) -> Any:
        with _transform_expr(expr) as expr:
            return super().execute(expr, params=params, limit=limit)
