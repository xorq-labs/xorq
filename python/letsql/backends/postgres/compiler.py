from __future__ import annotations

from typing import Mapping, Any

from ibis.backends.sql.compilers.postgres import (
    PostgresCompiler as IbisPostgresCompiler,
)
from ibis.expr import types as ir

_UNIX_EPOCH = "1970-01-01T00:00:00Z"


class PostgresCompiler(IbisPostgresCompiler):
    __slots__ = ()

    def to_sqlglot(
        self,
        expr: ir.Expr,
        *,
        limit: str | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
    ):
        op = expr.op()
        from letsql.expr.relations import replace_cache_table

        out = op.map_clear(replace_cache_table)
        return super().to_sqlglot(out.to_expr(), limit=limit, params=params)


compiler = PostgresCompiler()
