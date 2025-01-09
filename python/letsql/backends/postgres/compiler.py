from __future__ import annotations

from typing import Mapping, Any

from ibis.backends.sql.compilers.postgres import (
    PostgresCompiler as IbisPostgresCompiler,
)
from ibis.expr import types as ir
import ibis.expr.operations as ops
import sqlglot as sg

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

    def visit_MarkedRemoteTable(
        self,
        op,
        *,
        name: str,
        schema,
        source,
        namespace: ops.Namespace,
        remote_expr,
    ):
        return sg.table(
            name, db=namespace.database, catalog=namespace.catalog, quoted=self.quoted
        )


compiler = PostgresCompiler()
