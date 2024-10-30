from ibis.backends.sql.compilers.duckdb import DuckDBCompiler as IbisDuckDBCompiler

import ibis.expr.operations as ops
import sqlglot as sg


class DuckDBCompiler(IbisDuckDBCompiler):
    __slots__ = ()

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
