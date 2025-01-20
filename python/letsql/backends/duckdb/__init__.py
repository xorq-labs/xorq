from contextlib import contextmanager
from typing import Mapping, Any

from ibis.backends.duckdb import Backend as IbisDuckDBBackend
from ibis.expr import types as ir
from ibis.util import gen_name

from letsql.backends.duckdb.compiler import DuckDBCompiler
from letsql.expr.relations import register_and_transform_remote_tables

import pyarrow as pa


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
        batch_reader = self.to_pyarrow_batches(expr, params=params, limit=limit)
        return expr.__pandas_result__(
            batch_reader.read_pandas(timestamp_as_object=True)
        )

    def read_record_batches(self, source, table_name=None):
        table_name = table_name or gen_name("read_record_batches")
        self.con.register(table_name, source)
        return self.table(table_name)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 10_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        return self._to_duckdb_relation(
            expr, params=params, limit=limit
        ).fetch_arrow_reader(chunk_size)
