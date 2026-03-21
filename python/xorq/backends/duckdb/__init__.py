from typing import Any, Mapping

import pyarrow as pa

from xorq.vendor.ibis.backends.duckdb import Backend as IbisDuckDBBackend
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import gen_name


class Backend(IbisDuckDBBackend):
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
        isolated: bool = False,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        if isolated:
            # Use a dedicated cursor so that multiple concurrent readers
            # from the same connection don't invalidate each other.
            # DuckDB only supports one active streaming result per
            # connection handle; a second con.sql() silently exhausts
            # the first.
            self._run_pre_execute_hooks(expr)
            sql = self.compile(expr.as_table(), limit=limit, params=params)
            cursor = self.con.cursor()
            return cursor.sql(sql).fetch_arrow_reader(chunk_size)
        return self._to_duckdb_relation(
            expr, params=params, limit=limit
        ).fetch_arrow_reader(chunk_size)
