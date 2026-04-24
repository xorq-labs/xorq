from typing import Any, Mapping

import pyarrow as pa

from xorq.vendor.ibis.backends.gizmosql import Backend as IbisGizmoSQLBackend
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import gen_name


class Backend(IbisGizmoSQLBackend):
    def tokenize_table(self, dt):
        from xorq.common.utils.dask_normalize.dask_normalize_expr import (  # noqa: PLC0415
            normalize_seq_with_caller,
        )

        return normalize_seq_with_caller(
            dt.name,
            dt.schema,
            dt.source,
            dt.namespace,
            caller="normalize_remote_databasetable",
        )

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping | None = None,
        limit: str | None = "default",
        **_: Any,
    ) -> Any:
        self._run_pre_execute_hooks(expr)
        table = self._to_pyarrow_table(expr, params=params, limit=limit)
        return expr.__pandas_result__(table.to_pandas(timestamp_as_object=True))

    def read_record_batches(self, source, table_name=None):
        table_name = table_name or gen_name("read_record_batches")
        source = self._normalize_arrow_schema(pa.Table.from_batches(source))
        batches = source.to_batches(max_chunksize=10_000)
        reader = pa.RecordBatchReader.from_batches(source.schema, batches)
        with self.con.cursor() as cur:
            cur.adbc_ingest(table_name, reader, mode="replace")
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
        self._run_pre_execute_hooks(expr)
        table = self._to_pyarrow_table(expr, params=params, limit=limit)
        batches = table.to_batches(max_chunksize=chunk_size)
        return pa.ipc.RecordBatchReader.from_batches(table.schema, batches)
