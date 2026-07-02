from typing import Any, Mapping

import pyarrow as pa
from batchcorder import StreamCache

from xorq.common.utils.rbr_utils import coerce_to_arrow_table
from xorq.vendor.ibis.backends.gizmosql import Backend as IbisGizmoSQLBackend
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import gen_name


__all__ = [
    "Backend",
]


class Backend(IbisGizmoSQLBackend):
    def publish_strategy(self, mode):
        """Incremental WAP publish mechanism (ADR-0017): DuckDB-backed, duckdb dialect."""
        from xorq.writes.enums import PublishStrategy  # noqa: PLC0415

        return PublishStrategy.NATIVE_MERGE

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

    def read_record_batches(
        self,
        source: pa.Table | pa.RecordBatchReader | StreamCache,
        table_name: str | None = None,
    ) -> ir.Table:
        table_name = table_name or gen_name("read_record_batches")
        # coerce_to_arrow_table carries the schema through, so an empty stream
        # (zero batches) still materializes the declared columns instead of
        # raising "Must pass schema, or at least one RecordBatch".
        table = self._normalize_arrow_schema(coerce_to_arrow_table(source))
        batches = table.to_batches(max_chunksize=10_000)
        # An empty table yields zero batches, but the ADBC Flight SQL ingest
        # rejects a stream with no messages ("Stream finished before first
        # message sent"). Send a single zero-row batch so the table is still
        # created with the right schema.
        if not batches:
            batches = [pa.RecordBatch.from_pylist([], schema=table.schema)]
        reader = pa.RecordBatchReader.from_batches(table.schema, batches)
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
