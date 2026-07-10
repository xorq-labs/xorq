from __future__ import annotations

from typing import Any, Mapping

import pyarrow as pa
from batchcorder import StreamCache

from xorq.vendor.ibis.backends.duckdb import Backend as IbisDuckDBBackend
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import gen_name


__all__ = [
    "Backend",
]


class Backend(IbisDuckDBBackend):
    def publish_strategy(self):
        """Incremental WAP publish mechanism for this backend (ADR-0017)."""
        from xorq.writes.enums import PublishStrategy  # noqa: PLC0415

        return PublishStrategy.NATIVE_MERGE

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

    def read_record_batches(
        self,
        source: pa.Table | pa.RecordBatchReader | StreamCache,
        table_name: str | None = None,
    ) -> ir.Table:
        # duckdb registers ``source`` (typically a StreamCache) directly so it
        # can replay the stream across scans; a casting wrapper would not be
        # replayable, so casting to the logical schema happens upstream, before
        # the StreamCache, in register_and_transform_remote_tables.
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
