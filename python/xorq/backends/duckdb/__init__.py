from typing import Any, Mapping

import pyarrow as pa

from xorq.vendor.ibis.backends.duckdb import Backend as IbisDuckDBBackend
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import gen_name


class Backend(IbisDuckDBBackend):
    def tokenize_table(self, dt):
        import re  # noqa: PLC0415

        import sqlglot as sg  # noqa: PLC0415

        from xorq.common.utils.dask_normalize.dask_normalize_expr import (  # noqa: PLC0415
            normalize_duckdb_file_read,
            normalize_memory_databasetable,
        )

        name = sg.table(dt.name, quoted=dt.source.compiler.quoted).sql(
            dialect=dt.source.name
        )
        ((_, plan),) = dt.source.raw_sql(f"EXPLAIN SELECT * FROM {name}").fetchall()
        scan_line = plan.split("\n")[1]
        execution_plan_name = r"\s*│\s*(\w+)\s*│\s*"
        match re.match(execution_plan_name, scan_line).group(1):
            case "ARROW_SCAN" | "PANDAS_SCAN":
                return normalize_memory_databasetable(dt)
            case "READ_PARQUET" | "READ_CSV" | "SEQ_SCAN":
                return normalize_duckdb_file_read(dt)
            case _:
                raise NotImplementedError(scan_line)

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
