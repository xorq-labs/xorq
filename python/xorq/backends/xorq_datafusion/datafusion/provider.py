from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

import xorq.vendor.ibis.expr.types as ir
from xorq.internal import AbstractTableProvider


if TYPE_CHECKING:
    from xorq.expr import Expr


class IbisTableProvider(AbstractTableProvider):
    # scan() calls back into the backend synchronously, so same-connection
    # DataFusion expressions will trigger a nested tokio runtime panic.
    # Callers must convert those to native DataFrames before registration.
    def __init__(self, table: ir.Table) -> None:
        self.table = table

    def schema(self) -> pa.Schema:
        return self.table.schema().to_pyarrow()

    def scan(self, filters: list[Expr] | None = None) -> pa.ipc.RecordBatchReader:
        table = self.table
        if filters:
            table = self.table.filter(filters)
        backend = table._find_backend()
        return backend.to_pyarrow_batches(table)
