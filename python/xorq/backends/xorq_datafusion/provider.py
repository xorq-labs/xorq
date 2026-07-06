import xorq.vendor.ibis.expr.types as ir
from xorq.internal import AbstractTableProvider


class IbisTableProvider(AbstractTableProvider):
    # scan() calls back into the source backend synchronously. xorq-datafusion
    # supports a runtime-within-runtime, so this re-entry is safe even when the
    # source is the same DataFusion connection (no nested tokio runtime panic).
    def __init__(self, table: ir.Table) -> None:
        self.table = table

    def schema(self):
        return self.table.schema().to_pyarrow()

    def scan(self, filters=None):
        table = self.table
        if filters:
            table = self.table.filter(filters)
        backend = table._find_backend()
        return backend.to_pyarrow_batches(table)
