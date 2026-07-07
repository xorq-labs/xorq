import xorq.vendor.ibis.expr.types as ir
from xorq.internal import AbstractTableProvider


class IbisTableProvider(AbstractTableProvider):
    # scan() calls back into the source backend synchronously, so a same-backend
    # DataFusion expression re-enters the same connection and can starve the
    # tokio worker pool (issue #1580). Callers must materialize those to native
    # DataFrames before registration; only cross-backend tables reach here.
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
