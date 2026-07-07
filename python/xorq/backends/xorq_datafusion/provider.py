import xorq.vendor.ibis.expr.types as ir
from xorq.internal import AbstractTableProvider


class IbisTableProvider(AbstractTableProvider):
    # scan() calls back into the source backend synchronously, so a same-backend
    # DataFusion expression re-enters the same connection and can starve the
    # tokio worker pool (issue #1580). register() sidesteps this by materializing
    # same-backend exprs to native DataFrames before registration.
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
