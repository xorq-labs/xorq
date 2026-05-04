import xorq.vendor.ibis.expr.types as ir
from xorq.internal import AbstractTableProvider


class IbisTableProvider(AbstractTableProvider):
    # scan() calls back into the backend synchronously, so same-connection
    # DataFusion expressions will trigger a nested tokio runtime panic.
    # Callers must convert those to native DataFrames before registration.
    def __init__(self, table: ir.Table):
        self.table = table

    def schema(self):
        return self.table.schema().to_pyarrow()

    def scan(self, filters=None):
        table = self.table
        if filters:
            table = self.table.filter(filters)
        backend = table._find_backend()
        return backend.to_pyarrow_batches(table)
