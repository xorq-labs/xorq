import ibis.expr.types as ir

from letsql.internal import AbstractTableProvider


class IbisTableProvider(AbstractTableProvider):
    def __init__(self, table: ir.Table):
        self.table = table

    def schema(self):
        return self.table.schema().to_pyarrow()

    def scan(self, filters=None):
        table = self.table
        if filters:
            table = self.table.filter(filters)

        return table.to_pyarrow_batches()
