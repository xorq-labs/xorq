from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlglot as sg

from xorq.vendor.ibis import Schema
from xorq.vendor.ibis.backends.sqlite import Backend as IbisSQLiteBackend
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import gen_name


if TYPE_CHECKING:
    import pyarrow as pa


class Backend(IbisSQLiteBackend):
    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        table_name = table_name or gen_name("read_record_batches")
        schema = Schema.from_pyarrow(record_batches.schema)
        table = sg.table(table_name, quoted=self.compiler.quoted, catalog="temp")
        create_stmt = self._generate_create_table(table, schema).sql(self.name)
        df = record_batches.read_pandas()

        data = df.itertuples(index=False)
        insert_stmt = self._build_insert_template(
            table_name, schema=schema, catalog="temp", columns=True
        )

        with self.begin() as cur:
            cur.execute(create_stmt)
            cur.executemany(insert_stmt, data)

        return self.table(table_name)
