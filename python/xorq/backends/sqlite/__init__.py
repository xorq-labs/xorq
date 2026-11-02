from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge

from xorq.common.utils.defer_utils import csv_shape_kwargs, read_csv_schema_kwargs
from xorq.common.utils.rbr_utils import coerce_to_arrow_table
from xorq.config import default_backend
from xorq.vendor.ibis import Schema, util
from xorq.vendor.ibis.backends.sqlite import Backend as IbisSQLiteBackend
from xorq.vendor.ibis.backends.sqlite import _arrow_frame, _bindable_rows, _quote
from xorq.vendor.ibis.expr import types as ir
from xorq.vendor.ibis.util import gen_name


if TYPE_CHECKING:
    import pyarrow as pa


__all__ = [
    "Backend",
]


class Backend(IbisSQLiteBackend):
    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader,
        table_name: str | None = None,
        mode: str = "create",
        overwrite: bool = True,
        **kwargs: Any,
    ) -> ir.Table:
        from xorq.common.utils.sqlite_utils import SQLiteADBC  # noqa: PLC0415

        table_name = table_name or gen_name("read_record_batches")

        in_memory = self.is_in_memory()
        catalog = "temp" if in_memory else None

        if overwrite:
            created_table_name = util.gen_name(f"{self.name}_table")
            created_table = sg.table(
                created_table_name,
                catalog=catalog,
                quoted=self.compiler.quoted,
            )
            table = sg.table(table_name, catalog=catalog, quoted=self.compiler.quoted)
        else:
            created_table_name = table_name
            created_table = table = sg.table(
                table_name, catalog=catalog, quoted=self.compiler.quoted
            )

        if in_memory:
            self._into_memory_record_batches(record_batches, created_table_name)
        else:
            sqlite_adbc = SQLiteADBC(self)
            sqlite_adbc.adbc_ingest(
                created_table_name, record_batches, mode=mode, **kwargs
            )

        with self.begin() as cur:
            if overwrite:
                cur.execute(
                    sge.Drop(kind="TABLE", this=table, exists=True).sql(self.name)
                )
                # SQLite's ALTER TABLE statement doesn't support using a
                # fully-qualified table reference after RENAME TO. Since we
                # never rename between databases, we only need the table name
                # here.
                quoted_name = _quote(table_name)
                cur.execute(
                    f"ALTER TABLE {created_table.sql(self.name)} RENAME TO {quoted_name}"
                )

        return self.table(table_name)

    def _into_memory_record_batches(
        self, record_batches: pa.RecordBatchReader, table_name: str
    ) -> None:
        schema = Schema.from_pyarrow(record_batches.schema)
        table = sg.table(table_name, quoted=self.compiler.quoted, catalog="temp")
        create_stmt = self._generate_create_table(table, schema).sql(self.name)
        data = _bindable_rows(_arrow_frame(coerce_to_arrow_table(record_batches)))
        insert_stmt = self._build_insert_template(
            table_name, schema=schema, catalog="temp", columns=True
        )
        with self.begin() as cur:
            cur.execute(create_stmt)
            cur.executemany(insert_stmt, data)

    def _post_connect(self, *args, **kwargs) -> None:
        super()._post_connect(*args, **kwargs)
        # sqlite reports no file for a database held in memory, whichever
        # spelling opened it (":memory:", "file::memory:", "mode=memory").
        # Asked once here: is_in_memory runs during expression hashing, which
        # may be off the connecting thread, where sqlite3 refuses its objects.
        query = "SELECT file FROM pragma_database_list WHERE name = 'main'"
        with closing(self.con.execute(query)) as cur:
            (file,) = cur.fetchone()
        self._in_memory = not file

    def is_in_memory(self):
        return self._in_memory

    def read_parquet(
        self,
        path: str | Path,
        table_name: str | None = None,
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        table_name = table_name or gen_name("xo_read_parquet")
        record_batches = default_backend().read_parquet(path).to_pyarrow_batches()
        return self.read_record_batches(record_batches, table_name, mode, **kwargs)

    def read_csv(
        self,
        path: str | Path,
        table_name: str | None = None,
        mode: str = "create",
        schema: Schema | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        table_name = table_name or gen_name("xo_read_csv")
        reader = default_backend()
        # column selection and naming go to the reader, the rest to the ingest
        reader_kwargs = csv_shape_kwargs(kwargs)
        kwargs = {
            key: value for key, value in kwargs.items() if key not in reader_kwargs
        }
        record_batches = reader.read_csv(
            path, **read_csv_schema_kwargs(reader, schema), **reader_kwargs
        ).to_pyarrow_batches()
        return self.read_record_batches(record_batches, table_name, mode, **kwargs)
