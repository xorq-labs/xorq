from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ibis.expr.datatypes as dt
import sqlglot as sg
import sqlglot.expressions as sge
from ibis import Schema, util
from ibis.backends.sql.compilers import SQLiteCompiler as IbisSQLiteCompiler
from ibis.backends.sqlite import Backend as IbisSQLiteBackend
from ibis.backends.sqlite import _init_sqlite3, _quote
from ibis.expr import types as ir
from ibis.util import gen_name

from xorq.backends import ExecutionBackend
from xorq.expr.api import read_csv, read_parquet


if TYPE_CHECKING:
    import pyarrow as pa


class SQLiteCompiler(IbisSQLiteCompiler):
    def visit_Hash(self, op, *, arg):
        return self.f.anon.city_hash_32(arg)


compiler = SQLiteCompiler()


class Backend(ExecutionBackend, IbisSQLiteBackend):
    compiler = compiler

    def do_connect(
        self,
        database: str | Path | None = None,
        type_map: dict[str, str | dt.DataType] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an Ibis client connected to a SQLite database.

        Multiple database files can be accessed using the `attach()` method.

        Parameters
        ----------
        database
            File path to the SQLite database file. If `None`, creates an
            in-memory transient database and you can use attach() to add more
            files
        type_map
            An optional mapping from a string name of a SQLite "type" to the
            corresponding Ibis DataType that it represents. This can be used
            to override schema inference for a given SQLite database.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.sqlite.connect()
        >>> t = con.create_table("my_table", schema=ibis.schema(dict(x="int64")))
        >>> con.insert("my_table", obj=[(1,), (2,), (3,)])
        >>> t
        DatabaseTable: my_table
          x int64
        >>> t.head(1).execute()
           x
        0  1
        """
        _init_sqlite3()

        self.uri = ":memory:" if database is None else database
        self.con = sqlite3.connect(self.uri, **kwargs)

        self._post_connect(type_map)

    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader,
        table_name: str | None = None,
        mode: str = "create",
        overwrite: bool = True,
        **kwargs: Any,
    ) -> ir.Table:
        from xorq.common.utils.sqlite_utils import SQLiteADBC

        table_name = table_name or gen_name("read_record_batches")

        catalog = "temp" if self.is_in_memory() else None

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

        if self.is_in_memory():
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

    def _into_memory_record_batches(self, record_batches, table_name):
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

    def is_in_memory(self):
        return "memory" in self.uri

    def read_parquet(
        self,
        path: str | Path,
        table_name: str | None = None,
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        table_name = table_name or gen_name("xo_read_parquet")
        record_batches = read_parquet(path).to_pyarrow_batches()
        return self.read_record_batches(record_batches, table_name, mode, **kwargs)

    def read_csv(
        self,
        path: str | Path,
        table_name: str | None = None,
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        table_name = table_name or gen_name("xo_read_csv")
        record_batches = read_csv(path).to_pyarrow_batches()
        return self.read_record_batches(record_batches, table_name, mode, **kwargs)
