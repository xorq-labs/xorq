from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
import sqlglot as sg
import sqlglot.expressions as sge

import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.schema as sch
import xorq.vendor.ibis.expr.types as ir
from xorq.vendor import ibis
from xorq.vendor.ibis.backends.datafusion import Backend as IbisDatafusionBackend
from xorq.vendor.ibis.common.dispatch import lazy_singledispatch
from xorq.vendor.ibis.util import gen_name


if TYPE_CHECKING:
    import pandas as pd


class Backend(IbisDatafusionBackend):
    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        self.con.from_arrow(op.data.to_pyarrow(op.schema), op.name)

    def create_table(
        self,
        name: str,
        obj: ir.Table | pa.Table | pa.RecordBatchReader | pa.RecordBatch | None = None,
        *,
        schema: sch.SchemaLike | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ):
        """Create a table in DataFusion.

        Parameters
        ----------
        name
            Name of the table to create
        obj
            The data with which to populate the table; optional, but at least
            one of `obj` or `schema` must be specified
        schema
            The schema of the table to create; optional, but at least one of
            `obj` or `schema` must be specified
        database
            The name of the database in which to create the table; if not
            passed, the current database is used.
        temp
            Create a temporary table
        overwrite
            If `True`, replace the table if it already exists, otherwise fail
            if the table exists

        """
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        quoted = self.compiler.quoted

        if isinstance(obj, ir.Expr):
            table = obj

            # If it's a memtable, it will get registered in the pre-execute hooks
            self._run_pre_execute_hooks(table)

            compiler = self.compiler
            relname = "_"
            query = sg.select(
                *(
                    compiler.cast(
                        sg.column(col, table=relname, quoted=quoted), dtype
                    ).as_(col, quoted=quoted)
                    for col, dtype in table.schema().items()
                )
            ).from_(
                compiler.to_sqlglot(table).subquery(
                    sg.to_identifier(relname, quoted=quoted)
                )
            )
        elif obj is not None:
            table_ident = sg.table(name, db=database, quoted=quoted).sql(self.dialect)
            _read_in_memory(obj, table_ident, self, overwrite=overwrite)
            return self.table(name, database=database)
        else:
            query = None

        table_ident = sg.table(name, db=database, quoted=quoted)

        if query is None:
            target = sge.Schema(
                this=table_ident,
                expressions=(schema or table.schema()).to_sqlglot(self.dialect),
            )
        else:
            target = table_ident

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
            expression=query,
            replace=overwrite,
        )

        with self._safe_raw_sql(create_stmt):
            pass

        return self.table(name, database=database)


@contextlib.contextmanager
def _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
    """Workaround inability to overwrite tables in dataframe API.

    DataFusion has helper methods for loading in-memory data, but these methods
    don't allow overwriting tables.
    The SQL interface allows creating tables from existing tables, so we register
    the data as a table using the dataframe API, then run a

    CREATE [OR REPLACE] TABLE table_name AS SELECT * FROM in_memory_thing

    and that allows us to toggle the overwrite flag.
    """
    src = sge.Create(
        this=table_name,
        kind="TABLE",
        expression=sg.select("*").from_(tmp_name),
        replace=overwrite,
    )

    yield

    _conn.raw_sql(src)
    _conn.drop_table(tmp_name)


@lazy_singledispatch
def _read_in_memory(
    source: Any, table_name: str, _conn: Backend, overwrite: bool = False
):
    raise NotImplementedError("No support for source or imports missing")


@_read_in_memory.register(dict)
def _pydict(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("pydict")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.from_pydict(source, name=tmp_name)


@_read_in_memory.register("polars.DataFrame")
def _polars(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("polars")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.from_polars(source, name=tmp_name)


@_read_in_memory.register("polars.LazyFrame")
def _polars(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("polars")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.from_polars(source.collect(), name=tmp_name)


@_read_in_memory.register("pyarrow.Table")
def _pyarrow_table(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("pyarrow")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.from_arrow(source, name=tmp_name)


@_read_in_memory.register("pyarrow.RecordBatchReader")
def _pyarrow_rbr(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("pyarrow")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.from_arrow(source.read_all(), name=tmp_name)


@_read_in_memory.register("pyarrow.RecordBatch")
def _pyarrow_rb(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("pyarrow")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.register_record_batches(tmp_name, [[source]])


@_read_in_memory.register("pyarrow.dataset.Dataset")
def _pyarrow_rb(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("pyarrow")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.register_dataset(tmp_name, source)


@_read_in_memory.register("pandas.DataFrame")
def _pandas(source: pd.DataFrame, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("pandas")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.from_pandas(source, name=tmp_name)
