from __future__ import annotations

import inspect
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import ibis
import ibis.expr.operations as ops
import ibis.expr.types as ir
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from ibis.backends.base import BaseBackend, CanCreateSchema
from ibis.backends.base.sqlglot import STAR
from sqlglot import exp, transforms
from sqlglot.dialects import Postgres
from sqlglot.dialects.dialect import rename_func

from letsql.compiler.core import translate

from ibis.expr.operations.udf import InputType
from ibis.formats.pyarrow import PyArrowType
import ibis.expr.datatypes as dt
from ibis.util import normalize_filename, gen_name

if TYPE_CHECKING:
    from collections.abc import Mapping

from letsql.internal import SessionContext, SessionConfig, udf

import ibis.expr.schema as sch

import sqlglot as sg

_exclude_exp = (exp.Pow, exp.ArrayContains)


class LetSql(Postgres):
    class Generator(Postgres.Generator):
        TRANSFORMS = {
            exp: trans
            for exp, trans in Postgres.Generator.TRANSFORMS.items()
            if exp not in _exclude_exp
        } | {
            exp.Select: transforms.preprocess(
                [
                    transforms.eliminate_qualify,
                ]
            ),
            exp.IsNan: rename_func("isnan"),
            exp.BitwiseXor: lambda self, e: self.binary(e, "#"),
        }


class Backend(BaseBackend, CanCreateSchema):
    name = "letsql"
    dialect = LetSql
    builder = None
    supports_in_memory_tables = True
    supports_arrays = True
    supports_python_udfs = False

    @property
    def version(self):
        return "0.1.2"

    def do_connect(
        self,
    ) -> None:
        df_config = SessionConfig(
            {"datafusion.sql_parser.dialect": "PostgreSQL"}
        ).with_information_schema(True)
        self.con = SessionContext(df_config)
        self._register_builtin_udfs()

    def _register_builtin_udfs(self):
        from letsql.compiler import udfs

        for name, func in inspect.getmembers(
            udfs,
            predicate=lambda m: callable(m)
            and not m.__name__.startswith("_")
            and m.__module__ == udfs.__name__,
        ):
            annotations = typing.get_type_hints(func)
            argnames = list(inspect.signature(func).parameters.keys())
            input_types = [
                PyArrowType.from_ibis(dt.dtype(annotations.get(arg_name)))
                for arg_name in argnames
            ]
            return_type = PyArrowType.from_ibis(dt.dtype(annotations["return"]))
            udf_fun = udf(
                func,
                input_types=input_types,
                return_type=return_type,
                volatility="immutable",
                name=name,
            )
            self.con.register_udf(udf_fun)

    def _register_udfs(self, expr: ir.Expr) -> None:
        for udf_node in expr.op().find(ops.ScalarUDF):
            if udf_node.__input_type__ == InputType.PYARROW:
                udf = self._compile_pyarrow_udf(udf_node)
                self.con.register_udf(udf)

        for udf_node in expr.op().find(ops.ElementWiseVectorizedUDF):
            udf = self._compile_elementwise_udf(udf_node)
            self.con.register_udf(udf)

    def _compile_pyarrow_udf(self, udf_node):
        return udf(
            udf_node.__func__,
            input_types=[PyArrowType.from_ibis(arg.dtype) for arg in udf_node.args],
            return_type=PyArrowType.from_ibis(udf_node.dtype),
            volatility=getattr(udf_node, "config", {}).get("volatility", "volatile"),
            name=udf_node.__full_name__,
        )

    def _compile_elementwise_udf(self, udf_node):
        return udf(
            udf_node.func,
            input_types=list(map(PyArrowType.from_ibis, udf_node.input_type)),
            return_type=PyArrowType.from_ibis(udf_node.return_type),
            volatility="volatile",
            name=udf_node.func.__name__,
        )

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        """List the available tables."""
        return self._filter_with_like(self.con.tables(), like)

    def table(self, name: str, schema: sch.Schema | None = None) -> ir.Table:
        """Get an ibis expression representing a LetSQL table.

        Parameters
        ----------
        name
            The name of the table to retrieve
        schema
            An optional schema for the table

        Returns
        -------
        Table
            A table expression
        """
        catalog = self.con.catalog()
        database = catalog.database()
        table = database.table(name)
        schema = sch.schema(table.schema)
        return ops.DatabaseTable(name, schema, self).to_expr()

    def _memtable_from_dataframe(
        self,
        data: pd.DataFrame | Any,
        schema=None,
        columns: Iterable[str] | None = None,
    ):
        df = pd.DataFrame(data, columns=columns)
        if df.columns.inferred_type != "string":
            cols = df.columns
            newcols = getattr(
                schema,
                "names",
                (f"{i}" for i in range(len(cols))),
            )
            df = df.rename(columns=dict(zip(cols, newcols)))
        return df

    def memtable(
        self,
        data: pa.Table | pa.RecordBatch | pa.Dataset | pd.DataFrame | Any,
        table_name: str | None = None,
        columns: Iterable[str] | None = None,
        schema=None,
    ):
        if table_name is None:
            table_name = gen_name("pandas_memtable")

        if isinstance(data, pa.Table):
            self.con.deregister_table(table_name)
            self.con.register_record_batches(table_name, [data.to_batches()])
            return self.table(table_name)
        elif isinstance(data, pa.RecordBatch):
            self.con.deregister_table(table_name)
            self.con.register_record_batches(table_name, [[data]])
            return self.table(table_name)
        elif isinstance(data, pa.dataset.Dataset):
            self.con.deregister_table(table_name)
            self.con.register_dataset(table_name, data)
            return self.table(table_name)
        else:
            try:
                df = self._memtable_from_dataframe(data, columns=columns, schema=schema)
                table = pa.Table.from_pandas(df)
                self.con.deregister_table(table_name)
                self.con.register_record_batches(table_name, [table.to_batches()])
                return self.table(table_name)
            except Exception:
                raise ValueError("Unknown source")

    def register(
        self,
        source: str | Path | pa.Table | pa.RecordBatch | pa.Dataset | pd.DataFrame,
        table_name: str | None = None,
        **kwargs: Any,
    ):
        """Register a data set with `table_name` located at `source`.

        Parameters
        ----------
        source
            The data source(s). Maybe a path to a file or directory of
            parquet/csv files, a pandas dataframe, or a pyarrow table, dataset
            or record batch.
        table_name
            The name of the table
        """

        if isinstance(source, (str, Path)):
            first = str(source)
        elif isinstance(source, pa.Table):
            self.con.deregister_table(table_name)
            self.con.register_record_batches(table_name, [source.to_batches()])
            return self.table(table_name)
        elif isinstance(source, pa.RecordBatch):
            self.con.deregister_table(table_name)
            self.con.register_record_batches(table_name, [[source]])
            return self.table(table_name)
        elif isinstance(source, pa.dataset.Dataset):
            self.con.deregister_table(table_name)
            self.con.register_dataset(table_name, source)
            return self.table(table_name)
        elif isinstance(source, pd.DataFrame):
            return self.register(pa.Table.from_pandas(source), table_name, **kwargs)
        else:
            raise ValueError("`source` must be either a string or a pathlib.Path")

        if first.startswith(("parquet://", "parq://")) or first.endswith(
            ("parq", "parquet")
        ):
            self.con.deregister_table(table_name)
            self.con.register_parquet(
                table_name, first, file_extension=".parquet", skip_metadata=False
            )
            return self.table(table_name)
        elif first.startswith(("csv://", "txt://")) or first.endswith(
            ("csv", "tsv", "txt")
        ):
            self.con.deregister_table(table_name)
            self.con.register_csv(table_name, first, **kwargs)
            return self.table(table_name)
        elif first.endswith("csv.gz"):
            self.con.deregister_table(table_name)
            self.con.register_csv(
                table_name, first, file_extension="gz", file_compression_type="gzip"
            )
            return self.table(table_name)
        else:
            self._register_failure()
            return None

    def _register_failure(self):
        import inspect

        msg = ", ".join(
            m[0] for m in inspect.getmembers(self) if m[0].startswith("read_")
        )
        raise ValueError(
            f"Cannot infer appropriate read function for input, "
            f"please call one of {msg} directly"
        )

    def _define_udf_translation_rules(self, expr):
        if self.supports_python_udfs:
            raise NotImplementedError(self.name)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        name = op.name
        schema = op.schema

        self.con.deregister_table(name)
        if batches := op.data.to_pyarrow(schema).to_batches():
            self.con.register_record_batches(name, [batches])
        else:
            empty_dataset = ds.dataset([], schema=schema.to_pyarrow())
            self.con.register_dataset(name=name, dataset=empty_dataset)

    def _register_in_memory_tables(self, expr: ir.Expr) -> None:
        if self.supports_in_memory_tables:
            for memtable in expr.op().find(ops.InMemoryTable):
                self._register_in_memory_table(memtable)

    def create_table(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def create_view(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def drop_table(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def drop_view(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    @property
    def current_schema(self) -> str:
        return NotImplementedError()

    def list_schemas(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        return self._filter_with_like(
            self.con.catalog(
                database if database is not None else "datafusion"
            ).names(),
            like=like,
        )

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        # not actually a table, but this is how sqlglot represents schema names
        schema_name = sg.table(name, db=database)
        self.raw_sql(sg.exp.Create(kind="SCHEMA", this=schema_name, exists=force))

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        schema_name = sg.table(name, db=database)
        self.raw_sql(sg.exp.Drop(kind="SCHEMA", this=schema_name, exists=force))

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        pa = self._import_pyarrow()

        self._register_in_memory_tables(expr)

        frame = self.con.sql(self.compile(expr.as_table(), params, **kwargs))
        batches = frame.collect()
        return pa.ipc.RecordBatchReader.from_batches(
            frame.schema() if not batches else batches[0].schema, batches
        )

    def _to_sqlglot(
        self, expr: ir.Expr, limit: str | None = None, params=None, **_: Any
    ):
        """Compile an Ibis expression to a sqlglot object."""
        table_expr = expr.as_table()

        if limit == "default":
            limit = ibis.options.sql.default_limit
        if limit is not None:
            table_expr = table_expr.limit(limit)

        if params is None:
            params = {}

        sql = translate(table_expr.op(), params=params)
        assert not isinstance(sql, sg.exp.Subquery)

        if isinstance(sql, sg.exp.Table):
            sql = sg.select(STAR).from_(sql)

        assert not isinstance(sql, sg.exp.Subquery)
        return sql

    def compile(
        self, expr: ir.Expr, limit: str | None = None, params=None, **kwargs: Any
    ):
        """Compile an Ibis expression to a LetSQL SQL string."""
        return self._to_sqlglot(expr, limit=limit, params=params, **kwargs).sql(
            dialect=self.dialect, pretty=True
        )

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: int | str | None = "default",
        **kwargs: Any,
    ):
        output = self.to_pyarrow(expr.as_table(), params=params, limit=limit, **kwargs)
        return expr.__pandas_result__(output.to_pandas(timestamp_as_object=True))

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        """Execute expression and return results in as a pyarrow table.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            Ibis expression to export to pyarrow
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        kwargs
            Keyword arguments

        Returns
        -------
        Table
            A pyarrow table holding the results of the executed expression.
        """
        pa = self._import_pyarrow()
        self._run_pre_execute_hooks(expr)

        table_expr = expr.as_table()
        schema = table_expr.schema()
        arrow_schema = schema.to_pyarrow()
        with self.to_pyarrow_batches(
            table_expr, params=params, limit=limit, **kwargs
        ) as reader:
            table = pa.Table.from_batches(reader, schema=reader.schema)
            # arrow_schema = reader.schema

        return expr.__pyarrow_result__(
            table.rename_columns(table_expr.columns).cast(arrow_schema)
        )

    def read_csv(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a CSV file as a table in the current database.

        Parameters
        ----------
        path
            The data source. A string or Path to the CSV file.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to Datafusion loading function.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        path = normalize_filename(path)
        table_name = table_name or gen_name("read_csv")
        # Our other backends support overwriting views / tables when reregistering
        self.con.deregister_table(table_name)
        self.con.register_csv(table_name, path, **kwargs)
        return self.table(table_name)

    def read_parquet(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a parquet file as a table in the current database.

        Parameters
        ----------
        path
            The data source.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to Datafusion loading function.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        path = normalize_filename(path)
        table_name = table_name or gen_name("read_parquet")
        # Our other backends support overwriting views / tables when re-registering
        self.con.deregister_table(table_name)
        self.con.register_parquet(table_name, path, **kwargs)
        return self.table(table_name)
