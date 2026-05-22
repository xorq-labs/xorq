from __future__ import annotations

import contextlib
import functools
import inspect
import itertools
import typing
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn


if TYPE_CHECKING:
    import pandas as pd
    import pyarrow.dataset as ds

import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
import sqlglot as sg
import sqlglot.expressions as sge
from sqlglot import exp, parse_one

import xorq
import xorq.common.exceptions as com
import xorq.expr.datatypes as dt
import xorq.internal as df
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.schema as sch
import xorq.vendor.ibis.expr.types as ir
from xorq.backends.xorq_datafusion.compiler import compiler
from xorq.backends.xorq_datafusion.provider import IbisTableProvider
from xorq.common.utils import classproperty
from xorq.common.utils.aws_utils import make_s3_connection
from xorq.expr import Expr
from xorq.expr.pyaggregator import PyAggregator, make_struct_type
from xorq.expr.udf import ExprScalarUDF
from xorq.internal import (
    DataFrame,
    SessionConfig,
    SessionContext,
    Table,
    WindowEvaluator,
    WindowUDF,
    udwf,
)
from xorq.vendor import ibis
from xorq.vendor.ibis.backends import (
    CanCreateCatalog,
    CanCreateDatabase,
    CanCreateSchema,
    NoUrl,
)
from xorq.vendor.ibis.backends.sql import SQLBackend
from xorq.vendor.ibis.backends.sql.compilers.base import C
from xorq.vendor.ibis.expr.operations.udf import InputType
from xorq.vendor.ibis.formats.pyarrow import PyArrowType
from xorq.vendor.ibis.util import gen_name, normalize_filename, normalize_filenames


def _select_and_cast(batch, schema):
    missing = set(schema.names) - set(batch.schema.names)
    if missing:
        raise ValueError(f"batch missing columns required by schema: {sorted(missing)}")
    return batch.select(schema.names).cast(schema)


def _compile_pyarrow_udwf(udwf_node):
    def make_datafusion_udwf(
        input_types,
        return_type,
        name,
        evaluate=None,
        evaluate_all=None,
        evaluate_all_with_rank=None,
        supports_bounded_execution=False,
        uses_window_frame=False,
        include_rank=False,
        volatility="immutable",
        **kwargs,
    ):
        def return_value(value):
            def f(_):
                return value

            return f

        kwds = {
            "evaluate": evaluate,
            "evaluate_all": evaluate_all,
            "evaluate_all_with_rank": evaluate_all_with_rank,
            "supports_bounded_execution": return_value(supports_bounded_execution),
            "uses_window_frame": return_value(uses_window_frame),
            "include_rank": return_value(include_rank),
            **kwargs,
        }
        mytyp = type(
            name,
            (WindowEvaluator,),
            kwds,
        )
        my_udwf = udwf(
            mytyp(),
            input_types,
            return_type,
            volatility=str(volatility),
            # datafusion normalizes to lower case and ibis doesn't quote
            name=name.lower(),
        )
        return my_udwf

    config = udwf_node.__config__ | {
        "input_types": tuple(
            dtype.to_pyarrow() for dtype in udwf_node.__config__["input_types"]
        ),
        "return_type": udwf_node.__config__["return_type"].to_pyarrow(),
    }

    my_udwf = make_datafusion_udwf(**config)
    return my_udwf


def _compile_pyarrow_udaf(udaf_node):
    func = udaf_node.__func__
    name = udaf_node.__func_name__
    return_type = PyArrowType.from_ibis(udaf_node.dtype)
    parameters = [
        (param_name, PyArrowType.from_ibis(param.annotation.pattern.dtype))
        for param_name, param in udaf_node.__signature__.parameters.items()
        if param_name != "where"
    ]
    if not parameters:
        raise ValueError(
            f"UDAF '{name}' has no non-'where' parameters; at least one input parameter is required"
        )
    names, input_types = map(list, zip(*parameters))  # noqa
    struct_type = make_struct_type(names, input_types)

    class MyAggregator(PyAggregator):
        @classproperty
        def struct_type(cls):
            return struct_type

        def py_evaluate(self):
            struct_array = self.pystate()
            args = (struct_array.field(field_name) for field_name in self.names)
            return func(*args)

        @classproperty
        def return_type(cls):
            return return_type

        @classproperty
        def name(cls):
            return name

    return df.udaf(
        accum=MyAggregator,
        input_type=input_types,
        return_type=return_type,
        state_type=[MyAggregator.state_type],
        volatility=MyAggregator.volatility,
        name=name,
    )


class Backend(SQLBackend, CanCreateCatalog, CanCreateDatabase, CanCreateSchema, NoUrl):
    name = "xorq_datafusion"
    supports_in_memory_tables = True
    supports_arrays = True
    compiler = compiler

    @staticmethod
    def _translate_sort(exprs: list[ir.Expr]):
        result = []
        for expr in exprs:
            if not isinstance(node := expr.op(), ops.SortKey):
                raise ValueError(f"Expected SortKey, got {type(node)}")

            column_identifier = str(sg.to_identifier(node.name, quoted=True))
            result.append(
                Expr.column(column_identifier).sort(node.ascending, node.nulls_first)
            )

        return result

    @property
    def version(self):
        return xorq.__version__

    def do_connect(self, config: SessionConfig | None = None) -> None:
        """Creates a connection.

        Parameters
        ----------
        config
            DataFusion `SessionConfig` for session-level options; defaults to sane xorq settings if `None`.

        Examples
        --------
        >>> import xorq.api as xo
        >>> con = xo.connect()

        """
        if config is None:
            config = SessionConfig()
        config = config.with_information_schema(True).set(
            "datafusion.sql_parser.dialect", "PostgreSQL"
        )

        self.con = SessionContext(config=config)

        self._register_builtin_udfs()

    def disconnect(self) -> None:
        pass

    @contextlib.contextmanager
    def _safe_raw_sql(self, sql: sge.Statement) -> Any:
        yield self.raw_sql(sql).collect()

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        name = gen_name("datafusion_metadata_view")
        table = sg.table(name, quoted=self.compiler.quoted)
        src = sge.Create(
            this=table,
            kind="VIEW",
            expression=sg.parse_one(query, read="datafusion"),
        )

        with self._safe_raw_sql(src):
            pass

        try:
            result = (
                self.raw_sql(f"DESCRIBE {table.sql(self.dialect)}")
                .to_arrow_table()
                .to_pydict()
            )
        finally:
            self.drop_view(name)
        return sch.Schema(
            {
                name: self.compiler.type_mapper.from_string(
                    type_string, nullable=is_nullable == "YES"
                )
                for name, type_string, is_nullable in zip(
                    result["column_name"], result["data_type"], result["is_nullable"]
                )
            }
        )

    def _register_builtin_udfs(self):
        from xorq.backends.xorq_datafusion import udfs  # noqa: PLC0415

        for name, func in inspect.getmembers(
            udfs,
            predicate=lambda m: (
                callable(m)
                and not m.__name__.startswith("_")
                and m.__module__ == udfs.__name__
            ),
        ):
            annotations = typing.get_type_hints(func)
            argnames = list(inspect.signature(func).parameters.keys())
            input_types = [
                PyArrowType.from_ibis(dt.dtype(annotations.get(arg_name)))
                for arg_name in argnames
            ]
            return_type = PyArrowType.from_ibis(dt.dtype(annotations["return"]))
            udf = df.udf(
                func,
                input_types=input_types,
                return_type=return_type,
                volatility="immutable",
                name=name,
            )
            self.con.register_udf(udf)

    def _register_udfs(self, expr: ir.Expr) -> None:
        for udf_node in expr.op().find(ops.ScalarUDF):
            if udf_node.__input_type__ == InputType.PYARROW:
                if isinstance(udf_node, ExprScalarUDF):
                    udf = self._compile_pyarrow_expr_udf(udf_node)
                else:
                    udf = self._compile_pyarrow_udf(udf_node)
                self.con.register_udf(udf)

        for udf_node in expr.op().find(ops.ElementWiseVectorizedUDF):
            udf = self._compile_elementwise_udf(udf_node)
            self.con.register_udf(udf)

        for agg_node in expr.op().find(ops.AggUDF):
            if agg_node.__input_type__ == InputType.PYARROW:
                if {"evaluate", "evaluate_all"}.intersection(agg_node.__config__):
                    compiled_udwf = _compile_pyarrow_udwf(agg_node)
                    self.con.register_udwf(compiled_udwf)
                else:
                    udaf = _compile_pyarrow_udaf(agg_node)
                    self.con.register_udaf(udaf)

    def _compile_pyarrow_expr_udf(self, udf_node):
        import pandas as pd  # noqa: PLC0415

        value = udf_node.computed_kwargs_expr.execute()
        if isinstance(value, pd.DataFrame):
            if value.shape != (1, 1):
                raise ValueError(
                    f"Expected scalar (1,1) DataFrame, got shape {value.shape}"
                )
            ((value,),) = value.values
        computed_arg = udf_node.post_process_fn(value)
        return df.udf(
            functools.partial(udf_node.__func__, computed_arg=computed_arg),
            input_types=[PyArrowType.from_ibis(arg.dtype) for arg in udf_node.args],
            return_type=PyArrowType.from_ibis(udf_node.dtype),
            volatility=getattr(udf_node, "__config__", {}).get(
                "volatility", "volatile"
            ),
            name=udf_node.__func_name__,
        )

    def _compile_pyarrow_udf(self, udf_node):
        return df.udf(
            udf_node.__func__,
            input_types=[PyArrowType.from_ibis(arg.dtype) for arg in udf_node.args],
            return_type=PyArrowType.from_ibis(udf_node.dtype),
            volatility=getattr(udf_node, "__config__", {}).get(
                "volatility", "volatile"
            ),
            name=udf_node.__func_name__,
        )

    def _compile_elementwise_udf(self, udf_node):
        return df.udf(
            udf_node.func,
            input_types=list(map(PyArrowType.from_ibis, udf_node.input_type)),
            return_type=PyArrowType.from_ibis(udf_node.return_type),
            volatility="volatile",
            name=udf_node.__func_name__,
        )

    def raw_sql(self, query: str | sge.Expression) -> Any:
        """Execute a SQL string `query` against the database."""
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect, pretty=True)
        self._log(query)
        return self.con.sql(query)

    @property
    def current_catalog(self) -> str:
        raise NotImplementedError("DataFusion backend does not support current_catalog")

    @property
    def current_database(self) -> str:
        raise NotImplementedError(
            "DataFusion backend does not support current_database"
        )

    def list_catalogs(self, like: str | None = None) -> list[str]:
        code = (
            sg.select(C.table_catalog)
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
        ).sql()
        result = self.con.sql(code).to_pydict()
        return self._filter_with_like(result["table_catalog"], like)

    def create_catalog(self, name: str, force: bool = False) -> None:
        with self._safe_raw_sql(
            sge.Create(kind="DATABASE", this=sg.to_identifier(name), exists=force)
        ):
            pass

    def drop_catalog(self, name: str, force: bool = False) -> None:
        raise com.UnsupportedOperationError(
            "DataFusion does not support dropping databases"
        )

    def list_databases(
        self, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        return self._filter_with_like(
            self.con.catalog(catalog if catalog is not None else "datafusion").names(),
            like=like,
        )

    def create_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        db_name = sg.table(name, db=catalog)
        with self._safe_raw_sql(sge.Create(kind="SCHEMA", this=db_name, exists=force)):
            pass

    def drop_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        db_name = sg.table(name, db=catalog)
        with self._safe_raw_sql(sge.Drop(kind="SCHEMA", this=db_name, exists=force)):
            pass

    def list_tables(
        self,
        like: str | None = None,
        database: str | None = None,
    ) -> list[str]:
        """Return the list of table names in the current database.

        Parameters
        ----------
        like
            A pattern in Python's regex format.
        database
            Unused in the datafusion backend.

        Returns
        -------
        list[str]
            The list of the table names that match the pattern `like`.
        """
        return self._filter_with_like(self.con.tables(), like)

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        if catalog is not None:
            catalog = self.con.catalog(catalog)
        else:
            catalog = self.con.catalog()

        if database is not None:
            database = catalog.database(database)
        else:
            database = catalog.database()

        table = database.table(table_name)
        return sch.schema(table.schema)

    def register(
        self,
        source: str
        | Path
        | pa.Table
        | pa.RecordBatch
        | pa.RecordBatchReader
        | ds.Dataset
        | pd.DataFrame
        | ir.Expr
        | Table
        | DataFrame,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a data set with `table_name` located at `source`.

        Parameters
        ----------
        source
            The data source(s). Maybe a path to a file or directory of
            parquet/csv files, a pandas dataframe, or a pyarrow table, dataset
            or record batch.
        table_name
            The name of the table
        kwargs
            Datafusion-specific keyword arguments

        """
        import pandas as pd  # noqa: PLC0415
        import pyarrow.dataset as ds  # noqa: PLC0415

        # Phase 1: resolve ir.Expr to a concrete type before dispatch.
        if isinstance(source, ir.Expr):
            source = self._resolve_expr_for_register(source, **kwargs)

        table_name = table_name or gen_name("register")
        table_ident = str(sg.to_identifier(table_name, quoted=self.compiler.quoted))

        # Phase 2: path sources delegate entirely to read helpers.
        if isinstance(source, (str, Path)):
            return self._register_path(str(source), table_name=table_name, **kwargs)

        # Phase 3: normalize pandas DataFrame to Arrow Table before dispatch.
        if isinstance(source, pd.DataFrame):
            source = pa.Table.from_pandas(source)
            source = source.drop(
                [col for col in source.column_names if col.startswith("__index_level_")]
            )

        # Phase 4: dispatch to the DataFusion registration API.
        self.con.deregister_table(table_ident)
        match source:
            case pa.Table():
                self.con.register_record_batches(table_ident, [source.to_batches()])
            case pa.RecordBatch():
                self.con.register_record_batches(table_ident, [[source]])
            case pa.RecordBatchReader():
                if "ordering" in kwargs:
                    kwargs["sort_order"] = self._translate_sort(kwargs.pop("ordering"))
                self.con.register_record_batch_reader(table_ident, source, **kwargs)
            case ds.Dataset():
                self.con.register_dataset(table_ident, source)
            case ir.Table():
                # Cross-backend expr: IbisTableProvider executes via source's own backend.
                self.con.register_table_provider(table_ident, IbisTableProvider(source))
            case ir.Expr():
                # Cross-backend non-table expr: materialize via source's own backend.
                self.con.register_record_batch_reader(
                    table_ident, source.to_pyarrow_batches()
                )
            case Table():
                self.con.register_table(table_ident, source)
            case DataFrame():
                self.con.register_dataframe(table_ident, source)
            case _:
                raise ValueError(f"Unknown `source` type {type(source)}")

        return self.table(table_name)

    def _resolve_expr_for_register(self, source: ir.Expr, **kwargs) -> Any:
        backends, has_unbound = source._find_backends()
        if has_unbound:
            raise ValueError(
                "Cannot register an expression with unbound tables; "
                "bind all tables to a backend before registering"
            )
        if len(backends) > 1:
            raise ValueError("Multiple backends found for this expression")
        if not backends:
            # Pure in-memory expr (e.g. memtable): materialize now.
            return self.execute(source)
        backend = backends[0]
        if not isinstance(backend, Backend):
            # Cross-backend expr: leave as ir.Expr; match arms handle it.
            return source
        # Same-backend DataFusion expr: compile to native DataFrame to avoid
        # nested tokio runtime panic that IbisTableProvider.scan() would cause.
        backend._register_udfs(source)
        backend._register_in_memory_tables(source)
        raw_sql = backend.compile(source.as_table(), **kwargs)
        return backend.con.sql(raw_sql)

    def _register_path(self, path: str, table_name: str, **kwargs) -> ir.Table:
        if path.startswith(("parquet://", "parq://")) or path.endswith(
            ("parq", "parquet")
        ):
            return self.read_parquet(path, table_name=table_name, **kwargs)
        if path.startswith(("csv://", "txt://")) or path.endswith(
            ("csv", "tsv", "txt")
        ):
            return self.read_csv(path, table_name=table_name, **kwargs)
        self._register_failure()

    def register_table_provider(
        self,
        source: ir.Table,
        table_name: str | None = None,
    ):
        table_name = table_name or gen_name("register_table_provider")
        table_ident = str(sg.to_identifier(table_name, quoted=self.compiler.quoted))
        self.con.deregister_table(table_ident)
        self.con.register_table_provider(table_ident, IbisTableProvider(source))
        return self.table(table_name)

    def _register_failure(self) -> NoReturn:
        msg = ", ".join(
            m[0] for m in inspect.getmembers(self) if m[0].startswith("read_")
        )
        raise ValueError(
            f"Cannot infer appropriate read function for input, "
            f"please call one of {msg} directly"
        )

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        name = op.name
        schema = op.schema

        self.con.deregister_table(name)
        if batches := op.data.to_pyarrow(schema).to_batches():
            self.con.register_record_batches(name, [batches])
        else:
            import pyarrow.dataset as ds  # noqa: PLC0415

            empty_dataset = ds.dataset([], schema=schema.to_pyarrow())
            self.con.register_dataset(name=name, dataset=empty_dataset)

    def _register_in_memory_tables(self, expr: ir.Expr) -> None:
        if self.supports_in_memory_tables:
            for memtable in expr.op().find(ops.InMemoryTable):
                self._register_in_memory_table(memtable)

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
        path = normalize_filenames(path)
        table_name = table_name or gen_name("read_csv")
        kwargs.setdefault("file_extension", Path(path[0]).suffix)

        storage_options, is_connection_set = make_s3_connection()
        if is_connection_set:
            kwargs["storage_options"] = storage_options | kwargs.get(
                "storage_options", {}
            )

        if schema := kwargs.get("schema"):
            if isinstance(schema, ibis.Schema):
                kwargs["schema"] = schema.to_pyarrow()

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
        path = normalize_filenames(path)
        table_name = table_name or gen_name("read_parquet")
        kwargs.setdefault("file_extension", Path(path[0]).suffix)

        self.con.deregister_table(table_name)
        self.con.register_parquet(table_name, path, **kwargs)
        return self.table(table_name)

    def read_delta(
        self, source_table: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a Delta Lake table as a table in the current database.

        Parameters
        ----------
        source_table
            The data source. Must be a directory
            containing a Delta Lake table.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to deltalake.DeltaTable.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        source_table = normalize_filename(source_table)

        table_name = table_name or gen_name("read_delta")

        self.con.deregister_table(table_name)

        try:
            from deltalake import DeltaTable  # noqa: PLC0415
        except ImportError as err:
            raise ImportError(
                "The deltalake extra is required to use the "
                "read_delta method. You can install it using pip:\n\n"
                "pip install 'ibis-framework[deltalake]'\n"
            ) from err

        delta_table = DeltaTable(source_table, **kwargs)

        return self.register(delta_table.to_pyarrow_dataset(), table_name=table_name)

    def read_record_batches(
        self,
        source: pa.Table | pa.RecordBatchReader | Iterable[pa.RecordBatch],
        table_name: str | None = None,
    ) -> ir.Table:
        """Register Arrow data as a table in the current database.

        Each batch is cast to the declared schema before being handed to
        DataFusion. This prevents silent data corruption when physical Arrow
        types differ from the declared schema (e.g. ``large_utf8`` batches
        with a ``utf8`` schema), which would otherwise cause DataFusion to
        misread 64-bit offsets as 32-bit across the C Data Interface boundary.

        Parameters
        ----------
        source
            The Arrow data to register. Accepts:

            - ``pa.Table`` — converted to batches via ``to_batches()``.
            - ``pa.RecordBatchReader`` — consumed directly; schema taken from
              the reader.
            - Any ``Iterable[pa.RecordBatch]`` (list, tuple, generator) —
              schema inferred from the first batch.
        table_name
            Name for the registered table. Defaults to a sequentially
            generated name.

        Returns
        -------
        ir.Table
            The just-registered table.

        Raises
        ------
        ValueError
            If ``source`` is an iterable or ``pa.Table`` that yields no batches,
            or if a batch is missing columns required by the declared schema.
            Cast errors from type incompatibilities are deferred: they surface
            during ``.execute()``, not at this call site, because batches are
            cast lazily as DataFusion consumes the reader.

        Examples
        --------
        From a ``pa.Table``:

        >>> import pyarrow as pa
        >>> import xorq.api as xo
        >>> t = xo.connect().read_record_batches(pa.table({"a": [1, 2, 3]}))

        From a list of ``pa.RecordBatch``:

        >>> batches = [pa.record_batch({"a": [1, 2]}), pa.record_batch({"a": [3]})]
        >>> t = xo.connect().read_record_batches(batches)

        """
        table_name = table_name or gen_name("read_record_batches")
        table_ident = str(sg.to_identifier(table_name, quoted=self.compiler.quoted))
        self.con.deregister_table(table_ident)
        schema: pa.Schema
        batches: Iterable[pa.RecordBatch]
        match source:
            case pa.Table():
                if source.num_rows == 0:
                    raise ValueError("source contains no record batches")
                schema = source.schema
                batches = source.to_batches()
            case pa.RecordBatchReader():
                schema = source.schema
                batches = source
            case _:
                it = iter(source)
                try:
                    first = next(it)
                except StopIteration:
                    raise ValueError("source contains no record batches") from None
                schema = first.schema
                batches = itertools.chain([first], it)
        self.con.register_record_batch_reader(
            table_ident,
            pa.RecordBatchReader.from_batches(
                schema, (_select_and_cast(batch, schema) for batch in batches)
            ),
        )
        return self.table(table_name)

    def execute(self, expr: ir.Expr, **kwargs: Any):
        batch_reader = self.to_pyarrow_batches(expr, **kwargs)
        return expr.__pandas_result__(
            batch_reader.read_pandas(timestamp_as_object=True)
        )

    def to_pyarrow(self, expr: ir.Expr, **kwargs: Any) -> pa.Table:
        batch_reader = self.to_pyarrow_batches(expr, **kwargs)
        arrow_table = batch_reader.read_all()
        return expr.__pyarrow_result__(arrow_table)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        return self._to_pyarrow_batches(expr, chunk_size=chunk_size, **kwargs)

    def _to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ):
        self._register_udfs(expr)
        self._register_in_memory_tables(expr)
        table_expr = expr.as_table()
        raw_sql = self.compile(table_expr, **kwargs)
        frame = self.con.sql(raw_sql)
        schema = table_expr.schema()
        pyarrow_schema = schema.to_pyarrow()
        struct_schema = schema.as_struct().to_pyarrow()

        def make_gen():
            yield from (
                pa.RecordBatch.from_struct_array(
                    pa.RecordBatch.from_arrays(
                        batch.to_pyarrow().columns, schema=pyarrow_schema
                    )
                    .to_struct_array()
                    .cast(struct_schema)
                )
                for batch in frame.execute_stream()
            )

        return pa.ipc.RecordBatchReader.from_batches(
            pyarrow_schema,
            make_gen(),
        )

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ):
        """Create a table in Datafusion.

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

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        quoted = self.compiler.quoted

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj, schema=schema)
            else:
                table = obj

            self._run_pre_execute_hooks(table)
            compiler = self.compiler

            relname = "_"
            query = sg.select(
                *(
                    compiler.cast(
                        sg.column(col, table=relname, quoted=quoted), dtype
                    ).as_(col, quoted=quoted)
                    if not isinstance(dtype, dt.LargeString)
                    else compiler.f.arrow_cast(
                        sg.column(col, table=relname, quoted=quoted), "LargeUtf8"
                    ).as_(col, quoted=quoted)
                    for col, dtype in table.schema().items()
                )
            ).from_(
                compiler.to_sqlglot(table).subquery(
                    sg.to_identifier(relname, quoted=quoted)
                )
            )
        else:
            query = None

        table_ident = sg.to_identifier(name, quoted=quoted)
        if query is None:
            column_defs = [
                sge.ColumnDef(
                    this=sg.to_identifier(colname, quoted=quoted),
                    kind=self.compiler.type_mapper.from_ibis(typ),
                    constraints=(
                        None
                        if typ.nullable
                        else [sge.ColumnConstraint(kind=sge.NotNullColumnConstraint())]
                    ),
                )
                for colname, typ in (schema or table.schema()).items()
            ]
            target = sge.Schema(this=table_ident, expressions=column_defs)
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

    def to_parquet(
        self,
        expr: ir.Table,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        import pyarrow.parquet as pq  # noqa: PLC0415

        with expr.to_pyarrow_batches(params=params) as batch_reader:
            with pq.ParquetWriter(path, batch_reader.schema, **kwargs) as writer:
                for batch in batch_reader:
                    writer.write_batch(batch)

    def _extract_catalog(self, query):
        tables = parse_one(query).find_all(exp.Table)
        return {table.name: self.table(table.name) for table in tables}

    def register_udwf(self, func: WindowUDF):
        self.con.register_udwf(func)


def connect(config: SessionConfig | None = None):
    con = Backend()
    con.do_connect(config)
    return con
