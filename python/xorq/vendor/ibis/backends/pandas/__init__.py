from __future__ import annotations

import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
import pyarrow_hotfix  # noqa: F401

import xorq.common.exceptions as com
import xorq.vendor.ibis.config
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.schema as sch
import xorq.vendor.ibis.expr.types as ir
from xorq.vendor import ibis
from xorq.vendor.ibis import util
from xorq.vendor.ibis.backends import BaseBackend, NoUrl
from xorq.vendor.ibis.common.dispatch import lazy_singledispatch
from xorq.vendor.ibis.formats.pandas import PandasData, PandasSchema
from xorq.vendor.ibis.formats.pyarrow import PyArrowData


if TYPE_CHECKING:
    import pathlib
    from collections.abc import Mapping, MutableMapping


class BasePandasBackend(BaseBackend, NoUrl):
    """Base class for backends based on pandas."""

    name = "pandas"
    dialect = None
    backend_table_type = pd.DataFrame

    class Options(xorq.vendor.ibis.config.Config):
        enable_trace: bool = False

    def do_connect(
        self,
        dictionary: MutableMapping[str, pd.DataFrame] | None = None,
    ) -> None:
        """Construct a client from a dictionary of pandas DataFrames.

        Parameters
        ----------
        dictionary
            An optional mapping of string table names to pandas DataFrames.

        Examples
        --------
        >>> import ibis
        >>> ibis.pandas.connect({"t": pd.DataFrame({"a": [1, 2, 3]})})  # doctest: +ELLIPSIS
        <ibis.backends.pandas.Backend object at 0x...>
        """
        warnings.warn(
            f"The {self.name} backend is slated for removal in 10.0.",
            DeprecationWarning,
        )
        self.dictionary = dictionary or {}
        self.schemas: MutableMapping[str, sch.Schema] = {}

    def disconnect(self) -> None:
        pass

    def from_dataframe(
        self,
        df: pd.DataFrame,
        name: str = "df",
        client: BasePandasBackend | None = None,
    ) -> ir.Table:
        """Construct an ibis table from a pandas DataFrame.

        Parameters
        ----------
        df
            A pandas DataFrame
        name
            The name of the pandas DataFrame
        client
            Client dictionary will be mutated with the name of the DataFrame,
            if not provided a new client is created

        Returns
        -------
        Table
            A table expression

        """
        if client is None:
            return self.connect({name: df}).table(name)
        client.dictionary[name] = df
        return client.table(name)

    def read_csv(
        self, source: str | pathlib.Path, table_name: str | None = None, **kwargs: Any
    ):
        """Register a CSV file as a table in the current session.

        Parameters
        ----------
        source
            The data source. Can be a local or remote file, pathlike objects
            also accepted.
        table_name
            An optional name to use for the created table. This defaults to
            a generated name.
        **kwargs
            Additional keyword arguments passed to Pandas loading function.
            See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
            for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        table_name = table_name or util.gen_name("read_csv")
        df = pd.read_csv(source, **kwargs)
        self.dictionary[table_name] = df
        return self.table(table_name)

    def read_parquet(
        self, source: str | pathlib.Path, table_name: str | None = None, **kwargs: Any
    ):
        """Register a parquet file as a table in the current session.

        Parameters
        ----------
        source
            The data source(s). May be a path to a file, an iterable of files,
            or directory of parquet files.
        table_name
            An optional name to use for the created table. This defaults to
            a generated name.
        **kwargs
            Additional keyword arguments passed to Pandas loading function.
            See https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
            for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        table_name = table_name or util.gen_name("read_parquet")
        df = pd.read_parquet(source, **kwargs)
        self.dictionary[table_name] = df
        return self.table(table_name)

    @property
    def version(self) -> str:
        return pd.__version__

    def list_tables(self, like=None, database=None):
        """Return the list of table names in the current database.

        Parameters
        ----------
        like
            A pattern in Python's regex format.
        database
            Unused in the pandas backend.

        Returns
        -------
        list[str]
            The list of the table names that match the pattern `like`.
        """
        return self._filter_with_like(list(self.dictionary.keys()), like)

    def table(self, name: str, schema: sch.Schema | None = None):
        inferred_schema = self.get_schema(name)
        overridden_schema = {**inferred_schema, **(schema or {})}
        return ops.DatabaseTable(name, overridden_schema, self).to_expr()

    def get_schema(self, table_name, *, database=None):
        try:
            schema = self.schemas[table_name]
        except KeyError:
            df = self.dictionary[table_name]
            self.schemas[table_name] = schema = PandasData.infer_table(df)

        return schema

    def compile(self, expr, *args, **kwargs):
        return expr

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a table."""
        if temp:
            com.XorqError(
                "Passing `temp=True` to the Pandas backend create_table method has no "
                "effect: all tables are in memory and temporary."
            )
        if database:
            com.XorqError(
                "Passing `database` to the Pandas backend create_table method has no "
                "effect: Pandas cannot set a database."
            )
        if obj is None and schema is None:
            raise com.XorqError("The schema or obj parameter is required")
        if schema is not None:
            schema = ibis.schema(schema)

        if obj is not None:
            df = self._convert_object(obj)
        else:
            dtypes = dict(PandasSchema.from_ibis(schema))
            df = pd.DataFrame(columns=dtypes.keys()).astype(dtypes)

        if name in self.dictionary and not overwrite:
            raise com.XorqError(f"Cannot overwrite existing table `{name}`")

        self.dictionary[name] = df

        if schema is not None:
            self.schemas[name] = schema
        return self.table(name)

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        return self.create_table(
            name, obj=obj, temp=None, database=database, overwrite=overwrite
        )

    def drop_view(self, name: str, *, force: bool = False) -> None:
        self.drop_table(name, force=force)

    def drop_table(self, name: str, *, force: bool = False) -> None:
        try:
            del self.dictionary[name]
        except KeyError:
            if not force:
                raise com.XorqError(f"Table {name} does not exist") from None

    def _convert_object(self, obj: Any) -> Any:
        return _convert_object(obj, self)

    @classmethod
    @lru_cache
    def _get_operations(cls):
        from xorq.vendor.ibis.backends.pandas.kernels import supported_operations

        return supported_operations

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        return operation in cls._get_operations()

    def _drop_cached_table(self, name):
        del self.dictionary[name]

    def to_pyarrow(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        table_expr = expr.as_table()
        output = pa.Table.from_pandas(
            self.execute(table_expr, params=params, limit=limit, **kwargs)
        )

        # cudf.pandas adds a column with the name `__index_level_0__` (and maybe
        # other index level columns) but these aren't part of the known schema
        # so we drop them
        output = output.drop(
            filter(lambda col: col.startswith("__index_level_"), output.column_names)
        )
        table = PyArrowData.convert_table(output, table_expr.schema())
        return expr.__pyarrow_result__(table)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1000000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        pa = self._import_pyarrow()
        pa_table = self.to_pyarrow(
            expr.as_table(), params=params, limit=limit, **kwargs
        )
        return pa.ipc.RecordBatchReader.from_batches(
            pa_table.schema, pa_table.to_batches(max_chunksize=chunk_size)
        )


class Backend(BasePandasBackend):
    name = "pandas"

    def execute(self, query, params=None, limit="default", **kwargs):
        from xorq.vendor.ibis.backends.pandas.executor import PandasExecutor

        if limit != "default" and limit is not None:
            raise ValueError(
                "limit parameter to execute is not yet implemented in the "
                "pandas backend"
            )

        if not isinstance(query, ir.Expr):
            raise TypeError(
                f"`query` has type {type(query).__name__!r}, expected ibis.expr.types.Expr"
            )

        params = params or {}
        params = {k.op() if isinstance(k, ir.Expr) else k: v for k, v in params.items()}

        return PandasExecutor.execute(query.op(), backend=self, params=params)

    def _create_cached_table(self, name, expr):
        return self.create_table(name, expr.execute())

    def _finalize_memtable(self, name: str) -> None:
        """No-op, let Python handle clean up."""


@lazy_singledispatch
def _convert_object(obj: Any, _conn):
    raise com.BackendConversionError(
        f"Unable to convert {obj.__class__} object "
        f"to backend type: {_conn.__class__.backend_table_type}"
    )


@_convert_object.register("ibis.expr.types.Table")
def _table(obj, _conn):
    if isinstance(op := obj.op(), ops.InMemoryTable):
        return op.data.to_frame()
    else:
        raise com.BackendConversionError(
            f"Unable to convert {obj.__class__} object "
            f"to backend type: {_conn.__class__.backend_table_type}"
        )


@_convert_object.register("polars.DataFrame")
@_convert_object.register("pyarrow.Table")
def _pa_polars(obj, _conn):
    return obj.to_pandas()


@_convert_object.register("polars.LazyFrame")
def _polars_lazy(obj, _conn):
    return obj.collect().to_pandas()


@_convert_object.register("pandas.DataFrame")
def _pandas(obj, _conn):
    return obj
