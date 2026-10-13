from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa

import xorq.common.exceptions as com
import xorq.vendor.ibis.config
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.schema as sch
import xorq.vendor.ibis.expr.types as ir
from xorq.common.utils.rbr_utils import coerce_to_arrow_table
from xorq.vendor import ibis
from xorq.vendor.ibis import util
from xorq.vendor.ibis.backends import BaseBackend, NoUrl
from xorq.vendor.ibis.common.dispatch import lazy_singledispatch
from xorq.vendor.ibis.formats.pandas import PandasData, PandasSchema
from xorq.vendor.ibis.formats.pyarrow import PyArrowData


def _csv_parse_dtype(typ):
    """The dtype pandas parses a csv column of this ibis type as.

    Integers and booleans take pandas' nullable dtypes so missing values
    read; floats and strings their usual ones. Every other type is read as
    text for ``PandasData.convert_table``: pandas refuses temporal dtypes in
    ``dtype=``, and inferring would turn a date like 20240101 or a decimal
    into a number.
    """
    if typ.is_integer():
        # ibis Int8..UInt64 are also pandas' nullable dtype names
        return pd.api.types.pandas_dtype(type(typ).__name__)
    if typ.is_boolean():
        return pd.BooleanDtype()
    if typ.is_floating() or typ.is_string():
        return typ.to_pandas()
    return str


def _csv_interval(column, typ):
    """Bare numbers are counted in the schema's unit; text intervals ('1 days',
    '00:00:01') parse as such, row by row. Integers are converted apart from
    fractions, so a fraction elsewhere in the column cannot push them through
    float64. PandasData casts intervals through numpy, which parses neither."""
    unit = typ.unit.short
    is_integer = column.str.fullmatch(r"\s*[+-]?[0-9]+\s*", na=False)
    integers = pd.to_numeric(
        column.where(is_integer), errors="coerce", dtype_backend="numpy_nullable"
    )
    if pd.api.types.is_unsigned_integer_dtype(integers):
        raise OverflowError(
            f"interval column {column.name!r} holds {integers.max()}, past int64"
        )
    fractions = pd.to_numeric(column.where(~is_integer), errors="coerce")
    text = pd.to_timedelta(column.where(~is_integer & fractions.isna()))
    return text.fillna(pd.to_timedelta(integers, unit=unit)).fillna(
        pd.to_timedelta(fractions, unit=unit)
    )


if TYPE_CHECKING:
    import pathlib
    from collections.abc import Mapping, MutableMapping

    from batchcorder import StreamCache


__all__ = [
    "Backend",
]


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
        >>> import xorq.api as xo
        >>> xo.pandas.connect({"t": pd.DataFrame({"a": [1, 2, 3]})})  # doctest: +ELLIPSIS
        <ibis.backends.pandas.Backend object at 0x...>
        """
        self.dictionary = dictionary or {}
        self.schemas = {}

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
        self,
        source: str | pathlib.Path,
        table_name: str | None = None,
        schema: sch.Schema | None = None,
        **kwargs: Any,
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
        schema
            The types to read the columns as, instead of inferring them.
            Integers, booleans, floats and strings are parsed by pandas;
            every other type is read as text and converted afterwards.
            Not combinable with ``dtype``, ``converters``, ``parse_dates``
            or ``engine="pyarrow"``; nested types are not parsed, read
            them as string.
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
        if schema is not None:
            if conflicting := {"dtype", "converters", "parse_dates"} & kwargs.keys():
                raise TypeError(
                    f"`schema` cannot be combined with {sorted(conflicting)}"
                )
            if kwargs.get("engine") == "pyarrow":
                raise TypeError(
                    "engine='pyarrow' applies dtypes after inference; `schema` needs the default engine"
                )
            schema = sch.schema(schema)
            if nested := [name for name, typ in schema.items() if typ.is_nested()]:
                raise com.XorqTypeError(
                    f"pandas read_csv does not parse nested columns {nested}; "
                    "read them as string"
                )
            kwargs["dtype"] = {
                name: _csv_parse_dtype(typ) for name, typ in schema.items()
            }
        df = pd.read_csv(source, **kwargs)
        if schema is not None:
            if unknown := set(df.columns) - set(schema.names):
                raise ValueError(f"columns missing from schema: {sorted(unknown)}")
            # usecols may read a subset of the columns the schema types
            schema = sch.Schema({name: schema[name] for name in df.columns})
            for name, typ in schema.items():
                if typ.is_interval():
                    df[name] = _csv_interval(df[name], typ)
            df = PandasData.convert_table(df, schema)
        # only now, so a read that failed above leaves an existing table as
        # it was; a reread infers afresh unless a schema is given
        self.dictionary[table_name] = df
        self.schemas.pop(table_name, None)
        if schema is not None:
            self.schemas[table_name] = schema
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
        from xorq.backends.pandas.kernels import supported_operations  # noqa: PLC0415

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
    # deferred_read_* rejects a list/tuple of paths up front: the readers
    # take a single path
    reads_single_path = True

    def execute(self, query, params=None, limit="default", **kwargs):
        from xorq.backends.pandas.executor import PandasExecutor  # noqa: PLC0415

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

    def drop_table(self, name: str, *, force: bool = False) -> None:
        if not force and name in self.dictionary:
            raise com.XorqError(
                "Cannot drop existing table. Call drop_table with force=True to drop existing table."
            )
        del self.dictionary[name]
        del self.schemas[name]

    def read_record_batches(
        self,
        record_batches: pa.Table
        | pa.RecordBatchReader
        | StreamCache
        | list[pa.RecordBatch]
        | tuple[pa.RecordBatch],
        table_name: str | None = None,
    ) -> ir.Table:
        self.dictionary[table_name] = coerce_to_arrow_table(record_batches).to_pandas()
        return self.table(table_name)


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
