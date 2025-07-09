"""xorq expression API definitions."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import pyarrow as pa
import pyarrow.dataset as ds
from opentelemetry import trace

import xorq.vendor.ibis.expr.types as ir
from xorq.common.utils.caching_utils import find_backend
from xorq.common.utils.defer_utils import (  # noqa: F403
    deferred_read_csv,
    deferred_read_parquet,
    rbr_wrapper,
)
from xorq.common.utils.io_utils import extract_suffix
from xorq.common.utils.otel_utils import tracer
from xorq.common.utils.rbr_utils import otel_instrument_reader
from xorq.expr.ml import (
    calc_split_column,
    train_test_splits,
)
from xorq.expr.relations import (
    CachedNode,
    register_and_transform_remote_tables,
)
from xorq.vendor.ibis.backends.sql.dialects import DataFusion
from xorq.vendor.ibis.expr import api
from xorq.vendor.ibis.expr.api import *  # noqa: F403
from xorq.vendor.ibis.expr.sql import SQLString
from xorq.vendor.ibis.expr.types import Table


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from io import TextIOWrapper
    from pathlib import Path

    import pandas as pd
    import pyarrow as pa

    from xorq.vendor.ibis.expr.schema import SchemaLike


__all__ = (
    "execute",
    "calc_split_column",
    "read_csv",
    "read_parquet",
    "read_postgres",
    "register",
    "train_test_splits",
    "to_parquet",
    "to_csv",
    "to_json",
    "to_pyarrow",
    "to_pyarrow_batches",
    "to_sql",
    "get_plans",
    "deferred_read_csv",
    "deferred_read_parquet",
    "get_object_metadata",
    *api.__all__,
)


def memtable(
    data,
    *,
    columns: Iterable[str] | None = None,
    schema: SchemaLike | None = None,
    name: str | None = None,
) -> Table:
    """Construct an ibis table expression from in-memory data.

    Parameters
    ----------
    data
        A table-like object (`pandas.DataFrame`, `pyarrow.Table`, or
        `polars.DataFrame`), or any data accepted by the `pandas.DataFrame`
        constructor (e.g. a list of dicts).

        Note that ibis objects (e.g. `MapValue`) may not be passed in as part
        of `data` and will result in an error.

        Do not depend on the underlying storage type (e.g., pyarrow.Table),
        it's subject to change across non-major releases.
    columns
        Optional [](`typing.Iterable`) of [](`str`) column names. If provided,
        must match the number of columns in `data`.
    schema
        Optional [`Schema`](./schemas.qmd#ibis.expr.schema.Schema).
        The functions use `data` to infer a schema if not passed.
    name
        Optional name of the table.

    Returns
    -------
    Table
        A table expression backed by in-memory data.

    Examples
    --------
    >>> import xorq as xo
    >>> xo.options.interactive = False
    >>> t = xo.memtable([{"a": 1}, {"a": 2}])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             a
          0  1
          1  2

    >>> t = xo.memtable([{"a": 1, "b": "foo"}, {"a": 2, "b": "baz"}])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             a    b
          0  1  foo
          1  2  baz

    Create a table literal without column names embedded in the data and pass
    `columns`

    >>> t = xo.memtable([(1, "foo"), (2, "baz")], columns=["a", "b"])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             a    b
          0  1  foo
          1  2  baz

    Create a table literal without column names embedded in the data. Ibis
    generates column names if none are provided.

    >>> t = xo.memtable([(1, "foo"), (2, "baz")])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             col0 col1
          0     1  foo
          1     2  baz

    """

    if isinstance(data, ds.InMemoryDataset):
        data = data.to_table()

    if isinstance(data, pa.RecordBatch):
        data = data.to_pandas()

    return api.memtable(data, columns=columns, schema=schema, name=name)


def read_csv(
    sources: str | Path | Sequence[str | Path],
    table_name: str | None = None,
    **kwargs: Any,
) -> ir.Table:
    """Lazily load a CSV or set of CSVs.

    This function delegates to the `read_csv` method on the current default
    backend (DuckDB or `xorq.config.default_backend`).

    Parameters
    ----------
    sources
        A filesystem path or URL or list of same.  Supports CSV and TSV files.
    table_name
        A name to refer to the table.  If not provided, a name will be generated.
    kwargs
        Backend-specific keyword arguments for the file type. For the DuckDB
        backend used by default, please refer to:

        * CSV/TSV: https://duckdb.org/docs/data/csv/overview.html#parameters.

    Returns
    -------
    ir.Table
        Table expression representing a file

    Examples
    --------
    >>> import xorq
    >>> xorq.options.interactive = True
    >>> lines = '''a,b
    ... 1,d
    ... 2,
    ... ,f
    ... '''
    >>> with open("/tmp/lines.csv", mode="w") as f:
    ...     nbytes = f.write(lines)  # nbytes is unused
    >>> t = xorq.read_csv("/tmp/lines.csv")
    >>> t
    ┏━━━━━━━┳━━━━━━━━┓
    ┃ a     ┃ b      ┃
    ┡━━━━━━━╇━━━━━━━━┩
    │ int64 │ string │
    ├───────┼────────┤
    │     1 │ d      │
    │     2 │ NULL   │
    │  NULL │ f      │
    └───────┴────────┘

    """
    from xorq.config import _backend_init

    con = _backend_init()
    return con.read_csv(sources, table_name=table_name, **kwargs)


def read_parquet(
    sources: str | Path | Sequence[str | Path],
    table_name: str | None = None,
    **kwargs: Any,
) -> ir.Table:
    """Lazily load a parquet file or set of parquet files.

    This function delegates to the `read_parquet` method on the current default
    backend (DuckDB or `ibis.config.default_backend`).

    Parameters
    ----------
    sources
        A filesystem path or URL or list of same.
    table_name
        A name to refer to the table.  If not provided, a name will be generated.
    kwargs
        Backend-specific keyword arguments for the file type. For the DuckDB
        backend used by default, please refer to:

        * Parquet: https://duckdb.org/docs/data/parquet

    Returns
    -------
    ir.Table
        Table expression representing a file

    Examples
    --------
    >>> import xorq
    >>> import pandas as pd
    >>> xorq.options.interactive = True
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": list("ghi")})
    >>> df
       a  b
    0  1  g
    1  2  h
    2  3  i
    >>> df.to_parquet("/tmp/data.parquet")
    >>> t = xorq.read_parquet("/tmp/data.parquet")
    >>> t
    ┏━━━━━━━┳━━━━━━━━┓
    ┃ a     ┃ b      ┃
    ┡━━━━━━━╇━━━━━━━━┩
    │ int64 │ string │
    ├───────┼────────┤
    │     1 │ g      │
    │     2 │ h      │
    │     3 │ i      │
    └───────┴────────┘

    """
    from xorq.config import _backend_init

    con = _backend_init()
    return con.read_parquet(sources, table_name=table_name, **kwargs)


def register(
    source: str | Path | pa.Table | pa.RecordBatch | pa.Dataset | pd.DataFrame,
    table_name: str | None = None,
    **kwargs: Any,
):
    from xorq.config import _backend_init

    con = _backend_init()
    return con.register(source, table_name=table_name, **kwargs)


def read_postgres(
    uri: str,
    table_name: str | None = None,
    **kwargs: Any,
):
    from xorq.config import _backend_init

    con = _backend_init()
    return con.read_postgres(uri, table_name=table_name, **kwargs)


@functools.cache
def _cached_with_op(op, pretty):
    from xorq.config import _backend_init

    con = _backend_init()

    expr = op.to_expr()
    sg_expr = con.compiler.to_sqlglot(expr)
    sql = sg_expr.sql(dialect=DataFusion, pretty=pretty)
    return sql


def to_sql(expr: ir.Expr, pretty: bool = True) -> SQLString:
    """Return the formatted SQL string for an expression.

    Parameters
    ----------
    expr
        Ibis expression.
    pretty
        Whether to use pretty formatting.

    Returns
    -------
    str
        Formatted SQL string

    """

    return SQLString(_cached_with_op(expr.unbind().op(), pretty))


@tracer.start_as_current_span("_register_and_transform_cache_tables")
def _register_and_transform_cache_tables(expr):
    """This function will sequentially execute any cache node that is not already cached"""

    def fn(node, kwargs):
        if kwargs:
            node = node.__recreate__(kwargs)
        if isinstance(node, CachedNode):
            uncached, storage = node.parent, node.storage
            node = storage.set_default(uncached, uncached.op())
        return node

    op = expr.op()
    out = op.replace(fn)

    return out.to_expr()


@tracer.start_as_current_span("_transform_deferred_reads")
def _transform_deferred_reads(expr):
    dt_to_read = {}

    span = trace.get_current_span()

    def replace_read(node, _kwargs):
        from xorq.expr.relations import Read

        if isinstance(node, Read):
            read_kwargs = dict(node.read_kwargs)
            span.add_event(
                "replace_read",
                {
                    "engine": node.source.name,
                    "method_name": node.method_name,
                    "path": read_kwargs.get("path") or read_kwargs.get("source"),
                },
            )
            if node.source.name == "pandas":
                # FIXME: pandas read is not lazy, leave it to the pandas executor to do
                node = dt_to_read[node] = node.make_dt()
            else:
                node = dt_to_read[node] = node.make_dt()
        else:
            if _kwargs:
                node = node.__recreate__(_kwargs)
        return node

    expr = expr.op().replace(replace_read).to_expr()
    return expr, dt_to_read


@tracer.start_as_current_span("execute")
def execute(expr: ir.Expr, **kwargs: Any):
    """Execute an expression against its backend if one exists.

    Parameters
    ----------
    kwargs
        Keyword arguments

    Examples
    --------
    >>> import xorq as xo
    >>> t = xo.examples.penguins.fetch()
    >>> t.execute()
           species     island  bill_length_mm  ...  body_mass_g     sex  year
    0       Adelie  Torgersen            39.1  ...       3750.0    male  2007
    1       Adelie  Torgersen            39.5  ...       3800.0  female  2007
    2       Adelie  Torgersen            40.3  ...       3250.0  female  2007
    3       Adelie  Torgersen             NaN  ...          NaN    None  2007
    4       Adelie  Torgersen            36.7  ...       3450.0  female  2007
    ..         ...        ...             ...  ...          ...     ...   ...
    339  Chinstrap      Dream            55.8  ...       4000.0    male  2009
    340  Chinstrap      Dream            43.5  ...       3400.0  female  2009
    341  Chinstrap      Dream            49.6  ...       3775.0    male  2009
    342  Chinstrap      Dream            50.8  ...       4100.0    male  2009
    343  Chinstrap      Dream            50.2  ...       3775.0  female  2009
    [344 rows x 8 columns]

    Scalar parameters can be supplied dynamically during execution.
    >>> species = xo.param("string")
    >>> expr = t.filter(t.species == species).order_by(t.bill_length_mm)
    >>> expr.execute(limit=3, params={species: "Gentoo"})
      species  island  bill_length_mm  ...  body_mass_g     sex  year
    0  Gentoo  Biscoe            40.9  ...         4650  female  2007
    1  Gentoo  Biscoe            41.7  ...         4700  female  2009
    2  Gentoo  Biscoe            42.0  ...         4150  female  2007
    <BLANKLINE>
    [3 rows x 8 columns]
    """
    batch_reader = to_pyarrow_batches(expr, **kwargs)
    with tracer.start_as_current_span("read_pandas"):
        df = batch_reader.read_pandas(timestamp_as_object=True)
    return expr.__pandas_result__(df)


@tracer.start_as_current_span("_transform_expr")
def _transform_expr(expr):
    expr = _register_and_transform_cache_tables(expr)
    expr, created = register_and_transform_remote_tables(expr)
    expr, dt_to_read = _transform_deferred_reads(expr)
    return (expr, created)


@tracer.start_as_current_span("to_pyarrow_batches")
def to_pyarrow_batches(
    expr: ir.Expr,
    *,
    chunk_size: int = 1_000_000,
    **kwargs: Any,
):
    """Execute expression and return a RecordBatchReader.

    This method is eager and will execute the associated expression
    immediately.

    Parameters
    ----------
    chunk_size
        Maximum number of rows in each returned record batch.
    kwargs
        Keyword arguments

    Returns
    -------
    results
        RecordBatchReader
    """
    from xorq.expr.relations import FlightExpr, FlightUDXF

    span = trace.get_current_span()

    if isinstance(expr.op(), (FlightExpr, FlightUDXF)):
        # TODO: verify correct caching behavior
        span.set_attribute("engine", "flight")
        return expr.op().to_rbr()
    (expr, created) = _transform_expr(expr)
    con, _ = find_backend(expr.op(), use_default=True)

    span.set_attribute("engine", con.name)
    reader = con.to_pyarrow_batches(expr, chunk_size=chunk_size, **kwargs)

    def clean_up():
        for table_name, conn in created.items():
            try:
                conn.drop_table(table_name, force=True)
            except Exception:
                conn.drop_view(table_name)

    return otel_instrument_reader(rbr_wrapper(reader, clean_up))


def to_pyarrow(expr: ir.Expr, **kwargs: Any):
    """Execute expression and return results in as a pyarrow table.

    This method is eager and will execute the associated expression
    immediately.

    Parameters
    ----------
    kwargs
        Keyword arguments

    Returns
    -------
    Table
        A pyarrow table holding the results of the executed expression.
    """
    batch_reader = to_pyarrow_batches(expr, **kwargs)
    arrow_table = batch_reader.read_all()
    return expr.__pyarrow_result__(arrow_table)


def to_parquet(
    expr: ir.Expr,
    path: str | Path,
    params: Mapping[ir.Scalar, Any] | None = None,
    **kwargs: Any,
):
    """Write the results of executing the given expression to a parquet file.

    This method is eager and will execute the associated expression
    immediately.

    See https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html for details.

    Parameters
    ----------
    path
        A string or Path where the Parquet file will be written.
    params
        Mapping of scalar parameter expressions to value.
    **kwargs
        Additional keyword arguments passed to pyarrow.parquet.ParquetWriter

    Examples
    --------
    Write out an expression to a single parquet file.

    >>> import ibis
    >>> import tempfile
    >>> penguins = ibis.examples.penguins.fetch()
    >>> penguins.to_parquet(tempfile.mktemp())
    """
    import pyarrow  # noqa: ICN001, F401
    import pyarrow.parquet as pq
    import pyarrow_hotfix  # noqa: F401

    with to_pyarrow_batches(expr, params=params) as batch_reader:
        with pq.ParquetWriter(path, batch_reader.schema, **kwargs) as writer:
            for batch in batch_reader:
                writer.write_batch(batch)


def to_csv(
    expr: ir.Expr,
    path: str | Path,
    params: Mapping[ir.Scalar, Any] | None = None,
    **kwargs: Any,
):
    """Write the results of executing the given expression to a CSV file.

    This method is eager and will execute the associated expression
    immediately.

    Parameters
    ----------
    path
        The data source. A string or Path to the CSV file.
    params
        Mapping of scalar parameter expressions to value.
    **kwargs
        Additional keyword arguments passed to pyarrow.csv.CSVWriter

    https://arrow.apache.org/docs/python/generated/pyarrow.csv.CSVWriter.htmlditional keyword arguments passed to pyarrow.csv.CSVWriter
    """

    import pyarrow  # noqa: ICN001, F401
    import pyarrow.csv as pcsv
    import pyarrow_hotfix  # noqa: F401

    with pcsv.CSVWriter(path, schema=expr.schema().to_pyarrow(), **kwargs) as writer:
        with to_pyarrow_batches(expr, params=params) as batch_reader:
            for batch in batch_reader:
                writer.write_batch(batch)


def to_json(
    expr: ir.Expr,
    path: str | Path | TextIOWrapper,
    params: Mapping[ir.Scalar, Any] | None = None,
):
    """Write the results of `expr` to a NDJSON file.

    This method is eager and will execute the associated expression
    immediately.

    Parameters
    ----------
    path
        The data source. A string or Path to the Delta Lake table.
    **kwargs
        Additional, backend-specific keyword arguments.

    https://github.com/ndjson/ndjson-spec
    """
    import pyarrow  # noqa: ICN001, F401
    import pyarrow_hotfix  # noqa: F401

    from xorq.common.utils.io_utils import maybe_open

    with maybe_open(path, "w") as f:
        with to_pyarrow_batches(expr, params=params) as batch_reader:
            for batch in batch_reader:
                df = batch.to_pandas()
                batch_json = df.to_json(orient="records", lines=True)
                f.write(batch_json)


def get_plans(expr):
    _expr, _ = _transform_expr(expr)
    con, _ = find_backend(_expr.op())
    sql = f"EXPLAIN {to_sql(_expr)}"
    return con.con.sql(sql).to_pandas().set_index("plan_type")["plan"].to_dict()


def get_object_metadata(path: str, **kwargs: Any) -> dict:
    from xorq.config import _backend_init

    con = _backend_init()

    suffix = extract_suffix(path).lstrip(".")

    if "storage_options" in kwargs:
        kwargs["storage_options"] = dict(kwargs.pop("storage_options"))

    return con.con.get_object_metadata(path, suffix, **kwargs)
