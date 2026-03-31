"""xorq expression API definitions."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import pyarrow as pa
import toolz
from opentelemetry import trace

import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.types as ir
from xorq.backends.xorq import Backend
from xorq.common.exceptions import XorqError
from xorq.common.utils.caching_utils import find_backend
from xorq.common.utils.defer_utils import (  # noqa: F403
    deferred_read_csv,
    deferred_read_parquet,
    rbr_wrapper,
)
from xorq.common.utils.graph_utils import replace_nodes, walk_nodes
from xorq.common.utils.io_utils import (
    extract_suffix,
    maybe_open,
)
from xorq.common.utils.otel_utils import tracer
from xorq.common.utils.rbr_utils import otel_instrument_reader
from xorq.expr.ml import (
    calc_split_column,
    train_test_splits,
)
from xorq.expr.operations import NamedScalarParameter
from xorq.expr.relations import (
    CachedNode,
    HashingTag,
    Tag,
    register_and_transform_remote_tables,
)
from xorq.vendor.ibis.expr import api
from xorq.vendor.ibis.expr.api import *  # noqa: F403
from xorq.vendor.ibis.expr.sql import SQLString


if TYPE_CHECKING:
    from collections.abc import Sequence
    from io import TextIOWrapper
    from pathlib import Path

    import pandas as pd
    import pyarrow as pa


__all__ = (
    "execute",
    "calc_split_column",
    "read_csv",
    "read_parquet",
    "read_postgres",
    "read_pyarrow_stream",
    "register",
    "train_test_splits",
    "to_parquet",
    "to_csv",
    "to_json",
    "to_pyarrow",
    "to_pyarrow_batches",
    "to_pyarrow_stream",
    "to_sql",
    "get_plans",
    "deferred_read_csv",
    "deferred_read_parquet",
    "get_object_metadata",
    "bind_params",
    *api.__all__,
)


def read_pyarrow_stream(
    source,
    con=None,
    table_name=None,
    **kwargs,
) -> ir.Table:
    from xorq.config import _backend_init  # noqa: PLC0415

    con = con or _backend_init()
    rbr = pa.ipc.open_stream(source, **kwargs)
    return con.read_record_batches(rbr, table_name=table_name)


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
    >>> import xorq.api as xo
    >>> xo.options.interactive = True
    >>> lines = '''a,b
    ... 1,d
    ... 2,
    ... ,f
    ... '''
    >>> with open("/tmp/lines.csv", mode="w") as f:
    ...     nbytes = f.write(lines)  # nbytes is unused
    >>> t = xo.read_csv("/tmp/lines.csv")
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
    from xorq.config import _backend_init  # noqa: PLC0415

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
    >>> import xorq.api as xo
    >>> import pandas as pd
    >>> xo.options.interactive = True
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": list("ghi")})
    >>> df
       a  b
    0  1  g
    1  2  h
    2  3  i
    >>> df.to_parquet("/tmp/data.parquet")
    >>> t = xo.read_parquet("/tmp/data.parquet")
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
    from xorq.config import _backend_init  # noqa: PLC0415

    con = _backend_init()
    return con.read_parquet(sources, table_name=table_name, **kwargs)


def register(
    source: str | Path | pa.Table | pa.RecordBatch | pa.Dataset | pd.DataFrame,
    table_name: str | None = None,
    **kwargs: Any,
):
    from xorq.config import _backend_init  # noqa: PLC0415

    con = _backend_init()
    return con.register(source, table_name=table_name, **kwargs)


def read_postgres(
    uri: str,
    table_name: str | None = None,
    **kwargs: Any,
):
    from xorq.config import _backend_init  # noqa: PLC0415

    con = _backend_init()
    return con.read_postgres(uri, table_name=table_name, **kwargs)


@functools.cache
def _cached_with_op(op, pretty, compiler):
    expr = op.to_expr()
    sg_expr = compiler.to_sqlglot(expr)
    sql = sg_expr.sql(dialect=compiler.dialect, pretty=pretty)
    return sql


get_compiler = toolz.excepts(
    (XorqError, AttributeError),
    lambda e: e._find_backend(use_default=True).compiler,
    lambda _: Backend.compiler,
)


def to_sql(expr: ir.Expr, compiler=None, pretty: bool = True) -> SQLString:
    """Return the formatted SQL string for an expression.

    Parameters
    ----------
    expr
        Ibis expression.
    compiler
        The target compiler to use to translate the Ibis expr
    pretty
        Whether to use pretty formatting.

    Returns
    -------
    str
        Formatted SQL string

    """

    assert isinstance(expr, ir.Expr)

    if compiler is None:
        compiler = get_compiler(expr)

    unbound = _remove_tag_nodes(expr).unbind().op()
    return SQLString(_cached_with_op(unbound, pretty, compiler))


@tracer.start_as_current_span("_register_and_transform_cache_tables")
def _register_and_transform_cache_tables(expr):
    """This function will sequentially execute any cache node that is not already cached"""

    def fn(node, kwargs):
        if kwargs:
            node = node.__recreate__(kwargs)
        if isinstance(node, CachedNode):
            uncached, cache = node.parent, node.cache
            node = cache.set_default(uncached, uncached.op())
        return node

    op = expr.op()
    out = op.replace(fn)

    return out.to_expr()


@tracer.start_as_current_span("_transform_deferred_reads")
def _transform_deferred_reads(expr):
    dt_to_read = {}

    span = trace.get_current_span()

    def replace_read(node, _kwargs):
        from xorq.expr.relations import Read  # noqa: PLC0415

        if isinstance(node, Read):
            read_kwargs = dict(node.read_kwargs)
            span.add_event(
                "replace_read",
                {
                    "engine": node.source.name,
                    "method_name": node.method_name,
                    "path": str(
                        read_kwargs.get("path")
                        or read_kwargs.get("source")
                        or read_kwargs.get("source_list")
                    ),
                },
            )
            # FIXME: pandas read is not lazy, leave it to the pandas executor to do
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
    >>> import xorq.api as xo
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

    if (con := expr._find_backend(use_default=True)).name == "pandas":
        return _pandas_execute(con, expr, **kwargs)

    batch_reader = to_pyarrow_batches(expr, **kwargs)
    with tracer.start_as_current_span("read_pandas"):
        df = batch_reader.read_pandas(timestamp_as_object=True)
    return expr.__pandas_result__(df)


@tracer.start_as_current_span("_remove_tag_nodes")
def _remove_tag_nodes(expr):
    from xorq.common.utils.graph_utils import replace_nodes  # noqa: PLC0415

    def replacer(node, kwargs):
        if isinstance(node, Tag):
            while isinstance(node, Tag):
                node = node.parent
            node = replace_nodes(replacer, node)
        elif kwargs:
            node = node.__recreate__(kwargs)
        return node

    return replace_nodes(replacer, expr).to_expr()


@tracer.start_as_current_span("_remove_non_hashing_tag_nodes")
def _remove_non_hashing_tag_nodes(expr):
    """Strip Tag nodes but preserve HashingTag nodes during hash computation."""
    from xorq.common.utils.graph_utils import replace_nodes  # noqa: PLC0415

    def replacer(node, kwargs):
        match node:
            case HashingTag():
                if kwargs:
                    node = node.__recreate__(kwargs)
                return node
            case Tag():
                while isinstance(node, Tag) and not isinstance(node, HashingTag):
                    node = node.parent
                return replace_nodes(replacer, node)
            case _:
                if kwargs:
                    node = node.__recreate__(kwargs)
                return node

    return replace_nodes(replacer, expr).to_expr()


@tracer.start_as_current_span("_resolve_params")
def _resolve_params(params):
    """Resolve param keys to a {name: value} dict.

    Accepts a mapping where keys can be:
    - ``xorq.param()`` expressions (NamedScalarParameter)
    - plain strings (param names)

    Raises TypeError for legacy ``ibis.param()`` expressions or unsupported key types.
    """
    from xorq.vendor.ibis.expr.operations.generic import (  # noqa: PLC0415
        ScalarParameter,
    )

    name_values = {}
    for p, v in (params or {}).items():
        match getattr(p, "op", lambda: None)():
            case NamedScalarParameter() as op:
                name_values[op.label] = v
            case ScalarParameter():
                raise TypeError(
                    "Legacy ibis.param() expressions are not supported as param keys. "
                    "Use xorq.param(name, dtype) and pass {name: value} dicts instead."
                )
            case None if isinstance(p, str):
                name_values[p] = v
            case _:
                raise TypeError(f"Unsupported param key type: {type(p)}")
    return name_values


def _transform_expr(expr, params=None, **kwargs):
    """Transform an expression for execution, binding any named scalar parameters."""
    name_values = _resolve_params(params)
    expr = (
        bind_params(expr, name_values)
        if name_values or walk_nodes(NamedScalarParameter, expr)
        else expr
    )
    expr = _remove_tag_nodes(expr)
    expr = _register_and_transform_cache_tables(expr)
    expr, created = register_and_transform_remote_tables(expr, **kwargs)
    expr, dt_to_read = _transform_deferred_reads(expr)
    return (expr, created)


def _pandas_execute(con, expr: ir.Expr, **kwargs):
    from xorq.expr.relations import FlightExpr, FlightUDXF  # noqa: PLC0415

    span = trace.get_current_span()

    node = expr.op()
    if isinstance(node, (FlightExpr, FlightUDXF)):
        # TODO: verify correct caching behavior
        span.set_attribute("engine", "flight")
        df = node.to_rbr().read_pandas(timestamp_as_object=True)
        return expr.__pandas_result__(df)
    params = kwargs.pop("params", None)
    expr, created = _transform_expr(expr, params=params)

    span.set_attribute("engine", "pandas")
    return con.execute(expr, **kwargs)


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
    from xorq.expr.relations import FlightExpr, FlightUDXF  # noqa: PLC0415

    span = trace.get_current_span()

    if isinstance(expr.op(), (FlightExpr, FlightUDXF)):
        # TODO: verify correct caching behavior
        span.set_attribute("engine", "flight")
        return expr.op().to_rbr()
    params = kwargs.pop("params", None)
    expr, created = _transform_expr(expr, params=params)
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


def to_pyarrow_stream(
    expr: ir.Expr,
    sink: Any,
    params: Mapping[ir.Scalar, Any] | None = None,
    **kwargs: Any,
):
    rbr = expr.to_pyarrow_batches(params=params)
    with maybe_open(sink, "wb") as fh:
        try:
            writer = pa.ipc.new_stream(fh, rbr.schema, **kwargs)
            for batch in rbr:
                writer.write_batch(batch)
        finally:
            writer.close()


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
    import pyarrow  # noqa: F401, ICN001, PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    import pyarrow_hotfix  # noqa: F401, PLC0415

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

    import pyarrow  # noqa: F401, ICN001, PLC0415
    import pyarrow.csv as pcsv  # noqa: PLC0415
    import pyarrow_hotfix  # noqa: F401, PLC0415

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
    import pyarrow  # noqa: F401, ICN001, PLC0415
    import pyarrow_hotfix  # noqa: F401, PLC0415

    from xorq.common.utils.io_utils import maybe_open  # noqa: PLC0415

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
    from xorq.config import _backend_init  # noqa: PLC0415

    con = _backend_init()

    suffix = extract_suffix(path).lstrip(".")

    if "storage_options" in kwargs:
        kwargs["storage_options"] = dict(kwargs.pop("storage_options"))

    return con.con.get_object_metadata(path, suffix, **kwargs)


def param(name: str, type, default=None) -> "ir.Scalar":
    """Create a named scalar parameter for use in parameterized expressions.

    Parameters
    ----------
    name
        Human-readable label for the parameter (e.g. ``"cutoff"``).
    type
        ibis data type for the parameter, e.g. ``"float64"``, ``"date"``,
        ``dt.timestamp()``.
    default
        Optional default Python value used when the parameter is not supplied
        at execution time (e.g. ``0.5``, ``datetime.date(2024, 1, 1)``).

    Returns
    -------
    ir.Scalar
        A scalar expression backed by a :class:`NamedScalarParameter` node.
        Pass it to :meth:`execute` via ``params={param_expr: value}``.

    Examples
    --------
    >>> import xorq as xo
    >>> cutoff = xo.param("cutoff", "date")
    >>> threshold = xo.param("threshold", "float64", default=0.5)
    >>> t = xo.memtable({"d": ["2024-01-01", "2024-06-01"], "v": [1, 2]})
    """
    dtype = dt.dtype(type)
    return NamedScalarParameter(dtype=dtype, label=name, default=default).to_expr()


def bind_params(expr, params: dict) -> "ir.Expr":
    """Bind named parameters by name→value dict, applying defaults for omitted ones.

    Parameters
    ----------
    expr
        Expression containing :class:`NamedScalarParameter` nodes.
    params
        Mapping of parameter name to Python value.
    Raises
    ------
    ValueError
        If any required parameter (no default) is absent from *params*.
    TypeError
        If *params* contains names not found in *expr*, or values
        incompatible with the declared dtype.
    """
    named = {node.label: node for node in walk_nodes(NamedScalarParameter, expr)}

    errors = []

    inapplicable = tuple(sorted(set(params) - set(named)))
    if inapplicable:
        errors.append(f"Got unexpected extra parameter: {', '.join(inapplicable)}")

    for name, value in params.items():
        if name in named and not dt.infer(value).castable(named[name].dtype):
            errors.append(
                f"Parameter {name!r}: value {value!r} (inferred {dt.infer(value)}) "
                f"is not compatible with declared dtype {named[name].dtype}"
            )

    if errors:
        raise TypeError("\n".join(errors))

    missing = tuple(
        f"{name} ({node.dtype})"
        for name, node in named.items()
        if name not in params and node.default is None
    )
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")

    op_params = {
        node: value
        for name, node in named.items()
        if (value := params.get(name, node.default)) is not None
    }

    def replacer(node, kwargs):
        if kwargs:
            node = node.__recreate__(kwargs)
        if isinstance(node, NamedScalarParameter) and node in op_params:
            return ops.Literal(value=op_params[node], dtype=node.dtype)
        return node

    return replace_nodes(replacer, expr).to_expr()
