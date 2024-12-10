"""LETSQL expression API definitions."""

from __future__ import annotations

import datetime
import functools
import operator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, overload

import ibis
import ibis.expr.builders as bl
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
import toolz
from ibis import api
from ibis.backends.sql.dialects import DataFusion
from ibis.common.deferred import Deferred, _, deferrable
from ibis.expr.schema import Schema
from ibis.expr.sql import SQLString
from ibis.expr.types import (
    Column,
    DateValue,
    Scalar,
    Table,
    TimestampValue,
    TimeValue,
    Value,
    array,
    literal,
    map,
    null,
    struct,
)

from letsql.expr.relations import (
    RemoteTable,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from ibis.expr.schema import SchemaLike
    import pyarrow as pa
    import pandas as pd

__all__ = (
    "Column",
    "Deferred",
    "Scalar",
    "Schema",
    "Table",
    "Value",
    "_",
    "aggregate",
    "and_",
    "array",
    "asc",
    "asof_join",
    "case",
    "coalesce",
    "cross_join",
    "cume_dist",
    "date",
    "deferred",
    "dense_rank",
    "desc",
    "difference",
    "e",
    "execute",
    "following",
    "greatest",
    "ifelse",
    "infer_dtype",
    "infer_schema",
    "intersect",
    "interval",
    "join",
    "least",
    "literal",
    "map",
    "memtable",
    "now",
    "ntile",
    "null",
    "or_",
    "param",
    "percent_rank",
    "pi",
    "preceding",
    "random",
    "range",
    "rank",
    "read_csv",
    "read_parquet",
    "read_postgres",
    "read_sqlite",
    "register",
    "row_number",
    "schema",
    "struct",
    "table",
    "time",
    "today",
    "to_parquet",
    "to_pyarrow",
    "to_pyarrow_batches",
    "to_sql",
    "timestamp",
    "union",
    "uuid",
)

infer_dtype = dt.infer
infer_schema = sch.infer
aggregate = ir.Table.aggregate
cross_join = ir.Table.cross_join
join = ir.Table.join
asof_join = ir.Table.asof_join

e = ops.E().to_expr()
pi = ops.Pi().to_expr()

deferred = _


def param(type: Union[dt.DataType, str]) -> ir.Scalar:
    """Create a deferred parameter of a given type.

    Parameters
    ----------
    type
        The type of the unbound parameter, e.g., double, int64, date, etc.

    Returns
    -------
    Scalar
        A scalar expression backend by a parameter

    Examples
    --------
    >>> from datetime import date
    >>> import letsql
    >>> start = letsql.param("date")
    >>> t = letsql.memtable(
    ...     {
    ...         "date_col": [date(2013, 1, 1), date(2013, 1, 2), date(2013, 1, 3)],
    ...         "value": [1.0, 2.0, 3.0],
    ...     },
    ... )
    >>> expr = t.filter(t.date_col >= start).value.sum()
    >>> expr.execute(params={start: date(2013, 1, 1)})
    6.0
    >>> expr.execute(params={start: date(2013, 1, 2)})
    5.0
    >>> expr.execute(params={start: date(2013, 1, 3)})
    3.0

    """
    return api.param(type)


def schema(
    pairs: SchemaLike | None = None,
    names: Iterable[str] | None = None,
    types: Iterable[str | dt.DataType] | None = None,
) -> sch.Schema:
    """Validate and return a [`Schema`](./schemas.qmd#ibis.expr.schema.Schema) object.

    Parameters
    ----------
    pairs
        List or dictionary of name, type pairs. Mutually exclusive with `names`
        and `types` arguments.
    names
        Field names. Mutually exclusive with `pairs`.
    types
        Field types. Mutually exclusive with `pairs`.

    Returns
    -------
    Schema
        An ibis schema

    Examples
    --------
    >>> from letsql import schema
    >>> sc = schema([("foo", "string"), ("bar", "int64"), ("baz", "boolean")])
    >>> sc = schema(names=["foo", "bar", "baz"], types=["string", "int64", "boolean"])
    >>> sc = schema(dict(foo="string")) # no-op

    """
    return api.schema(pairs=pairs, names=names, types=types)


def table(
    schema: SchemaLike | None = None,
    name: str | None = None,
    catalog: str | None = None,
    database: str | None = None,
) -> ir.Table:
    """Create a table literal or an abstract table without data.

    Ibis uses the word database to refer to a collection of tables, and the word
    catalog to refer to a collection of databases. You can use a combination of
    `catalog` and `database` to specify a hierarchical location for table.

    Parameters
    ----------
    schema
        A schema for the table
    name
        Name for the table. One is generated if this value is `None`.
    catalog
        A collection of database.
    database
        A collection of tables. Required if catalog is not `None`.

    Returns
    -------
    Table
        A table expression

    Examples
    --------
    Create a table with no data backing it

    >>> import letsql
    >>> letsql.options.interactive = False
    >>> t = letsql.table(schema=dict(a="int", b="string"), name="t")
    >>> t
    UnboundTable: t
      a int64
      b string


    Create a table with no data backing it in a specific location

    >>> import letsql
    >>> letsql.options.interactive = False
    >>> t = letsql.table(schema=dict(a="int"), name="t", catalog="cat", database="db")
    >>> t
    UnboundTable: cat.db.t
      a int64
    """
    return api.table(schema=schema, name=name, catalog=catalog, database=database)


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
    >>> import letsql
    >>> letsql.options.interactive = False
    >>> t = letsql.memtable([{"a": 1}, {"a": 2}])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             a
          0  1
          1  2

    >>> t = letsql.memtable([{"a": 1, "b": "foo"}, {"a": 2, "b": "baz"}])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             a    b
          0  1  foo
          1  2  baz

    Create a table literal without column names embedded in the data and pass
    `columns`

    >>> t = letsql.memtable([(1, "foo"), (2, "baz")], columns=["a", "b"])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             a    b
          0  1  foo
          1  2  baz

    Create a table literal without column names embedded in the data. Ibis
    generates column names if none are provided.

    >>> t = letsql.memtable([(1, "foo"), (2, "baz")])
    >>> t
    InMemoryTable
      data:
        PandasDataFrameProxy:
             col0 col1
          0     1  foo
          1     2  baz

    """
    return api.memtable(data, columns=columns, schema=schema, name=name)


def desc(expr: ir.Column | str) -> ir.Value:
    """Create a descending sort key from `expr` or column name.

    Parameters
    ----------
    expr
        The expression or column name to use for sorting

    See Also
    --------
    [`Value.desc()`](./expression-generic.qmd#ibis.expr.types.generic.Value.desc)

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> t = letsql.examples.penguins.fetch()
    >>> t[["species", "year"]].order_by(letsql.desc("year")).head()
    ┏━━━━━━━━━┳━━━━━━━┓
    ┃ species ┃ year  ┃
    ┡━━━━━━━━━╇━━━━━━━┩
    │ string  │ int64 │
    ├─────────┼───────┤
    │ Adelie  │  2009 │
    │ Adelie  │  2009 │
    │ Adelie  │  2009 │
    │ Adelie  │  2009 │
    │ Adelie  │  2009 │
    └─────────┴───────┘

    Returns
    -------
    ir.ValueExpr
        An expression

    """
    return api.desc(expr)


def asc(expr: ir.Column | str) -> ir.Value:
    """Create an ascending sort key from `asc` or column name.

    Parameters
    ----------
    expr
        The expression or column name to use for sorting

    See Also
    --------
    [`Value.asc()`](./expression-generic.qmd#ibis.expr.types.generic.Value.asc)

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> t = letsql.examples.penguins.fetch()
    >>> t[["species", "year"]].order_by(letsql.asc("year")).head()
    ┏━━━━━━━━━┳━━━━━━━┓
    ┃ species ┃ year  ┃
    ┡━━━━━━━━━╇━━━━━━━┩
    │ string  │ int64 │
    ├─────────┼───────┤
    │ Adelie  │  2007 │
    │ Adelie  │  2007 │
    │ Adelie  │  2007 │
    │ Adelie  │  2007 │
    │ Adelie  │  2007 │
    └─────────┴───────┘

    Returns
    -------
    ir.ValueExpr
        An expression

    """
    return api.asc(expr)


def preceding(value) -> ir.Value:
    return api.preceding(value)


def following(value) -> ir.Value:
    return api.following(value)


def and_(*predicates: ir.BooleanValue) -> ir.BooleanValue:
    """Combine multiple predicates using `&`.

    Parameters
    ----------
    predicates
        Boolean value expressions

    Returns
    -------
    BooleanValue
        A new predicate that evaluates to True if all composing predicates are
        True. If no predicates were provided, returns True.

    """
    return api.and_(*predicates)


def or_(*predicates: ir.BooleanValue) -> ir.BooleanValue:
    """Combine multiple predicates using `|`.

    Parameters
    ----------
    predicates
        Boolean value expressions

    Returns
    -------
    BooleanValue
        A new predicate that evaluates to True if any composing predicates are
        True. If no predicates were provided, returns False.

    """
    return api.or_(*predicates)


def random() -> ir.FloatingScalar:
    """Return a random floating point number in the range [0.0, 1.0).

    Similar to [](`random.random`) in the Python standard library.

    ::: {.callout-note}
    ## Repeated use of `random`

    `ibis.random()` will generate a column of distinct random numbers even if
    the same instance of `ibis.random()` is re-used.

    When Ibis compiles an expression to SQL, each place where `random` is used
    will render as a separate call to the given backend's random number
    generator.

    >>> import letsql
    >>> r_a = letsql.random() # doctest: +SKIP

    Returns
    -------
    FloatingScalar
        Random float value expression

    """
    return api.random()


def uuid() -> ir.UUIDScalar:
    """Return a random UUID version 4 value.

    Similar to [('uuid.uuid4`) in the Python standard library.

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> letsql.uuid()  # doctest: +SKIP
    UUID('e57e927b-aed2-483b-9140-dc32a26cad95')

    Returns
    -------
    UUIDScalar
        Random UUID value expression
    """
    return api.uuid()


def case() -> bl.SearchedCaseBuilder:
    """Begin constructing a case expression.

    Use the `.when` method on the resulting object followed by `.end` to create a
    complete case expression.

    Returns
    -------
    SearchedCaseBuilder
        A builder object to use for constructing a case expression.

    See Also
    --------
    [`Value.case()`](./expression-generic.qmd#ibis.expr.types.generic.Value.case)

    Examples
    --------
    >>> import letsql
    >>> from ibis import _
    >>> letsql.options.interactive = True
    >>> t = letsql.memtable(
    ...     {
    ...         "left": [1, 2, 3, 4],
    ...         "symbol": ["+", "-", "*", "/"],
    ...         "right": [5, 6, 7, 8],
    ...     }
    ... )
    >>> t.mutate(
    ...     result=(
    ...         letsql.case()
    ...         .when(_.symbol == "+", _.left + _.right)
    ...         .when(_.symbol == "-", _.left - _.right)
    ...         .when(_.symbol == "*", _.left * _.right)
    ...         .when(_.symbol == "/", _.left / _.right)
    ...         .end()
    ...     )
    ... )
    ┏━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━┓
    ┃ left  ┃ symbol ┃ right ┃ result  ┃
    ┡━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━┩
    │ int64 │ string │ int64 │ float64 │
    ├───────┼────────┼───────┼─────────┤
    │     1 │ +      │     5 │     6.0 │
    │     2 │ -      │     6 │    -4.0 │
    │     3 │ *      │     7 │    21.0 │
    │     4 │ /      │     8 │     0.5 │
    └───────┴────────┴───────┴─────────┘

    """
    return api.case()


def now() -> ir.TimestampScalar:
    """Return an expression that will compute the current timestamp.

    Returns
    -------
    TimestampScalar
        An expression representing the current timestamp.

    """
    return api.now()


def today() -> ir.DateScalar:
    """Return an expression that will compute the current date.

    Returns
    -------
    DateScalar
        An expression representing the current date.

    """
    return api.today()


def rank() -> ir.IntegerColumn:
    """Compute position of first element within each equal-value group in sorted order.

    Equivalent to SQL's `RANK()` window function.

    Returns
    -------
    Int64Column
        The min rank

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> t = letsql.memtable({"values": [1, 2, 1, 2, 3, 2]})
    >>> t.mutate(rank=letsql.rank().over(order_by=t.values))
    ┏━━━━━━━━┳━━━━━━━┓
    ┃ values ┃ rank  ┃
    ┡━━━━━━━━╇━━━━━━━┩
    │ int64  │ int64 │
    ├────────┼───────┤
    │      1 │     0 │
    │      1 │     0 │
    │      2 │     2 │
    │      2 │     2 │
    │      2 │     2 │
    │      3 │     5 │
    └────────┴───────┘

    """
    return api.rank()


def dense_rank() -> ir.IntegerColumn:
    """Position of first element within each group of equal values.

    Values are returned in sorted order and duplicate values are ignored.

    Equivalent to SQL's `DENSE_RANK()`.

    Returns
    -------
    IntegerColumn
        The rank

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> t = letsql.memtable({"values": [1, 2, 1, 2, 3, 2]})
    >>> t.mutate(rank=letsql.dense_rank().over(order_by=t.values))
    ┏━━━━━━━━┳━━━━━━━┓
    ┃ values ┃ rank  ┃
    ┡━━━━━━━━╇━━━━━━━┩
    │ int64  │ int64 │
    ├────────┼───────┤
    │      1 │     0 │
    │      1 │     0 │
    │      2 │     1 │
    │      2 │     1 │
    │      2 │     1 │
    │      3 │     2 │
    └────────┴───────┘

    """
    return api.dense_rank()


def percent_rank() -> ir.FloatingColumn:
    """Return the relative rank of the values in the column.

    Returns
    -------
    FloatingColumn
        The percent rank

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> t = letsql.memtable({"values": [1, 2, 1, 2, 3, 2]})
    >>> t.mutate(pct_rank=letsql.percent_rank().over(order_by=t.values))
    ┏━━━━━━━━┳━━━━━━━━━━┓
    ┃ values ┃ pct_rank ┃
    ┡━━━━━━━━╇━━━━━━━━━━┩
    │ int64  │ float64  │
    ├────────┼──────────┤
    │      1 │      0.0 │
    │      1 │      0.0 │
    │      2 │      0.4 │
    │      2 │      0.4 │
    │      2 │      0.4 │
    │      3 │      1.0 │
    └────────┴──────────┘

    """
    return api.percent_rank()


def cume_dist() -> ir.FloatingColumn:
    """Return the cumulative distribution over a window.

    Returns
    -------
    FloatingColumn
        The cumulative distribution

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> t = letsql.memtable({"values": [1, 2, 1, 2, 3, 2]})
    >>> t.mutate(dist=letsql.cume_dist().over(order_by=t.values))
    ┏━━━━━━━━┳━━━━━━━━━━┓
    ┃ values ┃ dist     ┃
    ┡━━━━━━━━╇━━━━━━━━━━┩
    │ int64  │ float64  │
    ├────────┼──────────┤
    │      1 │ 0.333333 │
    │      1 │ 0.333333 │
    │      2 │ 0.833333 │
    │      2 │ 0.833333 │
    │      2 │ 0.833333 │
    │      3 │ 1.000000 │
    └────────┴──────────┘

    """
    return api.cume_dist()


def ntile(buckets: int | ir.IntegerValue) -> ir.IntegerColumn:
    """Return the integer number of a partitioning of the column values.

    Parameters
    ----------
    buckets
        Number of buckets to partition into

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> t = letsql.memtable({"values": [1, 2, 1, 2, 3, 2]})
    >>> t.mutate(ntile=letsql.ntile(2).over(order_by=t.values))
    ┏━━━━━━━━┳━━━━━━━┓
    ┃ values ┃ ntile ┃
    ┡━━━━━━━━╇━━━━━━━┩
    │ int64  │ int64 │
    ├────────┼───────┤
    │      1 │     0 │
    │      1 │     0 │
    │      2 │     0 │
    │      2 │     1 │
    │      2 │     1 │
    │      3 │     1 │
    └────────┴───────┘

    """
    return api.ntile(buckets)


def row_number() -> ir.IntegerColumn:
    """Return an analytic function expression for the current row number.

    ::: {.callout-note}
    `row_number` is normalized across backends to start at 0
    :::

    Returns
    -------
    IntegerColumn
        A column expression enumerating rows

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> t = letsql.memtable({"values": [1, 2, 1, 2, 3, 2]})
    >>> t.mutate(rownum=letsql.row_number())
    ┏━━━━━━━━┳━━━━━━━━┓
    ┃ values ┃ rownum ┃
    ┡━━━━━━━━╇━━━━━━━━┩
    │ int64  │ int64  │
    ├────────┼────────┤
    │      1 │      0 │
    │      2 │      1 │
    │      1 │      2 │
    │      2 │      3 │
    │      3 │      4 │
    │      2 │      5 │
    └────────┴────────┘

    """
    return api.row_number()


def read_csv(
    sources: str | Path | Sequence[str | Path],
    table_name: str | None = None,
    **kwargs: Any,
) -> ir.Table:
    """Lazily load a CSV or set of CSVs.

    This function delegates to the `read_csv` method on the current default
    backend (DuckDB or `ibis.config.default_backend`).

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
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> lines = '''a,b
    ... 1,d
    ... 2,
    ... ,f
    ... '''
    >>> with open("/tmp/lines.csv", mode="w") as f:
    ...     nbytes = f.write(lines)  # nbytes is unused
    >>> t = letsql.read_csv("/tmp/lines.csv")
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
    from letsql.config import _backend_init

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
    >>> import letsql
    >>> import pandas as pd
    >>> letsql.options.interactive = True
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": list("ghi")})
    >>> df
       a  b
    0  1  g
    1  2  h
    2  3  i
    >>> df.to_parquet("/tmp/data.parquet")
    >>> t = letsql.read_parquet("/tmp/data.parquet")
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
    from letsql.config import _backend_init

    con = _backend_init()
    return con.read_parquet(sources, table_name=table_name, **kwargs)


def register(
    source: str | Path | pa.Table | pa.RecordBatch | pa.Dataset | pd.DataFrame,
    table_name: str | None = None,
    **kwargs: Any,
):
    from letsql.config import _backend_init

    con = _backend_init()
    return con.read_parquet(source, table_name=table_name, **kwargs)


def read_postgres(
    uri: str,
    table_name: str | None = None,
    **kwargs: Any,
):
    from letsql.config import _backend_init

    con = _backend_init()
    return con.read_postgres(uri, table_name=table_name, **kwargs)


def read_sqlite(path: str | Path, *, table_name: str | None = None):
    from letsql.config import _backend_init

    con = _backend_init()
    return con.read_sqlite(path, table_name=table_name)


def union(table: ir.Table, *rest: ir.Table, distinct: bool = False) -> ir.Table:
    """Compute the set union of multiple table expressions.

    The input tables must have identical schemas.

    Parameters
    ----------
    table
        A table expression
    *rest
        Additional table expressions
    distinct
        Only return distinct rows

    Returns
    -------
    Table
        A new table containing the union of all input tables.

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> t1 = letsql.memtable({"a": [1, 2]})
    >>> t1
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    │     2 │
    └───────┘
    >>> t2 = letsql.memtable({"a": [2, 3]})
    >>> t2
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     2 │
    │     3 │
    └───────┘
    >>> letsql.union(t1, t2)  # union all by default
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    │     2 │
    │     2 │
    │     3 │
    └───────┘
    >>> letsql.union(t1, t2, distinct=True).order_by("a")
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    │     2 │
    │     3 │
    └───────┘

    """
    return table.union(*rest, distinct=distinct) if rest else table


def intersect(table: ir.Table, *rest: ir.Table, distinct: bool = True) -> ir.Table:
    """Compute the set intersection of multiple table expressions.

    The input tables must have identical schemas.

    Parameters
    ----------
    table
        A table expression
    *rest
        Additional table expressions
    distinct
        Only return distinct rows

    Returns
    -------
    Table
        A new table containing the intersection of all input tables.

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> t1 = letsql.memtable({"a": [1, 2]})
    >>> t1
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    │     2 │
    └───────┘
    >>> t2 = letsql.memtable({"a": [2, 3]})
    >>> t2
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     2 │
    │     3 │
    └───────┘
    >>> letsql.intersect(t1, t2)
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     2 │
    └───────┘

    """
    return table.intersect(*rest, distinct=distinct) if rest else table


def difference(table: ir.Table, *rest: ir.Table, distinct: bool = True) -> ir.Table:
    """Compute the set difference of multiple table expressions.

    The input tables must have identical schemas.

    Parameters
    ----------
    table
        A table expression
    *rest
        Additional table expressions
    distinct
        Only diff distinct rows not occurring in the calling table

    Returns
    -------
    Table
        The rows present in `self` that are not present in `tables`.

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> t1 = letsql.memtable({"a": [1, 2]})
    >>> t1
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    │     2 │
    └───────┘
    >>> t2 = letsql.memtable({"a": [2, 3]})
    >>> t2
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     2 │
    │     3 │
    └───────┘
    >>> letsql.difference(t1, t2)
    ┏━━━━━━━┓
    ┃ a     ┃
    ┡━━━━━━━┩
    │ int64 │
    ├───────┤
    │     1 │
    └───────┘

    """
    return table.difference(*rest, distinct=distinct) if rest else table


@deferrable
def ifelse(condition: Any, true_expr: Any, false_expr: Any) -> ir.Value:
    """Construct a ternary conditional expression.

    Parameters
    ----------
    condition
        A boolean expression
    true_expr
        Expression to return if `condition` evaluates to `True`
    false_expr
        Expression to return if `condition` evaluates to `False` or `NULL`

    Returns
    -------
    Value : ir.Value
        The value of `true_expr` if `condition` is `True` else `false_expr`

    See Also
    --------
    [`BooleanValue.ifelse()`](./expression-numeric.qmd#ibis.expr.types.logical.BooleanValue.ifelse)

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> t = letsql.memtable({"condition": [True, False, True, None]})
    >>> letsql.ifelse(t.condition, "yes", "no")
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ IfElse(condition, 'yes', 'no') ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ string                         │
    ├────────────────────────────────┤
    │ yes                            │
    │ no                             │
    │ yes                            │
    │ no                             │
    └────────────────────────────────┘

    """
    return api.ifelse(condition, true_expr, false_expr)


@deferrable
def coalesce(*args: Any) -> ir.Value:
    """Return the first non-null value from `args`.

    Parameters
    ----------
    args
        Arguments from which to choose the first non-null value

    Returns
    -------
    Value
        Coalesced expression

    See Also
    --------
    [`Value.coalesce()`](#ibis.expr.types.generic.Value.coalesce)
    [`Value.fill_null()`](#ibis.expr.types.generic.Value.fill_null)

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> letsql.coalesce(None, 4, 5)
    4

    """
    return api.coalesce(*args)


@deferrable
def greatest(*args: Any) -> ir.Value:
    """Compute the largest value among the supplied arguments.

    Parameters
    ----------
    args
        Arguments to choose from

    Returns
    -------
    Value
        Maximum of the passed arguments

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> letsql.greatest(None, 4, 5)
    5

    """
    return api.greatest(*args)


@deferrable
def least(*args: Any) -> ir.Value:
    """Compute the smallest value among the supplied arguments.

    Parameters
    ----------
    args
        Arguments to choose from

    Returns
    -------
    Value
        Minimum of the passed arguments

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True
    >>> letsql.least(None, 4, 5)
    4

    """
    return api.least(*args)


@functools.singledispatch
def range(start, stop, step) -> ir.ArrayValue:
    """Generate a range of values.

    Integer ranges are supported, as well as timestamp ranges.

    ::: {.callout-note}
    `start` is inclucive and `stop` is exclusive, just like Python's builtin
    [`range`](range).

    When `step` equals 0, however, this function will return an empty array.

    Python's `range` will raise an exception when `step` is zero.
    :::

    Parameters
    ----------
    start
        Lower bound of the range, inclusive.
    stop
        Upper bound of the range, exclusive.
    step
        Step value. Optional, defaults to 1.

    Returns
    -------
    ArrayValue
        An array of values

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True

    Range using only a stop argument

    >>> letsql.range(5)
    [0, 1, ... +3]

    Simple range using start and stop

    >>> letsql.range(1, 5)
    [1, 2, ... +2]


    Generate an empty range

    >>> letsql.range(0)
    []

    Negative step values are supported

    >>> letsql.range(10, 4, -2)
    [10, 8, ... +1]


    `ibis.range` behaves the same as Python's range ...

    >>> letsql.range(0, 7, -1)
    []

    ... except when the step is zero, in which case `ibis.range` returns an
    empty array

    >>> letsql.range(0, 5, 0)
    []

    Because the resulting expression is array, you can unnest the values

    >>> letsql.range(5).unnest().name("numbers")
    ┏━━━━━━━━━┓
    ┃ numbers ┃
    ┡━━━━━━━━━┩
    │ int8    │
    ├─────────┤
    │       0 │
    │       1 │
    │       2 │
    │       3 │
    │       4 │
    └─────────┘

    """
    raise NotImplementedError()


@range.register(int)
@range.register(ir.IntegerValue)
def _int_range(
    start: int,
    stop: int | ir.IntegerValue | None = None,
    step: int | ir.IntegerValue | None = None,
) -> ir.ArrayValue:
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    return ops.IntegerRange(start=start, stop=stop, step=step).to_expr()


@overload
def timestamp(value_or_year: Any, /, timezone: str | None = None) -> TimestampValue: ...


@overload
def timestamp(
    value_or_year: int | ir.IntegerValue | Deferred,
    month: int | ir.IntegerValue | Deferred,
    day: int | ir.IntegerValue | Deferred,
    hour: int | ir.IntegerValue | Deferred,
    minute: int | ir.IntegerValue | Deferred,
    second: int | ir.IntegerValue | Deferred,
    /,
    timezone: str | None = None,
) -> TimestampValue: ...


@deferrable
def timestamp(
    value_or_year,
    month=None,
    day=None,
    hour=None,
    minute=None,
    second=None,
    /,
    timezone=None,
):
    """Construct a timestamp scalar or column.

    Parameters
    ----------
    value_or_year
        Either a string value or `datetime.datetime` to coerce to a timestamp,
        or an integral value representing the timestamp year component.
    month
        The timestamp month component; required if `value_or_year` is a year.
    day
        The timestamp day component; required if `value_or_year` is a year.
    hour
        The timestamp hour component; required if `value_or_year` is a year.
    minute
        The timestamp minute component; required if `value_or_year` is a year.
    second
        The timestamp second component; required if `value_or_year` is a year.
    timezone
        The timezone name, or none for a timezone-naive timestamp.

    Returns
    -------
    TimestampValue
        A timestamp expression

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True

    Create a timestamp scalar from a string

    >>> letsql.timestamp("2023-01-02T03:04:05")
    Timestamp('2023-01-02 03:04:05')


    Create a timestamp scalar from components

    >>> letsql.timestamp(2023, 1, 2, 3, 4, 5)
    Timestamp('2023-01-02 03:04:05')


    Create a timestamp column from components

    >>> t = letsql.memtable({"y": [2001, 2002], "m": [1, 4], "d": [2, 5], "h": [3, 6]})
    >>> letsql.timestamp(t.y, t.m, t.d, t.h, 0, 0).name("timestamp")
    ┏━━━━━━━━━━━━━━━━━━━━━┓
    ┃ timestamp           ┃
    ┡━━━━━━━━━━━━━━━━━━━━━┩
    │ timestamp           │
    ├─────────────────────┤
    │ 2001-01-02 03:00:00 │
    │ 2002-04-05 06:00:00 │
    └─────────────────────┘

    """
    return ibis.timestamp(value_or_year, month, day, hour, minute, second, timezone)


@overload
def date(
    value_or_year: int | ir.IntegerValue | Deferred,
    month: int | ir.IntegerValue | Deferred,
    day: int | ir.IntegerValue | Deferred,
    /,
) -> DateValue: ...


@overload
def date(value_or_year: Any, /) -> DateValue: ...


@deferrable
def date(value_or_year, month=None, day=None, /):
    return ibis.date(value_or_year, month, day)


@overload
def time(
    value_or_hour: int | ir.IntegerValue | Deferred,
    minute: int | ir.IntegerValue | Deferred,
    second: int | ir.IntegerValue | Deferred,
    /,
) -> TimeValue: ...


@overload
def time(value_or_hour: Any, /) -> TimeValue: ...


@deferrable
def time(value_or_hour, minute=None, second=None, /):
    """Return a time literal if `value` is coercible to a time.

    Parameters
    ----------
    value_or_hour
        Either a string value or `datetime.time` to coerce to a time, or
        an integral value representing the time hour component.
    minute
        The time minute component; required if `value_or_hour` is an hour.
    second
        The time second component; required if `value_or_hour` is an hour.

    Returns
    -------
    TimeValue
        A time expression

    Examples
    --------
    >>> import letsql
    >>> letsql.options.interactive = True

    Create a time scalar from a string

    >>> letsql.time("01:02:03")
    datetime.time(1, 2, 3)


    Create a time scalar from hour, minute, and second

    >>> letsql.time(1, 2, 3)
    datetime.time(1, 2, 3)


    Create a time column from hour, minute, and second

    >>> t = letsql.memtable({"h": [1, 4], "m": [2, 5], "s": [3, 6]})
    >>> letsql.time(t.h, t.m, t.s).name("time")
    ┏━━━━━━━━━━┓
    ┃ time     ┃
    ┡━━━━━━━━━━┩
    │ time     │
    ├──────────┤
    │ 01:02:03 │
    │ 04:05:06 │
    └──────────┘

    """
    return ibis.time(value_or_hour, minute, second)


def interval(
    value: int | datetime.timedelta | None = None,
    unit: str = "s",
    *,
    years: int | None = None,
    quarters: int | None = None,
    months: int | None = None,
    weeks: int | None = None,
    days: int | None = None,
    hours: int | None = None,
    minutes: int | None = None,
    seconds: int | None = None,
    milliseconds: int | None = None,
    microseconds: int | None = None,
    nanoseconds: int | None = None,
) -> ir.IntervalScalar:
    """Return an interval literal expression.

    Parameters
    ----------
    value
        Interval value.
    unit
        Unit of `value`
    years
        Number of years
    quarters
        Number of quarters
    months
        Number of months
    weeks
        Number of weeks
    days
        Number of days
    hours
        Number of hours
    minutes
        Number of minutes
    seconds
        Number of seconds
    milliseconds
        Number of milliseconds
    microseconds
        Number of microseconds
    nanoseconds
        Number of nanoseconds

    Returns
    -------
    IntervalScalar
        An interval expression

    """
    return ibis.interval(
        value,
        unit,
        years=years,
        quarters=quarters,
        months=months,
        weeks=weeks,
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        milliseconds=milliseconds,
        microseconds=microseconds,
        nanoseconds=nanoseconds,
    )


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
    from letsql.config import _backend_init

    con = _backend_init()
    sg_expr = con.compiler.to_sqlglot(expr.unbind())
    sql = sg_expr.sql(dialect=DataFusion, pretty=pretty)
    return SQLString(sql)


def _check_collisions(expr: ir.Expr):
    names_to_backends = toolz.groupby(
        operator.itemgetter(0),
        ((t.name, t.source, id(t.source)) for t in expr.op().find(ops.DatabaseTable)),
    )
    bad_names = tuple(
        (name, tuple(con for _, con, _ in vs))
        for name, vs in names_to_backends.items()
        if len(vs) > 1
    )

    if bad_names:
        raise ValueError(f"name collision detected: {bad_names}")


def execute(expr: ir.Expr, **kwargs: Any):
    import letsql

    _check_collisions(expr)
    con = letsql.connect()
    for t in expr.op().find(ops.DatabaseTable):
        if t not in con._sources.sources and not isinstance(t, RemoteTable):
            con.register(t.to_expr(), t.name)

    return con.execute(expr, **kwargs)


def to_pyarrow_batches(
    expr: ir.Expr,
    *,
    chunk_size: int = 1_000_000,
    **kwargs: Any,
):
    import letsql

    _check_collisions(expr)
    con = letsql.connect()
    for t in expr.op().find(ops.DatabaseTable):
        if t not in con._sources.sources and not isinstance(t, RemoteTable):
            con.register(t.to_expr(), t.name)
    return con.to_pyarrow_batches(expr, chunk_size=chunk_size, **kwargs)


def to_pyarrow(expr: ir.Expr, **kwargs: Any):
    import letsql

    _check_collisions(expr)
    con = letsql.connect()
    for t in expr.op().find(ops.DatabaseTable):
        if t not in con._sources.sources and not isinstance(t, RemoteTable):
            con.register(t.to_expr(), t.name)

    return con.to_pyarrow(expr, **kwargs)


def to_parquet(expr: ir.Expr, path: str | Path, **kwargs: Any):
    import letsql

    _check_collisions(expr)
    con = letsql.connect()
    for t in expr.op().find(ops.DatabaseTable):
        if t not in con._sources.sources and not isinstance(t, RemoteTable):
            con.register(t.to_expr(), t.name)

    return con.to_parquet(expr, path, **kwargs)
