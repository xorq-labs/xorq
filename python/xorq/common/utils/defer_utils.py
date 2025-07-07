from itertools import chain
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow as pa
import toolz

import xorq as xo
import xorq.vendor.ibis.expr.types as ir
from xorq.backends.let import Backend
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_read_path_stat,
)
from xorq.common.utils.inspect_utils import (
    get_arguments,
)
from xorq.expr.relations import (
    Read,
)
from xorq.vendor import ibis
from xorq.vendor.ibis import Schema
from xorq.vendor.ibis.util import (
    gen_name,
    normalize_filenames,
)


DEFAULT_CHUNKSIZE = 10_000


def make_read_kwargs(f, *args, **kwargs):
    # FIXME: if any kwarg is a dictionary, we'll fail Concrete's hashable requirement, so just pickle
    read_kwargs = get_arguments(f, *args, **kwargs)
    kwargs = read_kwargs.pop("kwargs", {})
    tpl = tuple(read_kwargs.items()) + tuple(kwargs.items())
    return tpl


@toolz.curry
def infer_csv_schema_pandas(path, chunksize=DEFAULT_CHUNKSIZE, **kwargs):
    path = normalize_filenames(path)
    gen = pd.read_csv(path[0], chunksize=chunksize, **kwargs)
    df = next(gen)
    batch = pa.RecordBatch.from_pandas(df)
    schema = ibis.Schema.from_pyarrow(batch.schema)
    return schema


def read_csv_rbr(*args, schema=None, chunksize=DEFAULT_CHUNKSIZE, dtype=None, **kwargs):
    """Deferred and streaming csv reading via pandas"""
    if dtype is not None:
        raise Exception("pass `dtype` as pyarrow `schema`")
    if chunksize is None:
        raise ValueError("chunksize must not be `None`")
    if schema is not None:
        schema = xo.schema(schema)
        dtype = {col: typ.to_pandas() for col, typ in schema.items()}
        schema = schema.to_pyarrow()
    # schema is always nullable (this is good)
    paths = normalize_filenames(*args)

    gen = map(
        pa.RecordBatch.from_pandas,
        chain.from_iterable(
            pd.read_csv(
                path,
                dtype=dtype,
                chunksize=chunksize,
                **kwargs,
            )
            for path in paths
        ),
    )
    if schema is None:
        (el, gen) = toolz.peek(gen)
        schema = el.schema

    def cast_gen():
        yield from (batch.cast(schema) for batch in gen)

    rbr = pa.RecordBatchReader.from_batches(
        schema,
        cast_gen(),
    )
    return rbr


def deferred_read_csv(
    con: Backend,
    path: str | Path,
    table_name: str | None = None,
    schema: Schema | None = None,
    normalize_method: Callable = normalize_read_path_stat,
    **kwargs,
) -> ir.Table:
    """
    Create a deferred read operation for CSV files that will execute only when needed.

    This function creates a representation of a read operation that doesn't immediately
    load data into memory. Instead, it registers the operation to be performed when
    the resulting expression is executed.

    The function works with different backend engines (pandas, duckdb, postgres, etc.)
    and adapts the read parameters accordingly.


    Parameters
    ----------
    con : Backend
        The connection object representing the backend where the CSV will be read.
        This can be any backend that supports reading CSV files (pandas, duckdb,
        postgres, etc.).

    path : str or Path
        The path to the CSV file to be read. This can be a local file path or a URL.

    table_name : str, optional
        The name to give to the resulting table in the backend. If not provided,
        a unique name will be generated automatically.

    schema : Schema, optional
        The schema definition for the CSV data. If not provided, the schema will
        be inferred from the data by sampling the CSV file.

    kwargs : Any
        Additional keyword arguments that will be passed to the backend's read_csv
        method.

    Returns
    -------
    Expr
        An expression representing the deferred read operation.
    """

    infer_schema = kwargs.pop("infer_schema", infer_csv_schema_pandas)
    deferred_read_csv.method_name = method_name = "read_csv"
    method = getattr(con, method_name)
    if table_name is None:
        table_name = gen_name(f"xorq-{method_name}")
    if schema is None:
        schema = infer_schema(path)
    if con.name == "pandas":
        # FIXME: determine how to best handle schema
        read_kwargs = make_read_kwargs(method, path, table_name, **kwargs)
    elif con.name == "duckdb":
        read_kwargs = make_read_kwargs(
            method, path, table_name, columns=schema, **kwargs
        )
    else:
        read_kwargs = make_read_kwargs(
            method, path, table_name, schema=schema, **kwargs
        )
    return Read(
        method_name=method_name,
        name=table_name,
        schema=schema,
        source=con,
        read_kwargs=read_kwargs,
        normalize_method=normalize_method,
    ).to_expr()


def deferred_read_parquet(
    con: Backend,
    path: str | Path,
    table_name: str | None = None,
    normalize_method: Callable = normalize_read_path_stat,
    **kwargs,
) -> ir.Table:
    """
     Create a deferred read operation for Parquet files that will execute only when needed.

    This function creates a representation of a read operation that doesn't immediately
    load data into memory. Instead, it registers the operation to be performed when
    the resulting expression is executed.

    Parameters
    ----------
    con : Backend
        The connection object representing the backend where the Parquet data will be read.

    path : str or Path
        The path to the Parquet file or directory to be read.

    table_name : str, optional
        The name to give to the resulting table in the backend. If not provided,
        a unique name will be generated automatically.

    **kwargs : dict
        Additional keyword arguments passed to the backend's read_parquet method.

     Returns
     -------
     Expr
         An expression representing the deferred read operation.
    """

    deferred_read_parquet.method_name = method_name = "read_parquet"
    method = getattr(con, method_name)
    if table_name is None:
        table_name = gen_name(f"letsql-{method_name}")
    schema = xo.connect().read_parquet(path).schema()
    read_kwargs = make_read_kwargs(method, path, table_name, **kwargs)
    return Read(
        method_name=method_name,
        name=table_name,
        schema=schema,
        source=con,
        read_kwargs=read_kwargs,
        normalize_method=normalize_method,
    ).to_expr()


def rbr_wrapper(reader, clean_up):
    def gen():
        yield from reader
        clean_up()

    return pa.RecordBatchReader.from_batches(reader.schema, gen())
