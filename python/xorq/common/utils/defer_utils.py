from __future__ import annotations

import itertools
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import toolz

import xorq.vendor.ibis.expr.types as ir
from xorq.common.utils.file_utils import (
    normalize_read_path_md5sum,
    normalize_read_path_stat,
)
from xorq.common.utils.inspect_utils import (
    get_arguments,
)
from xorq.config import default_backend
from xorq.expr.relations import (
    Read,
)
from xorq.vendor import ibis
from xorq.vendor.ibis import Schema
from xorq.vendor.ibis.util import (
    gen_name,
    normalize_filenames,
)


if TYPE_CHECKING:
    from xorq.backends.xorq_datafusion import Backend


DEFAULT_CHUNKSIZE = 10_000

# Backends whose read_parquet/read_csv route through ADBC read_record_batches
# (so they accept ``mode="replace"`` to avoid "relation already exists" errors).
# Snowflake is excluded: its read_record_batches is ADBC, but read_parquet/read_csv
# are native (kwargs become FILE_FORMAT options, where ``mode`` is invalid).
_ADBC_BACKENDS = frozenset(("sqlite", "postgres", "databricks"))

# Backend-specific parameter names for the file path argument.
_PATH_PARAM_NAMES = frozenset(("path", "paths", "source", "source_list"))


def make_read_kwargs(f, *args, **kwargs):
    # FIXME: if any kwarg is a dictionary, we'll fail Concrete's hashable requirement, so just pickle
    read_kwargs = get_arguments(f, *args, **kwargs)
    kwargs = read_kwargs.pop("kwargs", {})
    # Normalize backend-specific path parameter names to "hash_path" so that
    # Read nodes are portable across backends.
    read_kwargs = {
        ("hash_path" if k in _PATH_PARAM_NAMES else k): v
        for k, v in read_kwargs.items()
    }
    tpl = tuple(read_kwargs.items()) + tuple(kwargs.items())
    return tpl


def relocatable_read_path(path: str | Path) -> tuple[str, str]:
    """Path parts (``(<dir>, <filename>)``) of a bundled relocatable read.

    Single source of truth for the on-disk layout of a relocated read: the
    directory comes from ``BundledSourceTypes.read`` and the filename is the
    content hash of the source. Both the pre-hash pass
    (``_prepare_relocatable_reads``) and the write pass
    (``ExprDumper._prepare_relocatable_read``) derive the serialized
    ``read_path`` from this, so they cannot drift.
    """
    from xorq.common.utils.dasher import tokenize  # noqa: PLC0415
    from xorq.ibis_yaml.enums import BundledSourceTypes  # noqa: PLC0415

    path = Path(path)
    return (
        BundledSourceTypes.read.value,
        f"{tokenize(normalize_read_path_md5sum(path))}{path.suffix}",
    )


def relocatable_read_path_str(path: str | Path) -> str:
    """Serialized ``read_path`` of a bundled relocatable read (``"reads/<hash>.ext"``).

    Single source of the *joined* form of :func:`relocatable_read_path`, shared by
    the pre-hash pass (``_ensure_relocatable_read_paths``) and the write pass
    (``ExprDumper._prepare_relocatable_read``) so the ``read_path`` string cannot
    drift between them -- byte-equality of the two is exactly what makes a
    relocated build load+rebuild hash-stable.
    """
    return "/".join(relocatable_read_path(path))


@toolz.curry
def infer_csv_schema_pandas(path, chunksize=DEFAULT_CHUNKSIZE, **kwargs):
    import pandas as pd  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415

    path = normalize_filenames(path)
    gen = pd.read_csv(path[0], chunksize=chunksize, **kwargs)
    df = next(gen)
    batch = pa.RecordBatch.from_pandas(df, preserve_index=False)
    schema = ibis.Schema.from_pyarrow(batch.schema)
    return schema


def read_csv_rbr(*args, schema=None, chunksize=DEFAULT_CHUNKSIZE, dtype=None, **kwargs):
    """Deferred and streaming csv reading via pandas"""
    import pandas as pd  # noqa: PLC0415
    import pyarrow as pa  # noqa: PLC0415

    if dtype is not None:
        raise TypeError("pass `dtype` as pyarrow `schema`")
    if chunksize is None:
        raise ValueError("chunksize must not be `None`")
    if schema is not None:
        schema = ibis.schema(schema)
        dtype = {col: typ.to_pandas() for col, typ in schema.items()}
        schema = schema.to_pyarrow()
    # schema is always nullable (this is good)
    paths = normalize_filenames(*args)

    gen = map(
        partial(pa.RecordBatch.from_pandas, preserve_index=False),
        itertools.chain.from_iterable(
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
        el, gen = toolz.peek(gen)
        schema = el.schema

    def cast_gen():
        yield from (batch.cast(schema) for batch in gen)

    rbr = pa.RecordBatchReader.from_batches(
        schema,
        cast_gen(),
    )
    return rbr


def deferred_read_csv(
    path: str | Path,
    con: Backend | None = None,
    table_name: str | None = None,
    schema: Schema | None = None,
    normalize_method: Callable = normalize_read_path_stat,
    relocatable: bool = False,
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
    path : str or Path
        The path to the CSV file to be read. This can be a local file path or a URL.

    con : Backend, optional
        The connection object representing the backend where the CSV will be read.
        This can be any backend that supports reading CSV files (pandas, duckdb,
        postgres, etc.).

    table_name : str, optional
        The name to give to the resulting table in the backend. If not provided,
        a unique name will be generated automatically.

    schema : Schema, optional
        The schema definition for the CSV data. If not provided, the schema will
        be inferred from the data by sampling the CSV file.

    relocatable : bool, optional
        When True, ``xorq build`` will copy the backing file into the build
        artifact and rewrite the path so the archive is self-contained.

    kwargs : Any
        Additional keyword arguments that will be passed to the backend's read_csv
        method.

    Returns
    -------
    Expr
        An expression representing the deferred read operation.
    """

    infer_schema = kwargs.pop("infer_schema", infer_csv_schema_pandas)
    method_name = "read_csv"

    if con is None:
        con = default_backend()

    method = getattr(con, method_name)

    if table_name is None:
        table_name = gen_name(f"xorq-{method_name}")
    if schema is None:
        schema = infer_schema(path)
    if con.name in _ADBC_BACKENDS:
        kwargs.setdefault("mode", "replace")
    if con.name == "pandas":
        # FIXME: determine how to best handle schema
        read_kwargs = make_read_kwargs(method, path, table_name, **kwargs)
    elif con.name == "duckdb":
        read_kwargs = make_read_kwargs(
            method, path, table_name=table_name, columns=schema, **kwargs
        )
    else:
        read_kwargs = make_read_kwargs(
            method, path, table_name, schema=schema, **kwargs
        )
    if relocatable:
        read_kwargs = read_kwargs + (("relocatable", True),)
        normalize_method = normalize_read_path_md5sum
    return Read(
        method_name=method_name,
        name=table_name,
        schema=schema,
        source=con,
        read_kwargs=read_kwargs,
        normalize_method=normalize_method,
    ).to_expr()


def deferred_read_parquet(
    path: str | Path,
    con: Backend | None = None,
    table_name: str | None = None,
    schema: Schema | None = None,
    normalize_method: Callable = normalize_read_path_stat,
    relocatable: bool = False,
    **kwargs,
) -> ir.Table:
    """
     Create a deferred read operation for Parquet files that will execute only when needed.

    This function creates a representation of a read operation that doesn't immediately
    load data into memory. Instead, it registers the operation to be performed when
    the resulting expression is executed.

    Parameters
    ----------
    path : str or Path
        The path to the Parquet file or directory to be read.

    con : Backend, optional
        The connection object representing the backend where the Parquet data will be read.

    table_name : str, optional
        The name to give to the resulting table in the backend. If not provided,
        a unique name will be generated automatically.

    normalize_method : Callable, optional
        The method that returns the values to be used in the hashing of the Read operation.

    relocatable : bool, optional
        When True, ``xorq build`` will copy the backing file into the build
        artifact and rewrite the path so the archive is self-contained.

    **kwargs : dict
        Additional keyword arguments passed to the backend's read_parquet method.

    Returns
    -------
    Expr
        An expression representing the deferred read operation.
    """

    method_name = "read_parquet"
    if con is None:
        con = default_backend()
    method = getattr(con, method_name)
    if table_name is None:
        table_name = gen_name(f"xorq-{method_name}")
    if not schema:
        from xorq.backends.xorq_datafusion import connect  # noqa: PLC0415

        schema = schema or connect().read_parquet(path).schema()
    if con.name in _ADBC_BACKENDS:
        kwargs.setdefault("mode", "replace")
    read_kwargs = make_read_kwargs(method, path, table_name=table_name, **kwargs)
    if relocatable:
        read_kwargs = read_kwargs + (("relocatable", True),)
        normalize_method = normalize_read_path_md5sum
    return Read(
        method_name=method_name,
        name=table_name,
        schema=schema,
        source=con,
        read_kwargs=read_kwargs,
        normalize_method=normalize_method,
    ).to_expr()
