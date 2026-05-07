import contextvars
import functools
import hashlib
import itertools
import pathlib
import re
import types
import urllib.error
import urllib.request

import dask
import sqlglot as sg
import yaml12

import xorq.expr.datatypes as dat
import xorq.expr.operations as xops
import xorq.expr.relations as rel
import xorq.vendor.ibis.expr.operations.relations as ir
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_seq_with_caller,
)
from xorq.common.utils.defer_utils import normalize_read_path_stat
from xorq.expr import api
from xorq.vendor import ibis
from xorq.vendor.ibis.expr.operations.udf import (
    AggUDF,
    InputType,
    ScalarUDF,
)


# load_expr_from_zip picks a fresh tempfile.mkdtemp(prefix="xorq-catalog-") per
# load. Strip everything up to and including that segment so paths relative to
# the build dir (whose name is a content hash) are stable across reloads.
_CATALOG_EXTRACT_DIR_RE = re.compile(r".*?/xorq-catalog-[^/]+/")


def _canonicalize_plan_path(s):
    return _CATALOG_EXTRACT_DIR_RE.sub("", s)


def _extract_plan_file_paths(ep_str):
    """Extract path strings from a DataFusion execution plan ``file_groups`` section.

    Restores the leading "/" that DataFusion strips, then strips any
    xorq-catalog-* tempdir prefix so catalog-zip paths are stable across
    re-extractions.
    """
    file_groups_match = re.search(r"file_groups=(\{[^}]*\})", ep_str)
    if not file_groups_match:
        return ()
    parsed = yaml12.parse_yaml(file_groups_match.group(1))
    (groups,) = parsed.values()
    return tuple(
        _canonicalize_plan_path(_to_path_str(raw))
        for raw in itertools.chain.from_iterable(groups)
    )


def expr_is_bound(expr):
    backends, _ = expr._find_backends()
    return bool(backends)


def unbound_expr_to_default_sql(expr, compiler=None):
    if expr_is_bound(expr):
        raise ValueError("expr must be unbound, but is already bound to a backend")
    default_sql = api.to_sql(expr, compiler=compiler)
    return str(default_sql)


def normalize_inmemorytable(dt):
    if not isinstance(dt, ir.InMemoryTable):
        raise ValueError(f"expected InMemoryTable, got {type(dt)}")
    return normalize_seq_with_caller(
        dt.schema.to_pandas(),
        # in memory: so we can assume it's reasonable to hash the data
        tuple(
            dask.base.tokenize(el.serialize().to_pybytes())
            for el in dt.to_expr().to_pyarrow_batches()
        ),
        caller="normalize_inmemorytable",
    )


def normalize_memory_databasetable(dt):
    if dt.source.name not in (
        "pandas",
        "xorq_datafusion",
        "datafusion",
        "duckdb",
        "sqlite",
    ):
        raise ValueError(f"expected in-memory backend, got {dt.source.name!r}")
    return normalize_seq_with_caller(
        # we are normalizing the data, we don't care about the connection
        # dt.source,
        dt.schema.to_pandas(),
        # in memory: so we can assume it's reasonable to hash the data
        tuple(
            dask.base.tokenize(el.serialize().to_pybytes())
            for el in dt.to_expr().to_pyarrow_batches()
        ),
        caller="normalize_memory_databasetable",
    )


def normalize_pandas_databasetable(dt):
    if dt.source.name != "pandas":
        raise ValueError(f"expected pandas backend, got {dt.source.name!r}")
    return normalize_memory_databasetable(dt)


_REMOTE_SCHEMES = ("http://", "https://", "s3://", "gs://", "gcs://")


def _to_path_str(raw):
    """Normalize a raw path token: preserve remote URLs, fix relative local paths."""
    if raw.startswith(_REMOTE_SCHEMES):
        return raw
    p = pathlib.Path(raw)
    return str(p if p.is_absolute() else pathlib.Path("/") / raw)


def _normalize_path_stat(path, **kwargs):
    """Return a stable metadata tuple for any path: HTTP HEAD, cloud metadata, or local stat."""
    match path:
        case str() if path.startswith(("http://", "https://")):
            req = urllib.request.Request(
                path, method="HEAD", headers={"User-Agent": "xorq-cache"}
            )
            resp = urllib.request.urlopen(req, timeout=10)
            headers = resp.info()
            return (
                ("url", path),
                *(
                    (k, headers.get(k))
                    for k in ("Last-Modified", "Content-Length", "Content-Type")
                ),
            )
        case str() if path.startswith(("s3://", "gs://", "gcs://")):
            metadata = api.get_object_metadata(path, **kwargs)
            return tuple(
                (k, metadata.get(k))
                for k in ("location", "last_modified", "size", "e_tag", "version")
            )
        case str():
            p = pathlib.Path(path)
            if p.exists():
                return normalize_read_path_stat(p)
            raise FileNotFoundError(f"local path does not exist: {path!r}")
        case _:
            raise TypeError(f"expected str path, got {type(path).__name__}")


def _extract_duckdb_file_paths(sql_ddl):
    """Extract path strings from read_parquet/read_csv literals in a DuckDB DDL statement."""
    tree = sg.parse_one(sql_ddl, dialect="duckdb")
    parquet_paths = tuple(
        _to_path_str(lit.this)
        for func in tree.find_all(sg.exp.ReadParquet)
        for lit in func.find_all(sg.exp.Literal)
        if lit.is_string
    )
    # Search only func.this (the path argument), not func.find_all — func.expressions
    # holds keyword args like (header = CAST('t' AS BOOLEAN)) whose string literals
    # would otherwise be mistaken for file paths.
    csv_paths = tuple(
        _to_path_str(lit.this)
        for func in tree.find_all(sg.exp.ReadCSV)
        if func.this is not None
        for lit in func.this.find_all(sg.exp.Literal)
        if lit.is_string
    )
    return parquet_paths + csv_paths


def normalize_datafusion_databasetable(dt):
    if dt.source.name not in ("datafusion", "xorq_datafusion"):
        raise ValueError(f"expected datafusion/xorq backend, got {dt.source.name!r}")
    table = dt.source.con.table(dt.name)
    ep_str = str(table.execution_plan())
    if ep_str.startswith(("ParquetExec:", "CsvExec:")) or re.match(
        r"DataSourceExec:.+file_type=(csv|parquet)", ep_str
    ):
        paths = _extract_plan_file_paths(ep_str)
        if paths:
            file_metadata = tuple((p, _normalize_path_stat(p)) for p in sorted(paths))
            return normalize_seq_with_caller(
                dt.schema.to_pandas(),
                file_metadata,
            )
        else:
            raise ValueError(
                f"no parquet/csv paths extractable from execution plan: {ep_str!r}"
            )
    elif ep_str.startswith(("MemoryExec:", "DataSourceExec:")):
        return normalize_memory_databasetable(dt)
    elif "PyRecordBatchProviderExec" in ep_str:
        return normalize_seq_with_caller(
            dt.schema.to_pandas(),
            dt.name,
        )
    elif ep_str.startswith("EmptyExec"):
        raise ValueError("No data to cache")
    else:
        raise ValueError(f"unrecognized DataFusion execution plan: {ep_str!r}")


def normalize_remote_databasetable(dt):
    return normalize_seq_with_caller(
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        caller="normalize_remote_databasetable",
    )


def normalize_postgres_databasetable(dt):
    from xorq.common.utils.postgres_utils import (  # noqa: PLC0415
        get_postgres_n_reltuples,
    )

    if dt.source.name != "postgres":
        raise ValueError(f"expected postgres backend, got {dt.source.name!r}")
    return normalize_seq_with_caller(
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        get_postgres_n_reltuples(dt),
        caller="normalize_postgres_databasetable",
    )


def normalize_pyiceberg_database_table(dt):
    from xorq.common.utils.pyiceberg_utils import (  # noqa: PLC0415
        get_iceberg_snapshots_ids,
    )

    if dt.source.name != "pyiceberg":
        raise ValueError(f"expected pyiceberg backend, got {dt.source.name!r}")

    return normalize_seq_with_caller(
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        get_iceberg_snapshots_ids(dt),
        caller="normalize_pyiceberg_databasetable",
    )


def normalize_snowflake_databasetable(dt):
    from xorq.common.utils.snowflake_utils import (  # noqa: PLC0415
        get_snowflake_last_modification_time,
    )

    if dt.source.name != "snowflake":
        raise ValueError(f"expected snowflake backend, got {dt.source.name!r}")
    return normalize_seq_with_caller(
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        get_snowflake_last_modification_time(dt).tobytes(),
        caller="normalize_snowflake_databasetable",
    )


def normalize_bigquery_databasetable(dt):
    # https://stackoverflow.com/questions/44288261/get-the-last-modified-date-for-all-bigquery-tables-in-a-bigquery-project/44290543#44290543
    if dt.source.name != "bigquery":
        raise ValueError(f"expected bigquery backend, got {dt.source.name!r}")
    # https://stackoverflow.com/a/44290543
    query = f"""
    SELECT last_modified_time
    FROM `{dt.namespace.database}.__TABLES__` where table_id = '{dt.name}'
    """
    ((last_modified_time,),) = dt.source.raw_sql(query).to_dataframe()
    return normalize_seq_with_caller(
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        last_modified_time,
    )


def normalize_duckdb_databasetable(dt):
    if dt.source.name != "duckdb":
        raise ValueError(f"expected duckdb backend, got {dt.source.name!r}")
    name = sg.table(dt.name, quoted=dt.source.compiler.quoted).sql(
        dialect=dt.source.name
    )

    ((_, plan),) = dt.source.raw_sql(f"EXPLAIN SELECT * FROM {name}").fetchall()
    scan_line = plan.split("\n")[1]
    execution_plan_name = r"\s*│\s*(\w+)\s*│\s*"
    match re.match(execution_plan_name, scan_line).group(1):
        case "ARROW_SCAN" | "PANDAS_SCAN":
            return normalize_memory_databasetable(dt)
        case "READ_PARQUET" | "READ_CSV" | "SEQ_SCAN":
            return normalize_duckdb_file_read(dt)
        case _:
            raise NotImplementedError(scan_line)


def normalize_sqlite_database_table(dt):
    from xorq.common.utils.sqlite_utils import get_sqlite_stats  # noqa: PLC0415

    if dt.source.name != "sqlite":
        raise ValueError(f"expected sqlite backend, got {dt.source.name!r}")

    if dt.source.is_in_memory():
        return normalize_memory_databasetable(dt)
    else:
        return normalize_seq_with_caller(
            dt.name,
            dt.schema,
            dt.source,
            dt.namespace,
            get_sqlite_stats(dt),
            caller="normalize_sqlite_database_table",
        )

    pass


def normalize_duckdb_file_read(dt):
    name = sg.exp.convert(dt.name).sql(dialect=dt.source.name)
    (sql_ddl_statement,) = dt.source.con.sql(
        f"select sql from duckdb_views() where view_name = {name} UNION select sql from duckdb_tables() where table_name = {name}"
    ).fetchone()
    paths = _extract_duckdb_file_paths(sql_ddl_statement)
    if paths:
        file_metadata = tuple((p, _normalize_path_stat(p)) for p in sorted(paths))
        return normalize_seq_with_caller(
            dt.schema.to_pandas(),
            file_metadata,
        )
    else:
        # No read_parquet/read_csv paths found (e.g. plain CREATE TABLE) —
        # fall back to DDL-string-based token (not file-change-sensitive).
        return normalize_seq_with_caller(
            dt.schema.to_pandas(),
            sql_ddl_statement,
        )


def rename_unbound_static(op, prefix="static-name"):
    count = itertools.count()

    def rename_unbound(node, kwargs):
        if isinstance(node, ir.UnboundTable):
            name = f"{prefix}-{next(count)}"
            return node.copy(name=name)
        else:
            if kwargs:
                return node.__recreate__(kwargs)
            return node

    return op.replace(rename_unbound)


def normalize_xorq_databasetable(dt):
    if dt.source.name != "xorq_datafusion":
        raise ValueError(f"expected xorq backend, got {dt.source.name!r}")
    if isinstance(dt, rel.FlightExpr):
        return dask.base.normalize_token(
            (
                "normalize_xorq_databasetable",
                dt.input_expr,
                # we need to "stabilize" the name of the tables in the unbound expr
                rename_unbound_static(dt.unbound_expr.op()).to_expr(),
                dt.make_connection,
            )
        )
    elif isinstance(dt, rel.FlightUDXF):
        return dask.base.normalize_token(
            (
                "normalize_xorq_databasetable",
                dt.input_expr,
                dt.udxf.exchange_f,
                dt.make_connection,
            )
        )
    return normalize_datafusion_databasetable(dt)


@dask.base.normalize_token.register(types.ModuleType)
def normalize_module(module):
    return normalize_seq_with_caller(
        module.__name__,
        module.__package__,
        caller="normalize_module",
    )


@dask.base.normalize_token.register(dat.DataType)
def normalize_ibis_datatype(datatype):
    return normalize_seq_with_caller(datatype.name.lower(), *datatype.args)


@dask.base.normalize_token.register(rel.Read)
def normalize_read(read):
    read_kwargs = dict(read.read_kwargs)
    path = read_kwargs["hash_path"]
    if isinstance(path, (list, tuple)):
        # normalize_filenames may have converted a single path to a list
        path = path[0] if len(path) == 1 else path
    if isinstance(path, (str, pathlib.Path)):
        path = str(path)
        if path.startswith(("http://", "https://")):
            tpls = _normalize_path_stat(path)
        elif path.startswith(("s3://", "gs://", "gcs://")):
            tpls = _normalize_path_stat(
                path, **{k: v for k, v in read_kwargs.items() if k != "hash_path"}
            )
        elif not pathlib.Path(path).is_absolute() and path == read_kwargs.get(
            "read_path"
        ):
            # Build-bundled Read whose hash_path was rewritten to the relative
            # read_path for deterministic YAML (see common.py register_node).
            # The filename is already a content hash, so use it directly.
            tpls = (("build-relative-path", path),)
        elif (path := pathlib.Path(path)).exists():
            tpls = read.normalize_method(path)
        else:
            raise NotImplementedError(f'Don\'t know how to deal with path "{path}"')
    elif isinstance(path, (list, tuple)) and all(isinstance(el, str) for el in path):
        raise NotImplementedError(f'Don\'t know how to deal with path "{path}"')
    else:
        raise NotImplementedError(f'Don\'t know how to deal with path "{path}"')
    tpls += tuple(
        (k, v)
        for k, v in read.read_kwargs
        if k
        in (
            "mode",
            "schema",
            "temporary",
        )
    )
    return normalize_seq_with_caller(
        read.schema,
        tpls,
        caller="normalize_read",
    )


@dask.base.normalize_token.register(ir.DatabaseTable)
def normalize_databasetable(dt):
    dct = {
        "pandas": normalize_pandas_databasetable,
        "datafusion": normalize_datafusion_databasetable,
        "postgres": normalize_postgres_databasetable,
        "snowflake": normalize_snowflake_databasetable,
        "xorq_datafusion": normalize_xorq_databasetable,
        "duckdb": normalize_duckdb_databasetable,
        "trino": normalize_remote_databasetable,
        "gizmosql": normalize_remote_databasetable,
        "bigquery": normalize_bigquery_databasetable,
        "pyiceberg": normalize_pyiceberg_database_table,
        "sqlite": normalize_sqlite_database_table,
    }
    f = dct[dt.source.name]
    return f(dt)


@dask.base.normalize_token.register(rel.RemoteTable)
def normalize_remote_table(dt):
    if not isinstance(dt, rel.RemoteTable):
        raise ValueError(f"expected RemoteTable, got {type(dt)}")

    return normalize_seq_with_caller(
        ("schema", dt.schema),
        ("expr", dt.remote_expr),
        # only thing that matters is the type of source its going into
        ("source", dt.source.name),
        caller="normalize_remote_table",
    )


@dask.base.normalize_token.register(rel.CachedNode)
def normalize_cached_node(node):
    return normalize_seq_with_caller(
        node.parent,
        node.cache,
        caller="normalize_cached_node",
    )


@dask.base.normalize_token.register(xops.NamedScalarParameter)
def normalize_named_scalar_parameter(node):
    return normalize_seq_with_caller(
        node.label,
        node.dtype,
        node.default,
        caller="normalize_named_scalar_parameter",
    )


@dask.base.normalize_token.register(ibis.backends.BaseBackend)
def normalize_backend(con):
    name = con.name
    if name == "snowflake":
        con_details = con.con._host
    elif name == "postgres":
        con_dct = con.con.info.get_parameters() | {"port": con.con.info.port}
        con_details = {k: con_dct[k] for k in ("host", "port", "dbname")}
    elif name == "pandas":
        con_details = id(con.dictionary)
    elif name in ("datafusion", "duckdb", "xorq_datafusion", "gizmosql"):
        con_details = (con._profile.con_name, con._profile.kwargs_tuple)
    elif name == "trino":
        con_details = con.con.host
    elif name == "bigquery":
        con_details = (
            con.project_id,
            con.dataset_id,
        )
    elif name == "pyiceberg":
        catalog_params = con.catalog_params
        con_details = (con.catalog.name,) + tuple(
            catalog_params[k] for k in ("type", "uri", "warehouse")
        )
    elif name == "sqlite":
        return id(con.con) if con.is_in_memory() else con.uri
    elif name == "databricks":
        con_details = (con._server_hostname, con._http_path)
    else:
        raise ValueError(f"no normalization rule for backend {name!r}")
    return normalize_seq_with_caller(
        name,
        con_details,
    )


@dask.base.normalize_token.register(ir.Schema)
def normalize_schema(schema):
    return normalize_seq_with_caller(
        schema.to_pandas(),
    )


@dask.base.normalize_token.register(ir.Namespace)
def normalize_namespace(ns):
    return normalize_seq_with_caller(
        ns.catalog,
        ns.database,
    )


@dask.base.normalize_token.register(InputType)
def tokenize_input_type(obj):
    return normalize_seq_with_caller(
        obj.__class__.__module__, obj.__class__.__name__, obj.name, obj.value
    )


def _normalize_computed_kwargs_expr(cke):
    """Content-stable, *structural-only* normalization of a computed_kwargs_expr.

    The default ``normalize_expr`` path generates SQL that includes
    session-dependent UDF class names (e.g. ``_inner_fit_0`` vs
    ``_inner_fit_3``) and is therefore non-deterministic across imports.  We
    avoid SQL by decomposing the expression into components whose registered
    normalizers are already name-insensitive.

    Per ADR-0010, this function must be **data-free** — the structural hash
    must not change when data inside the cke changes.  Data identity for any
    leaf reachable through ``ExprScalarUDF.computed_kwargs_expr`` is collected
    by the outer ``walk_nodes`` traversal in :func:`normalize_op_split` and
    appears in the outer ``data_deps``; here we contribute only structural
    shape (schemas, op types, UDF function identity, cache class).

    * ``InMemoryTable`` -> hashed by schema, **not** batch bytes.
    * ``Read`` -> hashed by schema + ``method_name`` + ``read_kwargs`` minus
      path identity (``hash_path``, ``read_path``); the path's data identity
      lives in outer ``data_deps``.
    * ``CachedNode`` -> hashed by schema + cache class name; the parent's
      data identity lives in outer ``data_deps``, and the parent's structure
      is captured via the agg/scalar UDFs hashed below.
    * ``AggUDF`` / ``ScalarUDF`` — hashed via their registered handlers,
      which key on ``__func__`` + arg dtypes (excluding the counter-suffixed
      ``__func_name__``).  ``ScalarUDF`` recurses through this function, so
      the structural-only contract holds transitively.
    """
    op = cke.op()
    mems = op.find(ir.InMemoryTable)
    agg_udfs = op.find(AggUDF)
    scalar_udfs = op.find(ScalarUDF)
    reads = op.find(rel.Read)
    cached = op.find(rel.CachedNode)
    # Strip path identity from read_kwargs, the path's data lives in outer
    # data_deps via the registered ``normalize_read`` on the leaf.
    _path_keys = ("hash_path", "read_path")
    read_structural = tuple(
        (
            r.schema,
            r.method_name,
            tuple((k, v) for k, v in r.read_kwargs if k not in _path_keys),
            r.normalize_method,
        )
        for r in reads
    )
    return normalize_seq_with_caller(
        cke.schema() if isinstance(cke, ibis.expr.types.Table) else cke.type(),
        tuple(m.schema for m in mems),
        agg_udfs,
        scalar_udfs,
        read_structural,
        tuple((c.schema, type(c.cache).__name__) for c in cached),
        caller="normalize_computed_kwargs_expr",
    )


@dask.base.normalize_token.register(ScalarUDF)
def normalize_scalar_udf(udf):
    typs = tuple(arg.dtype for arg in udf.args)
    computed_kwargs_expr = udf.__config__.get("computed_kwargs_expr")
    # Normalize computed_kwargs_expr via content-stable decomposition
    # rather than the default normalize_expr -> normalize_op -> SQL path,
    # which includes session-dependent UDF class names.
    if computed_kwargs_expr is not None:
        computed_kwargs_token = _normalize_computed_kwargs_expr(computed_kwargs_expr)
    else:
        computed_kwargs_token = None
    return normalize_seq_with_caller(
        ScalarUDF,
        typs,
        udf.dtype,
        udf.__func__,
        #
        # ExprScalarUDF
        computed_kwargs_token,
        # we are insensitive to these for now
        # udf.__udf_namespace__,
        # udf.__func_name__,
        caller="normalize_scalar_udf",
    )


@dask.base.normalize_token.register(AggUDF)
def normalize_agg_udf(udf):
    *args, where = udf.args
    if where is not None:
        # TODO: determine if sql string already contains
        #       the relevant information of `where`
        raise NotImplementedError
    typs = tuple(arg.dtype for arg in args)
    return normalize_seq_with_caller(
        AggUDF,
        typs,
        udf.dtype,
        udf.__func__,
        # we are insensitive to these for now
        # udf.__udf_namespace__,
        # udf.__func_name__,
        caller="normalize_agg_udf",
    )


# Leaf op types whose data identity contributes to ``data_deps``.  The list
# crosses opaque sub-expression boundaries (RemoteTable / CachedNode /
# FlightExpr / FlightUDXF / ExprScalarUDF) — see :func:`normalize_op_split`.
_LEAF_TYPES = (ir.DatabaseTable, ir.InMemoryTable, rel.Read)


def _normalize_data_leaf(node):
    """Compute the data-deps token for one leaf.

    ``InMemoryTable`` has no registered ``dask.base.normalize_token`` handler,
    so we route through :func:`normalize_inmemorytable` explicitly.  Other leaf
    types (DatabaseTable, Read) have registered handlers.
    """
    if isinstance(node, ir.InMemoryTable):
        return normalize_inmemorytable(node)
    return dask.base.normalize_token(node)


def _is_data_leaf(node):
    """Return True for leaves that contribute to ``data_deps``.

    Strictly excludes ``DatabaseTable`` subclasses (``CachedNode``, ``RemoteTable``,
    ``FlightExpr``, ``FlightUDXF``) — those are structural carriers handled by
    the opaque-replacer below, not data leaves.
    """
    if type(node) is ir.DatabaseTable:
        return True
    if isinstance(node, ir.InMemoryTable):
        return True
    if isinstance(node, rel.Read):
        return True
    return False


# Per-call memo for ``_opaque_structural_name``.  Without this, a deep pipeline
# whose opaque sub-expressions share roots (e.g. a chain of ``ExprScalarUDF``
# steps each referencing an accumulated ``computed_kwargs_expr``) recomputes the
# inner structural token once per opaque reference — O(depth²).  The cache is
# scoped per outer ``normalize_op_split`` call via ``contextvars`` so it doesn't
# leak across invocations or threads.
_opaque_structural_memo: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "xorq_opaque_structural_memo", default=None
)


def _opaque_structural_name(sub_expr):
    """Compute a data-free token for an opaque sub-expression using *its own* backend's compiler.

    The sub-expression's leaves still appear in the *outer* ``data_deps`` (because
    :func:`walk_nodes` descended through the opaque boundary to find them); here
    we only need its structural shape for use as a placeholder name in the outer
    structural rewrite, so the leaf data must not contribute again.

    Why the sub-expression's own compiler:  after the outer rewrite has unbound
    inner ``DatabaseTable`` leaves, ``get_compiler`` on the modified sub-expression
    falls back to the default backend, which fails on backend-specific ops
    (e.g. DuckDB's ``ArrayFilter`` under the DataFusion default compiler).  We
    capture the original compiler before any rewrite happens.
    """
    from xorq.expr.api import get_compiler  # noqa: PLC0415

    sub_op = sub_expr.op() if hasattr(sub_expr, "op") else sub_expr
    memo = _opaque_structural_memo.get()
    if memo is not None and sub_op in memo:
        return memo[sub_op]

    sub_compiler = get_compiler(sub_expr)
    structural = _normalize_structural(sub_op, compiler=sub_compiler)
    name = dask.base.tokenize(structural)
    if memo is not None:
        memo[sub_op] = name
    return name


@functools.cache
def _opaque_handled_here():
    from xorq.common.utils.graph_utils import opaque_ops  # noqa: PLC0415
    from xorq.expr.udf import ExprScalarUDF  # noqa: PLC0415

    return tuple(op for op in opaque_ops if op is not ExprScalarUDF)


def opaque_node_replacer(node, kwargs):
    """Replace opaque / data-bearing ops with data-free placeholders for structural compilation.

    Opaque sub-expressions (``RemoteTable``, ``CachedNode``, ``FlightExpr``,
    ``FlightUDXF``) are placeholdered by the *structural* token of their inner
    expression, computed with that expression's own backend compiler — never
    by ``dask.base.tokenize(sub_expr)`` directly, which would fold the
    sub-expression's data into the outer structural hash.

    Data leaves (``DatabaseTable``, ``InMemoryTable``, ``Read``) become
    ``UnboundTable`` placeholders.  Their data identity is captured separately
    in the outer ``data_deps`` collection.
    """
    match node:
        case rel.CachedNode():
            new_node = api.table(
                node.schema,
                name=dask.base.tokenize(
                    "cached_node",
                    _opaque_structural_name(node.parent),
                    node.cache,
                ),
            ).op()
        case rel.RemoteTable():
            new_node = api.table(
                node.schema,
                name=_opaque_structural_name(node.remote_expr),
            ).op()
        case rel.FlightUDXF() | rel.FlightExpr():
            new_node = api.table(
                node.schema,
                name=dask.base.tokenize(
                    type(node).__name__,
                    _opaque_structural_name(node.input_expr),
                ),
            ).op()
        case rel.Read():
            # Read is a data leaf (path / read_kwargs are data identity); the
            # structural placeholder is schema-derived only.
            new_node = api.table(
                node.schema,
                name=dask.base.tokenize("Read", node.schema),
            ).op()
        case rel.HashingTag():
            new_node = api.table(
                node.schema,
                name=dask.base.tokenize(
                    "hashing_tag",
                    _opaque_structural_name(node.parent.to_expr()),
                    node.metadata,
                ),
            ).op()
        case xops.NamedScalarParameter():
            new_node = (
                api.literal(value=None, type=node.dtype)
                .name(dask.base.tokenize(node))
                .op()
            )
        case _:
            if _is_data_leaf(node):
                new_node = ir.UnboundTable(name=node.name, schema=node.schema)
            else:
                # Loud-fail on unhandled opaque op types so a future addition
                # (e.g. a new Flight variant) cannot silently lose its data
                # dependency by falling through to the generic ``__recreate__``
                # path.  ``ExprScalarUDF`` is intentionally excluded — its
                # ``computed_kwargs_expr`` leaves are already collected by
                # ``walk_nodes``, and its own ``__dask_tokenize__`` handler
                # covers the structural side.
                if isinstance(node, _opaque_handled_here()):
                    raise ValueError(f"unhandled opaque node type: {type(node)}")
                if kwargs:
                    new_node = node.__recreate__(kwargs)
                else:
                    new_node = node
    return new_node


@dask.base.normalize_token.register(ibis.expr.types.Expr)
def normalize_expr(expr):
    """Return ``(slot_hashes, structural_hash)`` for ``dask.base.tokenize(expr)``.

    Both elements are hex strings.  Pre-hashing here means the final token
    reduces to ``md5(str((tuple_of_hex, hex)))``, which external callers can
    reproduce from a serialized metadata artifact with only ``hashlib``,
    see :func:`expr_metadata` and :func:`compute_expr_token`.
    """
    from xorq.expr.api import get_compiler  # noqa: PLC0415

    data_deps, structural = normalize_op(expr.op(), compiler=get_compiler(expr))
    return (
        tuple(dask.base.tokenize(d) for d in data_deps),
        dask.base.tokenize(structural),
    )


def normalize_op_split(op_or_expr, compiler=None):
    """Split an op's normalization into ``(leaf_dts, data_deps, structural)``.

    ``leaf_dts``    — data leaves in walk order: plain ``DatabaseTable``,
                       ``InMemoryTable``, and ``Read``.  Reachable through opaque
                       sub-expressions (``RemoteTable.remote_expr``,
                       ``CachedNode.parent``, ``FlightExpr/UDXF.input_expr``,
                       ``ExprScalarUDF.computed_kwargs_expr``) — i.e. cross-engine
                       and ML-pipeline dependencies appear as leaves so a swap of
                       e.g. a training table invalidates the cache.
    ``data_deps``   — tuple of normalized tokens, one per leaf in ``leaf_dts``.
    ``structural``  — normalized token for the data-free expression shape.
                       Opaque sub-expressions are placeholdered by their *own*
                       structural token (computed with their own backend's
                       compiler) so their data does not leak into the outer
                       structural hash.
    """
    from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415
    from xorq.expr.api import get_compiler  # noqa: PLC0415

    if hasattr(op_or_expr, "op"):
        op = op_or_expr.op()
        if compiler is None:
            compiler = get_compiler(op_or_expr)
    else:
        op = op_or_expr

    leaf_dts = tuple(n for n in walk_nodes(_LEAF_TYPES, op) if _is_data_leaf(n))
    data_deps = tuple(_normalize_data_leaf(dt) for dt in leaf_dts)

    # Install a per-call memo for ``_opaque_structural_name``.  Reentrant
    # callers (the recursion through inner opaque sub-expressions) inherit the
    # outer memo; only the outermost call seeds it.
    if _opaque_structural_memo.get() is None:
        token = _opaque_structural_memo.set({})
        try:
            structural = _normalize_structural(op, compiler=compiler)
        finally:
            _opaque_structural_memo.reset(token)
    else:
        structural = _normalize_structural(op, compiler=compiler)
    return leaf_dts, data_deps, structural


def normalize_op(op, compiler=None):
    """Return ``(data_deps, structural)`` for an op tree.

    ``data_deps`` is a tuple of per-leaf-data-table normalized tokens;
    ``structural`` is the normalized token for the data-free expression shape.
    """
    _, data_deps, structural = normalize_op_split(op, compiler=compiler)
    return data_deps, structural


def _normalize_structural(op, compiler=None):
    """Compute the structural (data-free) token for an op tree.

    Single-pass rewrite via :func:`replace_nodes` (descends through opaque
    sub-expressions so leaves inside them get unbound), then SQL-compile the
    result with the outer compiler.  Per-opaque-op compiler preservation is
    handled by :func:`opaque_node_replacer` calling
    :func:`_opaque_structural_name` *on the original sub-expression*.
    """
    from xorq.common.utils.graph_utils import (  # noqa: PLC0415
        replace_nodes,
    )

    rewritten = replace_nodes(opaque_node_replacer, op)
    sql = unbound_expr_to_default_sql(
        rewritten.to_expr().unbind(),
        compiler=compiler,
    )
    udfs = op.find((AggUDF, ScalarUDF))
    named_params = op.find(xops.NamedScalarParameter)
    return normalize_seq_with_caller(
        sql,
        udfs,
        *(
            (tuple(map(normalize_named_scalar_parameter, named_params)),)
            if named_params
            else ()
        ),
        caller="normalize_expr_structural",
    )


def expr_metadata(expr):
    """Produce a serializable metadata dict for cross-language token computation.

    The returned dict has the form::

        {
          "version": 2,
          "structural_hash": "<md5 hex>",
          "slots": [
              {"index": 0, "name": <leaf name>, "hash": "<md5 hex>"},
              ...,
          ],
        }

    ``index`` is the unambiguous slot key.  ``name`` is a human-readable hint
    (a leaf table name, file path, etc.) but may collide across slots — two
    ``DatabaseTable`` ops can share a name across backends, two ``InMemoryTable``
    ops can share a generated name, etc.  Cheap-substitution callers should
    look up by ``index``.

    The expression token can be recomputed from this dict (plus updated slot
    hashes) using :func:`compute_expr_token`, which only needs ``hashlib``.
    """
    leaf_dts, data_deps, structural = normalize_op_split(expr)
    return {
        "version": 2,
        "structural_hash": dask.base.tokenize(structural),
        "slots": [
            {
                "index": i,
                "name": getattr(dt, "name", None) or "",
                "hash": dask.base.tokenize(dep),
            }
            for i, (dt, dep) in enumerate(zip(leaf_dts, data_deps))
        ],
    }


def compute_expr_token(data_dep_hashes, structural_hash):
    """Compute an expression token from hex slot hashes plus a structural hex hash.

    Reproduces ``dask.base.tokenize(expr)`` when the registered ``Expr``
    normalizer returns ``(tuple_of_hex_slot_hashes, hex_structural_hash)``.
    Dask wraps the registered handler's return in an outer single-tuple via
    ``_normalize_seq_func``; the preimage therefore is::

        str(((tuple(data_dep_hashes), structural_hash),))

    Hex strings are identity-normalized by dask, so the formula reduces to a
    single ``hashlib.md5`` call — no xorq, dask, or ibis import required.
    """
    preimage = str(((tuple(data_dep_hashes), structural_hash),))
    return hashlib.md5(preimage.encode(), usedforsecurity=False).hexdigest()
