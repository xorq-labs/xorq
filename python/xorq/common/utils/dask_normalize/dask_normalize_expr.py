import itertools
import pathlib
import re
import types
import urllib.request

import dask
import sqlglot as sg
import yaml12

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


@dask.base.normalize_token.register(types.ModuleType)
def normalize_module(module):
    return normalize_seq_with_caller(
        module.__name__,
        module.__package__,
        caller="normalize_module",
    )


@dask.base.normalize_token.register(InputType)
def tokenize_input_type(obj):
    return normalize_seq_with_caller(
        obj.__class__.__module__, obj.__class__.__name__, obj.name, obj.value
    )


def _normalize_computed_kwargs_expr(cke):
    """Content-stable normalization of a computed_kwargs_expr.

    The default ``normalize_expr`` path generates SQL that includes
    session-dependent UDF class names (e.g. ``_inner_fit_0`` vs
    ``_inner_fit_3``).  These names contain a process-global counter that
    changes depending on how many UDFs were created before this expression
    was imported — making the token (and therefore the build hash)
    non-deterministic under parallel test execution or multi-module import.

    Instead of relying on the SQL string, we decompose the expression into
    components whose registered normalizers are already name-insensitive:

    * ``normalize_inmemorytable`` hashes pyarrow batch content
    * ``normalize_agg_udf`` / ``normalize_scalar_udf`` hash ``__func__``
      and arg types (excluding ``__func_name__``)
    * ``normalize_read`` / ``normalize_cached_node`` are stable
    """
    op = cke.op()
    mems = op.find(ir.InMemoryTable)
    agg_udfs = op.find(AggUDF)
    scalar_udfs = op.find(ScalarUDF)
    reads = op.find(rel.Read)
    cached = op.find(rel.CachedNode)
    return normalize_seq_with_caller(
        cke.schema() if isinstance(cke, ibis.expr.types.Table) else cke.type(),
        tuple(map(normalize_inmemorytable, mems)),
        agg_udfs,
        scalar_udfs,
        reads,
        cached,
        caller="normalize_computed_kwargs_expr",
    )


def opaque_node_replacer(node, kwargs):
    # FIXME: use xorq.common.utils.graph_utils.opaque_ops (includes ExprScalarUDF)
    opaque_ops = (
        rel.Read,
        rel.CachedNode,
        rel.RemoteTable,
        rel.FlightUDXF,
        rel.FlightExpr,
        # udf.ExprScalarUDF,
    )
    match node:
        case rel.CachedNode():
            new_node = api.table(
                node.schema,
                name=dask.base.tokenize(node.parent, node.cache),
            ).op()
        case rel.Read():
            new_node = api.table(
                node.schema,
                name=dask.base.tokenize(node),
            ).op()
        case rel.RemoteTable():
            new_node = api.table(
                node.schema,
                name=dask.base.tokenize(node.remote_expr),
            ).op()
        case rel.FlightUDXF() | rel.FlightExpr():
            new_node = api.table(
                node.schema,
                name=dask.base.tokenize(node),
            ).op()
        # # FIXME: what to do about ExprScalarUDF?
        # case udf.ExprScalarUDF():
        #     # ExprScalarUDF doesn't have a schema
        #     # it has computed_kwargs_expr and others
        #     node = xo.table(
        #         node.schema,
        #         name=dask.base.tokenize(node),
        #     ).op()
        case rel.HashingTag():
            new_node = api.table(
                node.schema,
                name=dask.base.tokenize(node.parent.to_expr(), node.metadata),
            ).op()
        case xops.NamedScalarParameter():
            new_node = (
                api.literal(value=None, type=node.dtype)
                .name(dask.base.tokenize(node))
                .op()
            )
        case _:
            if isinstance(node, opaque_ops):
                raise ValueError(f"unhandled opaque node type: {type(node)}")
            elif kwargs:
                new_node = node.__recreate__(kwargs)
            else:
                new_node = node
    return new_node


def normalize_op(op, compiler=None):
    sql = unbound_expr_to_default_sql(
        op.replace(opaque_node_replacer).to_expr().unbind(),
        compiler=compiler,
    )
    reads = op.find(rel.Read)
    dts = tuple(
        node
        for node in op.find(ir.DatabaseTable)
        if not isinstance(node, (rel.CachedNode, rel.Read))
    )
    udfs = op.find((AggUDF, ScalarUDF))
    mems = op.find(ir.InMemoryTable)
    named_params = op.find(xops.NamedScalarParameter)
    token = normalize_seq_with_caller(
        sql,
        reads,
        dts,
        udfs,
        tuple(map(normalize_inmemorytable, mems)),
        *(
            (tuple(map(dask.base.normalize_token, named_params)),)
            if named_params
            else ()
        ),
        caller="normalize_expr",
    )
    return token
