import dask
import ibis
import ibis.expr.operations.relations as ir
from letsql.expr.relations import (
    make_native_op,
)


def expr_is_bound(expr):
    backends, _ = expr._find_backends()
    return bool(backends)


def unbound_expr_to_default_sql(expr):
    if expr_is_bound(expr):
        raise ValueError
    default_sql = ibis.to_sql(
        expr,
        dialect=ibis.options.sql.default_dialect,
    )
    return str(default_sql)


def normalize_memory_databasetable(dt):
    if dt.source.name not in ("pandas", "datafusion"):
        raise ValueError
    return dask.base._normalize_seq_func(
        (
            dt.source,
            dt.schema.to_pandas(),
            # in memory: so we can assume its reasonable to hash the data
            dt.to_expr().to_pandas(),
        )
    )


def normalize_pandas_databasetable(dt):
    if dt.source.name != "pandas":
        raise ValueError
    return normalize_memory_databasetable(dt)


def normalize_datafusion_databasetable(dt):
    if dt.source.name not in ("datafusion", "let"):
        raise ValueError
    ep_str = str(dt.source.con.table(dt.name).execution_plan())
    if ep_str.startswith(("ParquetExec:", "CsvExec:")):
        return dask.base._normalize_seq_func(
            (
                dt.schema.to_pandas(),
                # ep_str denotes the parquet files to be read
                # FIXME: md5sum on detected .parquet files?
                ep_str,
            )
        )
    elif ep_str.startswith("MemoryExec:"):
        return normalize_memory_databasetable(dt)
    else:
        raise ValueError


def normalize_remote_databasetable(dt):
    if dt.source.name not in ("postgres", "snowflake"):
        raise ValueError
    return dask.base._normalize_seq_func(
        (
            dt.name,
            dt.schema,
            dt.source,
            dt.namespace,
        )
    )


def normalize_letsql_databasetable(dt):
    if dt.source.name != "let":
        raise ValueError
    native_source = dt.source._sources.get_backend(dt)
    if native_source.name == "let":
        return normalize_datafusion_databasetable(dt)
    new_dt = make_native_op(dt)
    return dask.base.normalize_token(new_dt)


@dask.base.normalize_token.register(ir.DatabaseTable)
def normalize_databasetable(dt):
    dct = {
        "pandas": normalize_pandas_databasetable,
        "datafusion": normalize_datafusion_databasetable,
        "postgres": normalize_remote_databasetable,
        "snowflake": normalize_remote_databasetable,
        "let": normalize_letsql_databasetable,
    }
    f = dct[dt.source.name]
    return f(dt)


@dask.base.normalize_token.register(ibis.backends.BaseBackend)
def normalize_backend(con):
    name = con.name
    if name in ("snowflake", "pandas", "datafusion"):
        return name
    elif name == "postgres":
        con_dct = con.con.get_dsn_parameters()
        return {k: con_dct[k] for k in ("host", "port", "dbname")}
    else:
        raise ValueError


@dask.base.normalize_token.register(ir.Schema)
def normalize_schema(schema):
    return dask.base._normalize_seq_func((schema.to_pandas(),))


@dask.base.normalize_token.register(ir.Namespace)
def normalize_namespace(ns):
    return dask.base._normalize_seq_func(
        (
            ns.catalog,
            ns.database,
        )
    )


@dask.base.normalize_token.register(ibis.expr.types.Expr)
def normalize_expr(expr):
    # how do cached tables interact with this?
    sql = unbound_expr_to_default_sql(expr.unbind())
    if not expr_is_bound(expr):
        return sql
    mem_dts = expr.op().find(ir.InMemoryTable)
    if mem_dts:
        # FIXME: decide whether to hash these
        # these could be large tables in memory in a remote postgres database, so possibly non-trivial to pull down and hash
        raise ValueError
    dts = expr.op().find(ir.DatabaseTable)
    return dask.base._normalize_seq_func(
        (
            sql,
            dts,
        )
    )


# @dask.base.normalize_token.register(ibis.common.graph.Node)
# def normalize_node(node):
#     return tuple(
#         map(
#             dask.base.normalize_token,
#             (
#                 type(node),
#                 *(node.args),
#             ),
#         )
#     )
