import itertools
import pathlib
import re
import types
import urllib.error
import urllib.request

import dask
import sqlglot as sg

import xorq as xo
import xorq.expr.datatypes as dat
import xorq.expr.relations as rel
import xorq.vendor.ibis.expr.operations.relations as ir
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_seq_with_caller,
)
from xorq.vendor import ibis
from xorq.vendor.ibis.expr.operations.udf import (
    AggUDF,
    InputType,
    ScalarUDF,
)


def expr_is_bound(expr):
    backends, _ = expr._find_backends()
    return bool(backends)


def unbound_expr_to_default_sql(expr):
    if expr_is_bound(expr):
        raise ValueError
    default_sql = xo.to_sql(expr)
    return str(default_sql)


def normalize_inmemorytable(dt):
    if not isinstance(dt, ir.InMemoryTable):
        raise ValueError
    return normalize_seq_with_caller(
        dt.schema.to_pandas(),
        # in memory: so we can assume it's reasonable to hash the data
        tuple(
            dask.base.tokenize(el.serialize().to_pybytes())
            for el in xo.to_pyarrow_batches(dt.to_expr())
        ),
        caller="normalize_inmemorytable",
    )


def normalize_memory_databasetable(dt):
    if dt.source.name not in ("pandas", "let", "datafusion", "duckdb"):
        raise ValueError
    return normalize_seq_with_caller(
        # we are normalizing the data, we don't care about the connection
        # dt.source,
        dt.schema.to_pandas(),
        # in memory: so we can assume it's reasonable to hash the data
        tuple(
            dask.base.tokenize(el.serialize().to_pybytes())
            for el in xo.to_pyarrow_batches(dt.to_expr())
        ),
        caller="normalize_memory_databasetable",
    )


def normalize_pandas_databasetable(dt):
    if dt.source.name != "pandas":
        raise ValueError
    return normalize_memory_databasetable(dt)


def normalize_datafusion_databasetable(dt):
    if dt.source.name not in ("datafusion", "let"):
        raise ValueError
    table = dt.source.con.table(dt.name)
    ep_str = str(table.execution_plan())
    if ep_str.startswith(("ParquetExec:", "CsvExec:")) or re.match(
        r"DataSourceExec:.+file_type=(csv|parquet)", ep_str
    ):
        return normalize_seq_with_caller(
            dt.schema.to_pandas(),
            # ep_str denotes the parquet files to be read
            # FIXME: md5sum on detected .parquet files?
            ep_str,
        )
    elif ep_str.startswith(("MemoryExec:", "DataSourceExec:")):
        return normalize_memory_databasetable(dt)
    elif ep_str.startswith("PyRecordBatchProviderExec"):
        return normalize_seq_with_caller(
            dt.schema.to_pandas(),
            dt.name,
        )

    else:
        raise ValueError


def normalize_remote_databasetable(dt):
    return normalize_seq_with_caller(
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        caller="normalize_remote_databasetable",
    )


def normalize_postgres_databasetable(dt):
    from xorq.common.utils.postgres_utils import get_postgres_n_reltuples

    if dt.source.name != "postgres":
        raise ValueError
    return normalize_seq_with_caller(
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        get_postgres_n_reltuples(dt),
        caller="normalize_postgres_databasetable",
    )


def normalize_pyiceberg_database_table(dt):
    from xorq.common.utils.pyiceberg_utils import get_iceberg_snapshots_ids

    if dt.source.name != "pyiceberg":
        raise ValueError

    return normalize_seq_with_caller(
        dt.name,
        dt.schema,
        dt.source,
        dt.namespace,
        get_iceberg_snapshots_ids(dt),
        caller="normalize_pyiceberg_databasetable",
    )


def normalize_snowflake_databasetable(dt):
    from xorq.common.utils.snowflake_utils import get_snowflake_last_modification_time

    if dt.source.name != "snowflake":
        raise ValueError
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
        raise ValueError
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
        raise ValueError
    name = sg.table(dt.name, quoted=dt.source.compiler.quoted).sql(
        dialect=dt.source.name
    )

    ((_, plan),) = dt.source.raw_sql(f"EXPLAIN SELECT * FROM {name}").fetchall()
    scan_line = plan.split("\n")[1]
    execution_plan_name = r"\s*│\s*(\w+)\s*│\s*"
    match re.match(execution_plan_name, scan_line).group(1):
        case "ARROW_SCAN":
            return normalize_memory_databasetable(dt)
        case "READ_PARQUET" | "READ_CSV" | "SEQ_SCAN":
            return normalize_duckdb_file_read(dt)
        case _:
            raise NotImplementedError(scan_line)


def normalize_duckdb_file_read(dt):
    name = sg.exp.convert(dt.name).sql(dialect=dt.source.name)
    (sql_ddl_statement,) = dt.source.con.sql(
        f"select sql from duckdb_views() where view_name = {name} UNION select sql from duckdb_tables() where table_name = {name}"
    ).fetchone()
    return normalize_seq_with_caller(
        dt.schema.to_pandas(),
        # sql_ddl_statement denotes the definition of the table, expressed as SQL DDL-statement.
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


def normalize_letsql_databasetable(dt):
    if dt.source.name != "let":
        raise ValueError
    if isinstance(dt, rel.FlightExpr):
        return dask.base.normalize_token(
            (
                "normalize_letsql_databasetable",
                dt.input_expr,
                # we need to "stabilize" the name of the tables in the unbound expr
                rename_unbound_static(dt.unbound_expr.op()).to_expr(),
                dt.make_connection,
            )
        )
    elif isinstance(dt, rel.FlightUDXF):
        return dask.base.normalize_token(
            (
                "normalize_letsql_databasetable",
                dt.input_expr,
                dt.udxf.exchange_f,
                dt.make_connection,
            )
        )
    native_source = dt.source._sources.get_backend(dt)

    if native_source.name == "let":
        return normalize_datafusion_databasetable(dt)
    new_dt = rel.make_native_op(dt)
    return dask.base.normalize_token(new_dt)


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
    path = next(
        el
        for el in (
            read_kwargs.get(name)
            for name in (
                "path",
                "source",
                "source_list",  # duckdb
            )
        )
        if el
    )
    if isinstance(path, (str, pathlib.Path)):
        path = str(path)
        if path.startswith("http") or path.startswith("https:"):
            req = urllib.request.Request(path, method="HEAD")
            resp = urllib.request.urlopen(req)

            headers = resp.info()

            tpls = tuple(
                (k, headers.get(k))
                for k in (
                    "Last-Modified",
                    "Content-Length",
                    "Content-Type",
                )
            )
        elif path.startswith(("s3", "gs", "gcs")):
            metadata = xo.get_object_metadata(
                path, **{k: v for k, v in read_kwargs.items() if k != "path"}
            )
            tpls = tuple(
                (k, metadata.get(k))
                for k in ("location", "last_modified", "size", "e_tag", "version")
            )
        elif (path := pathlib.Path(path)).exists():
            tpls = read.normalize_method(path)
        else:
            raise NotImplementedError(f'Don\'t know how to deal with path "{path}"')
    elif isinstance(path, (list, tuple)) and all(isinstance(el, str) for el in path):
        raise NotImplementedError
    else:
        raise NotImplementedError
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
        "let": normalize_letsql_databasetable,
        "duckdb": normalize_duckdb_databasetable,
        "trino": normalize_remote_databasetable,
        "bigquery": normalize_bigquery_databasetable,
        "pyiceberg": normalize_pyiceberg_database_table,
    }
    f = dct[dt.source.name]
    return f(dt)


@dask.base.normalize_token.register(rel.RemoteTable)
def normalize_remote_table(dt):
    if not isinstance(dt, rel.RemoteTable):
        raise ValueError

    return normalize_seq_with_caller(
        ("schema", dt.schema),
        ("expr", dt.remote_expr),
        # only thing that matters is the type of source its going into
        ("source", dt.source.name),
        caller="normalize_remote_table",
    )


@dask.base.normalize_token.register(ibis.backends.BaseBackend)
def normalize_backend(con):
    name = con.name
    if name == "snowflake":
        con_details = con.con._host
    elif name == "postgres":
        con_dct = con.con.get_dsn_parameters()
        con_details = {k: con_dct[k] for k in ("host", "port", "dbname")}
    elif name == "pandas":
        con_details = id(con.dictionary)
    elif name in ("datafusion", "duckdb", "let"):
        con_details = id(con.con)
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
    else:
        raise ValueError
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


@dask.base.normalize_token.register(ScalarUDF)
def normalize_scalar_udf(udf):
    typs = tuple(arg.dtype for arg in udf.args)
    return normalize_seq_with_caller(
        ScalarUDF,
        typs,
        udf.dtype,
        udf.__func__,
        #
        # ExprScalarUDF
        udf.__config__.get("computed_kwargs_expr"),
        # we are insensitive to these for now
        # udf.__udf_namespace__,
        # udf.__func_name__,
        caller="normalize_scalar_udf",
    )


@dask.base.normalize_token.register(AggUDF)
def normalize_agg_udf(udf):
    (*args, where) = udf.args
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


def opaque_node_replacer(node, kwargs):
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
            new_node = xo.table(
                node.schema,
                name=dask.base.tokenize(node.parent),
            ).op()
        case rel.Read():
            new_node = xo.table(
                node.schema,
                name=dask.base.tokenize(node),
            ).op()
        case rel.RemoteTable():
            new_node = xo.table(
                node.schema,
                name=dask.base.tokenize(node.remote_expr),
            ).op()
        case rel.FlightUDXF() | rel.FlightExpr():
            new_node = xo.table(
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
        case _:
            if isinstance(node, opaque_ops):
                raise ValueError(f"unhandled opaque node type: {type(node)}")
            elif kwargs:
                new_node = node.__recreate__(kwargs)
            else:
                new_node = node
    return new_node


@dask.base.normalize_token.register(ibis.expr.types.Expr)
def normalize_expr(expr):
    op = expr.op()
    sql = unbound_expr_to_default_sql(
        op.replace(opaque_node_replacer).to_expr().unbind()
    )
    reads = op.find(rel.Read)
    dts = op.find((ir.DatabaseTable, rel.FlightExpr, rel.FlightUDXF))
    udfs = op.find((AggUDF, ScalarUDF))
    mems = op.find(ir.InMemoryTable)
    token = normalize_seq_with_caller(
        sql,
        reads,
        dts,
        udfs,
        tuple(map(normalize_inmemorytable, mems)),
        caller="normalize_expr",
    )
    return token
