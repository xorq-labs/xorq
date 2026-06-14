import shutil

from toolz import curry

import xorq.expr.datatypes as dt
from xorq.expr.udf import agg, make_pandas_udf
from xorq.sinking.sink import BackendSink, ParquetSink, SinkMode
from xorq.vendor.ibis.expr.api import schema as _schema
from xorq.vendor.ibis.expr.types.generic import literal


STAGING = "staging"
FINAL = "final"
PASSED = "passed"
PUBLISHED = "published"


@curry
def audit_expr(expr, audit_fn, name):
    return agg.pandas_df(
        fn=audit_fn,
        schema=expr.schema(),
        return_type=dt.boolean,
        name=name,
    ).on_expr(expr)


@curry
def make_wap_expr(expr, staging, final, audit_fn, make_sink, publish):
    wap_expr = (
        expr.tee(make_sink(staging))
        .aggregate(**{PASSED: audit_expr(audit_fn=audit_fn, name=PASSED)})
        .mutate(**{STAGING: literal(staging), FINAL: literal(final)})
        .mutate(**{PUBLISHED: publish.on_expr})
    )
    return wap_expr


make_sink_with_parquet = curry(ParquetSink, mode=SinkMode.CREATE)


@make_pandas_udf(
    schema=_schema({STAGING: str, FINAL: str, PASSED: bool}),
    return_type=dt.boolean,
    name="publish_with_parquet",
)
def publish_with_parquet(df):
    ((staging, final, passed), *rest) = df.values
    if rest:
        raise ValueError
    written = False
    if passed:
        shutil.copy2(staging, final)
        written = True
    return [written]


make_parquet_wap_expr = make_wap_expr(
    make_sink=make_sink_with_parquet, publish=publish_with_parquet
)


@curry
def make_sink_with_iceberg(con, table_name, mode=SinkMode.CREATE):
    return BackendSink(con=con, table_name=table_name, mode=mode)


def make_publish_with_iceberg(con):
    @make_pandas_udf(
        schema=_schema({STAGING: str, FINAL: str, PASSED: bool}),
        return_type=dt.boolean,
        name="publish_with_iceberg",
    )
    def publish_with_iceberg(df):
        ((staging, final, passed), *rest) = df.values
        if rest:
            raise ValueError
        written = False
        if passed:
            staged = con.table(staging).execute()
            if final in con.list_tables():
                con.insert(final, staged, mode=SinkMode.APPEND)
            else:
                con.create_table(final, staged, overwrite=True)
            con.drop_table(staging)
            written = True
        return [written]

    return publish_with_iceberg


def make_iceberg_wap_expr(con):
    return make_wap_expr(
        make_sink=make_sink_with_iceberg(con),
        publish=make_publish_with_iceberg(con),
    )


@curry
def make_sink_with_iceberg_branch(con, table_name, branch):
    return BackendSink(
        con=con,
        table_name=table_name,
        mode=SinkMode.CREATE,
        kwargs={"branch": branch},
    )


def make_publish_with_iceberg_branch(con):
    @make_pandas_udf(
        schema=_schema({STAGING: str, FINAL: str, PASSED: bool}),
        return_type=dt.boolean,
        name="publish_with_iceberg_branch",
    )
    def publish_with_iceberg_branch(df):
        ((branch, table_name, passed), *rest) = df.values
        if rest:
            raise ValueError
        written = False
        if passed:
            full_name = f"{con.namespace}.{table_name}"
            ice = con.catalog.load_table(full_name)
            staging_snap = ice.refs()[branch].snapshot_id
            ice.manage_snapshots().set_current_snapshot(staging_snap).commit()
            ice = con.catalog.load_table(full_name)
            ice.manage_snapshots().remove_branch(branch).commit()
            written = True
        return [written]

    return publish_with_iceberg_branch


def make_iceberg_branch_wap_expr(con, table_name):
    return make_wap_expr(
        make_sink=make_sink_with_iceberg_branch(con, table_name),
        publish=make_publish_with_iceberg_branch(con),
    )
