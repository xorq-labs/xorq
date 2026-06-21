from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any, Callable

from toolz import curry


if TYPE_CHECKING:
    import pandas as pd

    from xorq.vendor.ibis.backends import BaseBackend
    from xorq.vendor.ibis.expr.types import Table
    from xorq.writes.write_through import (
        BackendWriteThrough,
        ParquetWriteThrough,
        WriteThrough,
    )

STAGING = "staging"
FINAL = "final"
PASSED = "passed"
PUBLISHED = "published"


@curry
def audit_expr(
    expr: Table, audit_fn: Callable[[pd.DataFrame], bool], name: str
) -> Table:
    import xorq.expr.datatypes as dt  # noqa: PLC0415
    from xorq.expr.udf import agg  # noqa: PLC0415

    return agg.pandas_df(
        fn=audit_fn,
        schema=expr.schema(),
        return_type=dt.boolean,
        name=name,
    ).on_expr(expr)


@curry
def make_wap_expr(
    expr: Table,
    staging: str,
    final: str,
    audit_fn: Callable[[pd.DataFrame], bool],
    make_sink: Callable[[str], WriteThrough],
    publish: Any,
) -> Table:
    from xorq.vendor.ibis.expr.types.generic import literal  # noqa: PLC0415

    wap_expr = (
        expr.tee(make_sink(staging))
        .aggregate(**{PASSED: audit_expr(audit_fn=audit_fn, name=PASSED)})
        .mutate(**{STAGING: literal(staging), FINAL: literal(final)})
        .mutate(**{PUBLISHED: publish.on_expr})
    )
    return wap_expr


def _make_sink_with_parquet(path: str) -> ParquetWriteThrough:
    from xorq.writes.enums import WriteMode  # noqa: PLC0415
    from xorq.writes.write_through import ParquetWriteThrough  # noqa: PLC0415

    return ParquetWriteThrough(path=path, mode=WriteMode.CREATE)


def _make_publish_with_parquet() -> Any:
    import xorq.expr.datatypes as dt  # noqa: PLC0415
    from xorq.expr.udf import make_pandas_udf  # noqa: PLC0415
    from xorq.vendor.ibis import schema  # noqa: PLC0415

    @make_pandas_udf(
        schema=schema({STAGING: str, FINAL: str, PASSED: bool}),
        return_type=dt.boolean,
        name="publish_with_parquet",
    )
    def publish_with_parquet(df: pd.DataFrame) -> list[bool]:
        if len(df) != 1:
            raise ValueError(f"expected 1 row, got {len(df)}")
        staging, final, passed = df.values[0]
        written = False
        if passed:
            shutil.copy2(staging, final)
            written = True
        return [written]

    return publish_with_parquet


def make_parquet_wap_expr(
    expr: Table,
    staging: str,
    final: str,
    audit_fn: Callable[[pd.DataFrame], bool],
) -> Table:
    return make_wap_expr(
        expr,
        staging,
        final,
        audit_fn,
        make_sink=_make_sink_with_parquet,
        publish=_make_publish_with_parquet(),
    )


@curry
def make_sink_with_iceberg(
    con: BaseBackend, staging: str, table_name: str | None = None
) -> BackendWriteThrough:
    from xorq.writes.enums import WriteMode  # noqa: PLC0415
    from xorq.writes.write_through import BackendWriteThrough  # noqa: PLC0415

    # Branch strategy (table_name set): write to a staging branch of table_name.
    # Table strategy (table_name None): write to a staging table named `staging`.
    if table_name is None:
        return BackendWriteThrough(con=con, table_name=staging, mode=WriteMode.CREATE)
    return BackendWriteThrough(
        con=con,
        table_name=table_name,
        mode=WriteMode.CREATE,
        kwargs={"branch": staging},
    )


def make_publish_with_iceberg(con: BaseBackend, branch: bool = False) -> Any:
    import xorq.expr.datatypes as dt  # noqa: PLC0415
    from xorq.expr.udf import make_pandas_udf  # noqa: PLC0415
    from xorq.vendor.ibis import schema  # noqa: PLC0415
    from xorq.writes.enums import WriteMode  # noqa: PLC0415

    @make_pandas_udf(
        schema=schema({STAGING: str, FINAL: str, PASSED: bool}),
        return_type=dt.boolean,
        name="publish_with_iceberg_branch" if branch else "publish_with_iceberg",
    )
    def publish_with_iceberg(df: pd.DataFrame) -> list[bool]:
        if len(df) != 1:
            raise ValueError(f"expected 1 row, got {len(df)}")
        staging, final, passed = df.values[0]
        written = False
        if passed:
            if branch:
                # staging=branch name, final=table name for the branch strategy
                full_name = f"{con.namespace}.{final}"
                ice = con.catalog.load_table(full_name)
                staging_snap = ice.refs()[staging].snapshot_id
                ice.manage_snapshots().set_current_snapshot(staging_snap).commit()
                ice = con.catalog.load_table(full_name)
                ice.manage_snapshots().remove_branch(staging).commit()
            else:
                staged = con.table(staging).execute()
                if final in con.list_tables():
                    con.insert(final, staged, mode=WriteMode.APPEND)
                else:
                    con.create_table(final, staged, overwrite=True)
                con.drop_table(staging)
            written = True
        return [written]

    return publish_with_iceberg


def make_iceberg_wap_expr(
    expr: Table,
    staging: str,
    final: str,
    audit_fn: Callable[[pd.DataFrame], bool],
    con: BaseBackend,
    table_name: str | None = None,
) -> Table:
    # Passing table_name selects the branch strategy on that table; otherwise
    # the table strategy stages into a separate table named by the caller.
    return make_wap_expr(
        expr,
        staging,
        final,
        audit_fn,
        make_sink=make_sink_with_iceberg(con, table_name=table_name),
        publish=make_publish_with_iceberg(con, branch=table_name is not None),
    )
