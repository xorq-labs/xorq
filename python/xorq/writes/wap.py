from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

from toolz import curry


if TYPE_CHECKING:
    import pandas as pd

    from xorq.backends.pyiceberg import Backend as PyIcebergBackend
    from xorq.vendor.ibis.expr.types import Table, Value
    from xorq.writes.write_through import (
        BackendWriteThrough,
        ParquetWriteThrough,
        WriteThrough,
    )

    class PublishUDF(Protocol):
        """UDF constructor returned by make_pandas_udf for the publish step."""

        fn: Callable[[pd.DataFrame], list[bool]]

        def on_expr(self, e: Table) -> Value: ...


# Column-name contract shared by the audit/publish UDF schemas and the mutate keys.
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
    publish: PublishUDF,
) -> Table:
    from xorq.vendor.ibis.expr.types.generic import literal  # noqa: PLC0415

    # Load-bearing ordering invariant: the staging write must fully commit before
    # publish reads it. `tee` writes to staging as a side effect as batches pass
    # through; `.aggregate` is a pipeline breaker that has to pull the *entire*
    # tee'd stream to produce its single `passed` row, so the staging write is
    # drained to completion before that row flows into the publish `.mutate`.
    #
    # This holds only because the default sinks write inline in the draining
    # thread (ParquetWriteThrough; per-batch BackendWriteThrough). A
    # background-threaded sink (ThreadedBackendWriteThrough / WritePrimaryWriteThrough)
    # would break it: draining the read side does not join the write thread, so
    # publish could run before staging commits. The publish UDFs raise if the
    # staging artifact is missing as a backstop against that.
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


def _make_publish_with_parquet() -> PublishUDF:
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

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
        row = df.iloc[0]
        written = False
        if row[PASSED]:
            staging = Path(row[STAGING])
            final = Path(row[FINAL])
            # Backstop for the ordering invariant in make_wap_expr: if the audit
            # really drained the inline staging write, the file is here by now.
            if not staging.exists():
                raise RuntimeError(
                    f"staging {str(staging)!r} missing at publish: the audit ran "
                    "before the staging write committed (async sink?)"
                )
            # Append into final, then consume staging — mirrors the iceberg
            # strategies (add_files / fast-forward main, then drop the staging
            # table / branch). Parquet has no metadata-only append, so we rewrite:
            # read both sides, write to a temp in final's dir, then swap. The temp
            # shares final's filesystem, so .replace is an atomic same-fs rename;
            # reading staging via pyarrow works across filesystems.
            final.parent.mkdir(parents=True, exist_ok=True)
            tables = [pq.read_table(staging)]
            if final.exists():
                tables.insert(0, pq.read_table(final))
            merged = final.with_name(final.name + ".merge.tmp")
            try:
                pq.write_table(pa.concat_tables(tables), merged)
                merged.replace(final)
            except BaseException:
                merged.unlink(missing_ok=True)
                raise
            # final now holds staging's rows; removing staging is cleanup, so a
            # failure here must not mask a successful publish.
            staging.unlink(missing_ok=True)
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
    con: PyIcebergBackend, staging: str, table_name: str | None = None
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


def make_publish_with_iceberg(
    con: PyIcebergBackend, branch: bool = False
) -> PublishUDF:
    import xorq.expr.datatypes as dt  # noqa: PLC0415
    from xorq.expr.udf import make_pandas_udf  # noqa: PLC0415
    from xorq.vendor.ibis import schema  # noqa: PLC0415

    @make_pandas_udf(
        schema=schema({STAGING: str, FINAL: str, PASSED: bool}),
        return_type=dt.boolean,
        name="publish_with_iceberg_branch" if branch else "publish_with_iceberg",
    )
    def publish_with_iceberg(df: pd.DataFrame) -> list[bool]:
        if len(df) != 1:
            raise ValueError(f"expected 1 row, got {len(df)}")
        row = df.iloc[0]
        written = False
        if row[PASSED]:
            if branch:
                # staging=branch name, final=table name for the branch strategy
                con.publish_branch(row[FINAL], row[STAGING])
            else:
                con.publish_staging_table(row[STAGING], row[FINAL])
            written = True
        return [written]

    return publish_with_iceberg


def make_iceberg_wap_expr(
    con: PyIcebergBackend, table_name: str | None = None
) -> Callable[..., Table]:
    # Unlike make_parquet_wap_expr (a flat function), this is a factory that binds
    # `con` up front and returns a curried builder, so it composes with `.pipe()`
    # and can be reused across exprs without re-threading the connection.
    #
    # Passing table_name selects the branch strategy on that table; otherwise
    # the table strategy stages into a separate table named by the caller.
    #
    # Executing a WAP expr is NOT idempotent: publish runs as a side effect of
    # execution, so re-executing publishes again. Every strategy appends to final
    # on each pass run and consumes staging (iceberg: fast-forward / add_files
    # then drop the branch/table; parquet: merge then unlink), so repeated passing
    # runs accumulate in final.
    #
    # On audit failure the staging branch (branch strategy), staging table
    # (table strategy), or staging file (parquet) is retained for inspection.
    # Re-running against the same target then fails because the stale staging ref
    # still exists; remove it first or stage under a fresh name before retrying.
    return make_wap_expr(
        make_sink=make_sink_with_iceberg(con, table_name=table_name),
        publish=make_publish_with_iceberg(con, branch=table_name is not None),
    )
