from __future__ import annotations

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

    # Load-bearing ordering invariant: staging must fully commit before publish
    # reads it. `tee` writes staging as a side effect while batches pass; the
    # `.aggregate` pipeline breaker pulls the *entire* stream to produce `passed`,
    # so staging is drained before that row reaches publish. This holds only for
    # sinks that write inline in the draining thread (the defaults) — a
    # background-threaded sink would let publish run before staging commits, which
    # is why the publish UDFs raise if the staging artifact is missing.
    wap_expr = (
        expr.tee(make_sink(staging))
        .aggregate(**{PASSED: audit_expr(audit_fn=audit_fn, name=PASSED)})
        .mutate(**{STAGING: literal(staging), FINAL: literal(final)})
        .mutate(**{PUBLISHED: publish.on_expr})
    )
    return wap_expr


def make_sink_with_parquet(path: str) -> ParquetWriteThrough:
    from xorq.writes.enums import WriteMode  # noqa: PLC0415
    from xorq.writes.write_through import ParquetWriteThrough  # noqa: PLC0415

    return ParquetWriteThrough(path=path, mode=WriteMode.CREATE)


def make_sink_with_backend(con, staging: str) -> BackendWriteThrough:
    """Inline ``BackendWriteThrough`` writing the changeset to a staging table.

    Inline (not the threaded sink) so staging fully commits before publish runs
    — the ordering invariant ``make_wap_expr`` relies on.
    """
    from xorq.writes.enums import WriteMode  # noqa: PLC0415
    from xorq.writes.write_through import BackendWriteThrough  # noqa: PLC0415

    return BackendWriteThrough(con=con, table_name=staging, mode=WriteMode.CREATE)


def _default_audit(key, mode) -> Callable[[pd.DataFrame], bool]:
    """Default ``audit_fn``: a no-duplicate-keys gate for ``UPSERT``/``MERGE``.

    Key uniqueness within a changeset is a data-level invariant the reconciliation
    contract cannot check on ``mode``/``key``/``columns`` alone. The mechanisms
    diverge on duplicates — native ``MERGE`` and iceberg ``upsert`` raise,
    ``REWRITE``/``STATEMENT_DML`` silently resolve differently — so the gate fails
    the publish before any tier runs (ADR-0017, open question 5). ``APPEND`` has no
    key, so it keeps the always-True pass-through.
    """
    from xorq.writes.enums import PublishMode  # noqa: PLC0415

    if mode is PublishMode.APPEND:
        return lambda df: True
    cols = list(key)
    return lambda df: not df.duplicated(subset=cols).any()


def make_publish_with_backend(con, *, key=(), mode) -> PublishUDF:
    """Publish UDF wrapping :func:`xorq.writes.publish.publish` for a backend target.

    The WAP-layer glue: reads the ``{staging, final, passed}`` row and hands off to
    the reconciliation layer, which discovers columns and dispatches by strategy.
    """
    import xorq.expr.datatypes as dt  # noqa: PLC0415
    from xorq.expr.udf import make_pandas_udf  # noqa: PLC0415
    from xorq.vendor.ibis import schema  # noqa: PLC0415
    from xorq.writes.enums import PublishMode  # noqa: PLC0415
    from xorq.writes.publish import publish  # noqa: PLC0415

    if mode is not PublishMode.APPEND and not key:
        raise ValueError(f"{mode} requires a non-empty key")

    @make_pandas_udf(
        schema=schema({STAGING: str, FINAL: str, PASSED: bool}),
        return_type=dt.boolean,
        name=f"wap_publish_{mode.value}",
    )
    def publish_udf(df: pd.DataFrame) -> list[bool]:
        if len(df) != 1:
            raise ValueError(f"expected 1 row, got {len(df)}")
        row = df.iloc[0]
        if not row[PASSED]:
            return [False]
        publish(con, row[STAGING], row[FINAL], key=key, mode=mode)
        return [True]

    return publish_udf


def make_publish_with_parquet(*, key=(), mode=None) -> PublishUDF:
    """Publish UDF wrapping :func:`xorq.writes.publish.publish_parquet` (file target).

    ``mode`` defaults to ``APPEND`` (the append-only parquet WAP); ``UPSERT``/
    ``MERGE`` do the anti-join + union-all rewrite over the two files.
    """
    import xorq.expr.datatypes as dt  # noqa: PLC0415
    from xorq.expr.udf import make_pandas_udf  # noqa: PLC0415
    from xorq.vendor.ibis import schema  # noqa: PLC0415
    from xorq.writes.enums import PublishMode  # noqa: PLC0415
    from xorq.writes.publish import publish_parquet  # noqa: PLC0415

    if mode is None:
        mode = PublishMode.APPEND
    if mode is not PublishMode.APPEND and not key:
        raise ValueError(f"{mode} requires a non-empty key")

    @make_pandas_udf(
        schema=schema({STAGING: str, FINAL: str, PASSED: bool}),
        return_type=dt.boolean,
        name=f"publish_with_parquet_{mode.value}",
    )
    def publish_with_parquet(df: pd.DataFrame) -> list[bool]:
        if len(df) != 1:
            raise ValueError(f"expected 1 row, got {len(df)}")
        row = df.iloc[0]
        if not row[PASSED]:
            return [False]
        publish_parquet(row[STAGING], row[FINAL], key=key, mode=mode)
        return [True]

    return publish_with_parquet


def make_backend_wap_expr(con, *, key=(), mode, audit_fn=None) -> Callable[..., Table]:
    """Curried WAP builder for any backend target; mirrors ``make_iceberg_wap_expr``.

    Usage::

        source.pipe(
            make_backend_wap_expr(con, key=["id"], mode=PublishMode.UPSERT),
            staging, final,
        )

    ``audit_fn`` defaults to a no-duplicate-keys gate for ``UPSERT``/``MERGE``
    (:func:`_default_audit`); ``APPEND`` keeps a vacuous always-True pass-through.
    """
    audit = audit_fn if audit_fn is not None else _default_audit(key, mode)
    return make_wap_expr(
        audit_fn=audit,
        make_sink=lambda staging: make_sink_with_backend(con, staging),
        publish=make_publish_with_backend(con, key=key, mode=mode),
    )


def make_parquet_wap_expr(*, key=(), mode=None, audit_fn=None) -> Callable[..., Table]:
    """Curried WAP builder for a parquet (file) target.

    Usage::

        source.pipe(
            make_parquet_wap_expr(key=["id"], mode=PublishMode.UPSERT),
            staging_path, final_path,
        )

    ``mode`` defaults to ``APPEND`` (append-only parquet WAP). ``audit_fn`` defaults
    to the no-duplicate-keys gate for ``UPSERT``/``MERGE`` (:func:`_default_audit`).
    """
    from xorq.writes.enums import PublishMode  # noqa: PLC0415

    if mode is None:
        mode = PublishMode.APPEND
    audit = audit_fn if audit_fn is not None else _default_audit(key, mode)
    return make_wap_expr(
        audit_fn=audit,
        make_sink=make_sink_with_parquet,
        publish=make_publish_with_parquet(key=key, mode=mode),
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
    # Factory that binds `con` and returns a curried builder, so it composes with
    # `.pipe()`. table_name set -> branch strategy on that table; None -> table
    # strategy staging into a separate table.
    #
    # Executing a WAP expr is NOT idempotent: publish is a side effect of
    # execution, every strategy appends to final on a pass and consumes staging,
    # so re-executing accumulates. On audit failure staging is retained for
    # inspection, which then blocks an identical retry (create-mode refuses the
    # stale ref) — drop it or stage under a fresh name to retry.
    if table_name is None:
        # Table strategy is the no-key APPEND case of the backend publish (iceberg
        # APPEND calls publish_staging_table / add_files — the same mechanism as
        # before). See ADR-0017.
        from xorq.writes.enums import PublishMode  # noqa: PLC0415

        return make_wap_expr(
            make_sink=lambda staging: make_sink_with_backend(con, staging),
            publish=make_publish_with_backend(con, mode=PublishMode.APPEND, key=[]),
        )
    # Branch strategy: staging branch + fast-forward main — a distinct mechanism
    # with no incremental-publish equivalent, so it keeps its own publish.
    return make_wap_expr(
        make_sink=make_sink_with_iceberg(con, table_name=table_name),
        publish=make_publish_with_iceberg(con, branch=True),
    )
