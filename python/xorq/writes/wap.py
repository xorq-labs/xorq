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
    make_sink: Callable[[str, str], WriteThrough],
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
    # make_sink receives both names: table-staging sinks use only `staging`,
    # branch staging writes to a branch (named `staging`) of `final` itself.
    wap_expr = (
        expr.tee(make_sink(staging, final))
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


def make_backend_wap_expr(
    con, *, key=(), mode, staging_strategy=None
) -> Callable[..., Table]:
    """WAP builder for any backend target.

    Binds the changeset config (``key``/``mode``); the returned builder takes
    ``(expr, staging, final, audit_fn=None)`` so ``audit_fn`` is supplied at the
    pipe like the core ``make_wap_expr``::

        # default no-duplicate-keys gate for UPSERT/MERGE (APPEND: always-True):
        source.pipe(make_backend_wap_expr(con, key=["id"], mode=PublishMode.UPSERT),
                    staging, final)
        # or supply a real DQ audit at the pipe:
        source.pipe(make_backend_wap_expr(con, key=["id"], mode=PublishMode.UPSERT),
                    staging, final, my_audit_fn)

    ``staging_strategy`` defaults to ``TABLE`` (stage into a separate table,
    publish via the reconciliation layer). ``BRANCH`` stages on a branch — named
    ``staging`` at the pipe — of ``final`` itself and publishes by fast-forward;
    it is ``APPEND``-only with no key, and the backend's type must declare
    ``publish_branch`` (capability is a backend fact, looked up like
    ``publish_strategy`` — see :func:`xorq.writes.publish._backend_strategy`)::

        source.pipe(make_backend_wap_expr(iceberg_con, mode=PublishMode.APPEND,
                    staging_strategy=StagingStrategy.BRANCH), staging, final)
    """
    from xorq.writes.enums import PublishMode, StagingStrategy  # noqa: PLC0415

    if staging_strategy is None:
        staging_strategy = StagingStrategy.TABLE
    if staging_strategy is StagingStrategy.BRANCH:
        if mode is not PublishMode.APPEND:
            raise ValueError(
                f"BRANCH staging publishes by snapshot fast-forward (all-or-"
                f"nothing); {mode} needs a keyed merge — use TABLE staging"
            )
        if key:
            raise ValueError("BRANCH staging takes no key (APPEND-only)")
        if getattr(type(con), "publish_branch", None) is None:
            raise ValueError(
                f"{type(con).__name__} does not support BRANCH staging "
                "(no publish_branch)"
            )

        def make_sink(staging, final):
            return make_sink_with_iceberg(con, staging, table_name=final)

        publish = make_publish_with_iceberg(con, branch=True)
    else:

        def make_sink(staging, final):
            return make_sink_with_backend(con, staging)

        publish = make_publish_with_backend(con, key=key, mode=mode)

    def build(expr, staging, final, audit_fn=None):
        audit = audit_fn if audit_fn is not None else _default_audit(key, mode)
        return make_wap_expr(
            expr,
            staging,
            final,
            audit,
            make_sink=make_sink,
            publish=publish,
        )

    return build


def make_parquet_wap_expr(*, key=(), mode=None) -> Callable[..., Table]:
    """WAP builder for a parquet (file) target.

    ``mode`` defaults to ``APPEND`` (append-only parquet WAP); pass ``key``/``mode``
    for upsert/merge. Like :func:`make_backend_wap_expr`, ``audit_fn`` is supplied
    at the pipe::

        source.pipe(make_parquet_wap_expr(), staging, final, audit_no_nulls)
        source.pipe(make_parquet_wap_expr(key=["id"], mode=PublishMode.UPSERT),
                    staging, final)
    """
    from xorq.writes.enums import PublishMode  # noqa: PLC0415

    if mode is None:
        mode = PublishMode.APPEND
    publish = make_publish_with_parquet(key=key, mode=mode)

    def build(expr, staging, final, audit_fn=None):
        audit = audit_fn if audit_fn is not None else _default_audit(key, mode)
        return make_wap_expr(
            expr,
            staging,
            final,
            audit,
            make_sink=lambda staging, final: make_sink_with_parquet(staging),
            publish=publish,
        )

    return build


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
    """Iceberg APPEND helper — an alias over :func:`make_backend_wap_expr`.

    ``table_name`` set selects ``BRANCH`` staging (the branch's table is the
    ``final`` supplied at the pipe, which must equal ``table_name``); ``None``
    selects ``TABLE`` staging. Prefer ``make_backend_wap_expr`` directly for
    new code — it also unlocks keyed ``UPSERT``/``MERGE`` on TABLE staging.

    Executing a WAP expr is NOT idempotent: publish is a side effect of
    execution, every strategy appends to final on a pass and consumes staging,
    so re-executing accumulates. On audit failure staging is retained for
    inspection, which then blocks an identical retry (create-mode refuses the
    stale ref) — drop it or stage under a fresh name to retry.
    """
    from xorq.writes.enums import PublishMode, StagingStrategy  # noqa: PLC0415

    strategy = StagingStrategy.TABLE if table_name is None else StagingStrategy.BRANCH
    return make_backend_wap_expr(
        con, mode=PublishMode.APPEND, staging_strategy=strategy
    )
