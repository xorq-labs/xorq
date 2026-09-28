"""Redshift connections must never send ``DEALLOCATE ALL``.

Redshift has no such statement. psycopg sends it whenever a transaction rolls
back while it holds server-side prepared statements, and on Redshift the
resulting syntax error rolls back the transaction it was clearing up after -- a
``drop_table`` inside ``con.transaction()`` then leaves its table behind. The
Redshift backend defaults ``prepare_threshold=None`` so psycopg never prepares
a statement and has none to deallocate.

A live PostgreSQL server stands in for Redshift here, and is a faithful one for
this question: whether ``DEALLOCATE ALL`` is sent is decided by psycopg on the
client, and the server only decides whether to reject it. The observable is
libpq's own protocol trace, not psycopg's internal state, so the assertion is
on what went over the wire.

Sited under ``backends/postgres/tests/`` because it needs that server: the
path gives it the ``postgres`` marker, and the job selecting that marker is the
one with the service. The offline Redshift tests live in
``python/xorq/tests/test_redshift_backend.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xorq.backends.postgres import Backend as PostgresBackend
from xorq.backends.redshift import Backend as RedshiftBackend
from xorq.common.utils.postgres_utils import (
    make_connection_defaults,
    make_credential_defaults,
)


def trace_a_rollback_after_repeated_queries(
    backend_cls: type[PostgresBackend], trace_path: Path
) -> tuple[str, int]:
    """Run one query past psycopg's default threshold of 5, roll back a
    transaction, and return the libpq trace and the server's prepared count."""
    con = backend_cls().connect(
        **make_credential_defaults(), **make_connection_defaults()
    )
    raw = con.con
    try:
        with trace_path.open("w") as trace:
            raw.pgconn.trace(trace.fileno())
            try:
                for _ in range(6):
                    with raw.cursor() as cursor:
                        cursor.execute("SELECT %s::int", (1,))
                with raw.cursor() as cursor:
                    (prepared,) = cursor.execute(
                        "SELECT count(*) FROM pg_prepared_statements"
                    ).fetchone()
                with raw.transaction(force_rollback=True):
                    pass
            finally:
                raw.pgconn.untrace()
    finally:
        con.disconnect()
    return trace_path.read_text(), prepared


def test_redshift_rollback_sends_no_deallocate(tmp_path: Path) -> None:
    trace, prepared = trace_a_rollback_after_repeated_queries(
        RedshiftBackend, tmp_path / "trace"
    )

    assert prepared == 0
    assert "DEALLOCATE" not in trace


def test_postgres_rollback_still_deallocates(tmp_path: Path) -> None:
    """The negative control: with psycopg's default threshold the same
    sequence prepares a statement and deallocates it, so the assertion above
    can see a ``DEALLOCATE`` when one is sent."""
    trace, prepared = trace_a_rollback_after_repeated_queries(
        PostgresBackend, tmp_path / "trace"
    )

    assert prepared >= 1
    assert "DEALLOCATE ALL" in trace


@pytest.mark.parametrize(
    "backend_cls",
    [
        pytest.param(RedshiftBackend, id="redshift"),
        pytest.param(PostgresBackend, id="postgres"),
    ],
)
def test_the_trace_captures_the_queries(
    backend_cls: type[PostgresBackend], tmp_path: Path
) -> None:
    """Guards the observable itself: an empty trace would pass the Redshift
    test for the wrong reason."""
    trace, _ = trace_a_rollback_after_repeated_queries(backend_cls, tmp_path / "t")

    assert "pg_prepared_statements" in trace
