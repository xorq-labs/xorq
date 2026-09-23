"""The Redshift cache-freshness probe must issue no DDL.

Sited in ``python/xorq/tests/`` rather than under
``python/xorq/backends/redshift/tests/`` for the reason given in
``test_redshift_backend.py``: ``backends/conftest.py`` auto-applies a backend
marker by path and CI selects by marker, so a test under the backend directory
would run only in the credential-gated workflow. Nothing here needs credentials.

The defect has two distinct shapes depending on which backend reaches Redshift,
and both are covered below because both are live in the field:

* Through the **postgres** backend -- which is how the rc16 customer reached
  Redshift, over an SSH tunnel as a Postgres profile -- dasher's per-backend
  dispatch calls ``get_postgres_n_reltuples``, which issues ``CHECKPOINT`` and
  then ``ANALYZE "<table>"``. The first is a syntax error on Redshift
  (transcript :5101). The second is worse than a syntax error: on a warehouse
  that does accept it, ``ANALYZE`` is a real, expensive, write-privileged
  operation run as a side effect of computing a cache key.
* Through the dedicated **redshift** backend, dasher's dispatch is a bare dict
  lookup with no ``redshift`` key and no default, so computing a cache key
  raises ``KeyError: 'redshift'`` before any SQL is sent.

So "it no longer issues CHECKPOINT" is not sufficient evidence on its own --
raising KeyError also satisfies it. These tests assert the positive property
too: a usable key that still carries a freshness component.
"""

from __future__ import annotations

import contextlib

import pytest

import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.schema as sch
from xorq.backends.redshift import DEFAULT_PORT
from xorq.backends.redshift import Backend as RedshiftBackend
from xorq.common.utils.dasher._relations import _databasetable_dispatcher
from xorq.common.utils.redshift_utils import (
    RedshiftFreshnessUnavailable,
    get_redshift_n_rows,
)


DDL_TOKENS = ("CHECKPOINT", "ANALYZE", "CREATE ", "DROP ", "VACUUM")


class RecordingConnection:
    """Records every statement and serves a canned row count.

    Deliberately *not* a mock that accepts anything: an unexpected call should
    be visible in ``statements`` rather than silently absorbed, because the
    whole point of these tests is which statements get issued.
    """

    def __init__(self, rows=((12345,),)):
        self.statements = []
        self._rows = rows

    def cursor(self):
        return _RecordingCursor(self)

    @contextlib.contextmanager
    def transaction(self):
        yield

    def execute(self, sql, *args, **kwargs):
        self.statements.append(str(sql))
        return _RecordingCursor(self)


class _RecordingCursor:
    def __init__(self, con):
        self._con = con

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, *args, **kwargs):
        self._con.statements.append(str(sql))
        return self

    def fetchall(self):
        return list(self._con._rows)

    def fetchone(self):
        return self._con._rows[0] if self._con._rows else None

    def close(self):
        pass


def make_con(rows=((12345,),)):
    con = RedshiftBackend()
    type(con).__init__(con, host="example.invalid", port=DEFAULT_PORT)
    con.con = RecordingConnection(rows)
    return con


def make_dt(con, name="offers", database="xorq_test"):
    return ops.DatabaseTable(
        name=name,
        schema=sch.Schema({"id": "int64", "amt": "float64"}),
        source=con,
        namespace=ops.Namespace(catalog=None, database=database),
    )


def test_normalizing_a_redshift_table_issues_no_ddl():
    """The headline assertion, and the one the issue is filed about."""
    con = make_con()
    _databasetable_dispatcher(make_dt(con))

    issued = " ".join(con.con.statements).upper()
    for token in DDL_TOKENS:
        assert token not in issued, f"{token!r} was issued to the warehouse: {issued!r}"


def test_normalizing_a_redshift_table_produces_a_usable_key():
    """Not raising is half the fix; producing a key is the other half.

    Without this, ``KeyError: 'redshift'`` would satisfy the no-DDL assertion
    above perfectly.
    """
    con = make_con()
    key = _databasetable_dispatcher(make_dt(con))
    assert isinstance(key, tuple)
    assert key, "normalizer returned an empty key"


def test_the_key_carries_a_freshness_component():
    """The key must change when the underlying row count changes.

    This is what separates the fix from routing Redshift to the identity-only
    ``normalize_remote_databasetable`` that trino and gizmosql use. That would
    also issue no DDL and also return a tuple -- and would let a SourceStorage
    cache serve stale numbers forever, converting a loud failure into a quiet
    wrong answer.
    """
    before = _databasetable_dispatcher(make_dt(make_con(rows=((100,),))))
    after = _databasetable_dispatcher(make_dt(make_con(rows=((200,),))))
    assert before != after, (
        "cache key is insensitive to the source row count, so a cached "
        "result would never be invalidated by upstream changes"
    )


def test_the_freshness_query_reads_a_redshift_catalog_view():
    """Redshift serves ``svv_table_info``; it does not serve ``pg_stat_user_tables``.

    Asserted on the emitted SQL rather than on the returned value, because a
    normalizer that silently fell back to a constant would pass every other
    test here.
    """
    con = make_con()
    _databasetable_dispatcher(make_dt(con))

    issued = " ".join(con.con.statements).lower()
    assert "svv_table_info" in issued, f"no Redshift catalog read issued: {issued!r}"
    assert "pg_stat_user_tables" not in issued
    assert "relpersistence" not in issued


@pytest.mark.parametrize("token", ["CHECKPOINT", "ANALYZE"])
def test_the_postgres_probe_helpers_are_not_reachable_from_redshift(token):
    """Guards the regression path specifically.

    ``get_postgres_n_reltuples`` is selected by backend *name*. A future rename,
    or a Redshift profile that reports itself as postgres, would route straight
    back into the DDL-issuing probe. This pins that the redshift-named backend
    does not reach it.
    """
    con = make_con()
    _databasetable_dispatcher(make_dt(con))
    assert not any(token in s.upper() for s in con.con.statements)


class _DeniedCursor:
    """Raises the way psycopg does when the view is not readable."""

    def __init__(self, con):
        self._con = con

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, *args, **kwargs):
        self._con.statements.append(str(sql))
        raise _InsufficientPrivilege("permission denied for relation svv_table_info")


class _InsufficientPrivilege(Exception):
    """Stands in for psycopg.errors.InsufficientPrivilege.

    Matched by class NAME in redshift_utils, so a stand-in named the same thing
    exercises the real code path without importing psycopg here.
    """


_InsufficientPrivilege.__name__ = "InsufficientPrivilege"


class DenyingConnection(RecordingConnection):
    def cursor(self):
        return _DeniedCursor(self)


def make_denying_dt():
    con = RedshiftBackend()
    type(con).__init__(con, host="example.invalid", port=DEFAULT_PORT)
    con.con = DenyingConnection()
    return make_dt(con)


def test_a_read_only_user_gets_an_actionable_error_not_insufficient_privilege():
    """Verified live 2026-09-23, and it is why this test exists.

    A Redshift user with USAGE on the schema and SELECT on its tables -- the
    shape of the rc16 reporter's user -- CANNOT read ``svv_table_info``:

        InsufficientPrivilege: permission denied for relation svv_table_info

    Offline tests could not have found this; the fix passed every one of them
    while being unusable for the exact user the issue came from. Left alone, a
    raw psycopg error surfaces from inside cache-key computation, which is the
    same genre of unactionable failure as the CHECKPOINT syntax error this
    issue is about.
    """
    with pytest.raises(RedshiftFreshnessUnavailable) as excinfo:
        get_redshift_n_rows(make_denying_dt())

    message = str(excinfo.value)
    assert "svv_table_info" in message
    assert "GRANT SELECT" in message
    assert "ParquetSnapshotCache" in message


def test_the_error_warns_against_the_reltuples_workaround():
    """The obvious workaround is readable by that user and silently wrong.

    Measured live 2026-09-23: after inserting 5 rows with no ANALYZE, the real
    count went 12 -> 17 and svv_table_info tracked it, while
    ``pg_class.reltuples`` stayed at 12. Keying on it yields a cache that never
    invalidates -- so the error names it explicitly rather than leaving the
    next person to rediscover it.
    """
    with pytest.raises(RedshiftFreshnessUnavailable) as excinfo:
        get_redshift_n_rows(make_denying_dt())
    assert "reltuples" in str(excinfo.value)


def test_a_genuinely_absent_table_still_returns_none_rather_than_raising():
    """Absence and unreadability must not be conflated.

    Verified live: a table that exists but has had no data written does not
    appear in svv_table_info at all. That is a legitimate empty table, not a
    privilege problem, and a cache key is the wrong place to fail on it.
    """
    con = make_con(rows=())
    assert get_redshift_n_rows(make_dt(con)) is None
