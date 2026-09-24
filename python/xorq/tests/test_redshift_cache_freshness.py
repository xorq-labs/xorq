"""The Redshift cache-freshness probe must issue no DDL.

Sited in ``python/xorq/tests/`` rather than under
``python/xorq/backends/redshift/tests/`` for the reason given in
``test_redshift_backend.py``: ``backends/conftest.py`` auto-applies a backend
marker by path and CI selects by marker, so a test under the backend directory
would run only in the credential-gated workflow. Nothing here needs credentials.

The defect has two distinct shapes depending on which backend reaches Redshift.
**Only the second is fixed, and only the second is covered here:**

* Through the **postgres** backend -- reaching Redshift over an SSH tunnel as a
  Postgres profile, which is how it was first reported -- dasher's per-backend
  dispatch calls ``get_postgres_n_reltuples``, which issues ``CHECKPOINT`` and
  then ``ANALYZE "<table>"``. The first is a syntax error on Redshift. The
  second is worse than a syntax error: on a warehouse that does accept it,
  ``ANALYZE`` is a real, expensive, write-privileged operation run as a side
  effect of computing a cache key. **This path is out of scope here**: dispatch
  keys on ``dt.source.name``, a postgres-named profile still reports
  ``postgres``, and separating a Redshift endpoint from a PostgreSQL one behind
  that name needs a signal nothing in this change establishes. Tracked
  separately; nothing below tests it, and no test here should be read as
  covering it.
* Through the dedicated **redshift** backend, dasher's dispatch is a bare dict
  lookup with no ``redshift`` key and no default, so computing a cache key
  raises ``KeyError: 'redshift'`` before any SQL is sent. That is what these
  tests cover.

So "it no longer issues CHECKPOINT" is not sufficient evidence on its own --
raising KeyError also satisfies it. These tests assert the positive property
too, and assert it **at the tokenize boundary**: a normalizer may return a
perfectly well-formed tuple that the only consumer of that tuple then refuses
to encode, which is invisible to any assertion that stops at ``isinstance(key,
tuple)``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from decimal import Decimal

import pytest


# Must run BEFORE the xorq.backends.redshift import below. That module reaches
# vendor/ibis/backends/postgres, which imports psycopg unguarded, and psycopg
# ships in the ``postgres`` extra rather than in the core dependencies. CI runs
# ``pytest -m <backend>`` with no path filter, so every job COLLECTS this file,
# and the nine matrix jobs without that extra failed collection outright rather
# than deselecting -- a red build that says ModuleNotFoundError, not a skip.
# ``backends/conftest.py`` guards ``backends/<name>/`` paths only, and this file
# is deliberately sited outside them (see the docstring above), so the guard has
# to be here. The E402s are that guard running first, not import sloppiness.
pytest.importorskip("psycopg")

import xorq.vendor.ibis.expr.operations as ops  # noqa: E402
import xorq.vendor.ibis.expr.schema as sch  # noqa: E402
from xorq.backends.redshift import DEFAULT_PORT  # noqa: E402
from xorq.backends.redshift import Backend as RedshiftBackend  # noqa: E402
from xorq.caching.strategy import SnapshotStrategy  # noqa: E402
from xorq.common.exceptions import RedshiftFreshnessUnavailable  # noqa: E402
from xorq.common.utils.dasher import HASHER  # noqa: E402
from xorq.common.utils.dasher._relations import _databasetable_dispatcher  # noqa: E402
from xorq.common.utils.redshift_utils import (  # noqa: E402
    get_redshift_row_counts,
    resolve_redshift_schema,
)


# Anything that writes, not merely anything that says CREATE. The probe is
# allowed exactly two shapes of statement: the svv_table_info read and the
# current_schema() resolution.
DDL_TOKENS = (
    "CHECKPOINT",
    "ANALYZE",
    "CREATE ",
    "DROP ",
    "VACUUM",
    "ALTER ",
    "INSERT ",
    "UPDATE ",
    "DELETE ",
    "TRUNCATE",
    "GRANT ",
)

SESSION_SCHEMA = "analytics"


class _ConnectionInfo:
    """The psycopg ``info`` surface ``normalize_redshift_backend`` reads."""

    port = DEFAULT_PORT

    @staticmethod
    def get_parameters() -> dict[str, str]:
        return {"host": "example.invalid", "dbname": "dev"}


class RecordingConnection:
    """Records every statement and serves canned catalog answers.

    Deliberately *not* a mock that accepts anything: an unexpected call should
    be visible in ``statements`` rather than silently absorbed, because the
    whole point of these tests is which statements get issued.
    """

    def __init__(
        self,
        rows: tuple = ((12345, 12345),),
        current_schema: str = SESSION_SCHEMA,
        relkind: tuple = (("r",),),
    ) -> None:
        # (sql, params) rather than sql alone. Recording only the statement
        # meant the schema and name actually BOUND were asserted nowhere, so a
        # probe that ignored its schema argument -- or swapped name and schema
        # -- passed every test in this file.
        self.calls: list[tuple[str, object]] = []
        self.info = _ConnectionInfo()
        self._rows = rows
        self._current_schema = current_schema
        self._relkind = relkind

    @property
    def statements(self) -> list[str]:
        return [sql for sql, _ in self.calls]

    def params_for(self, fragment: str) -> object:
        """The parameters bound to the one statement containing ``fragment``."""
        matches = [params for sql, params in self.calls if fragment in sql]
        assert len(matches) == 1, f"{fragment!r} matched {len(matches)} statements"
        return matches[0]

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        yield


class _RecordingCursor:
    def __init__(self, con: RecordingConnection) -> None:
        self._con = con
        self._last = ""

    def __enter__(self) -> _RecordingCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: object, *args: object, **kwargs: object) -> _RecordingCursor:
        self._last = str(sql)
        self._con.calls.append((self._last, args[0] if args else kwargs.get("params")))
        return self

    def fetchall(self) -> list[tuple]:
        # Three statement shapes reach this cursor and they must not be
        # conflated: ``current_database`` on this backend is a
        # ``current_schema()`` read, the probe reads svv_table_info, and the
        # absent-row disambiguation reads pg_class. Serving the row counts to
        # all three would hand the probe a schema name as a row count.
        lowered = self._last.lower()
        if "current_schema" in lowered:
            return [(self._con._current_schema,)]
        if "pg_class" in lowered:
            return list(self._con._relkind)
        return list(self._con._rows)


def make_con(
    rows: tuple = ((12345, 12345),),
    current_schema: str = SESSION_SCHEMA,
    relkind: tuple = (("r",),),
) -> RedshiftBackend:
    con = RedshiftBackend(host="example.invalid", port=DEFAULT_PORT)
    con.con = RecordingConnection(rows, current_schema=current_schema, relkind=relkind)
    return con


def make_dt(
    con: RedshiftBackend, name: str = "offers", database: str | None = "sales"
) -> ops.DatabaseTable:
    return ops.DatabaseTable(
        name=name,
        schema=sch.Schema({"id": "int64", "amt": "float64"}),
        source=con,
        namespace=ops.Namespace(catalog=None, database=database),
    )


def test_normalizing_a_redshift_table_issues_no_ddl() -> None:
    """The headline assertion, and the one the issue is filed about."""
    con = make_con()
    _databasetable_dispatcher(make_dt(con))

    issued = " ".join(con.con.statements).upper()
    for token in DDL_TOKENS:
        assert token not in issued, f"{token!r} was issued to the warehouse: {issued!r}"


def test_the_key_tokenizes() -> None:
    """Not raising in the normalizer is not the same as producing a usable key.

    The normalizer's tuple embeds ``dt.source``, and tokenizing that reaches
    dasher's backend rule -- a ``match con.name`` whose default raises. Before
    this was registered, every assertion in this file passed while
    ``HASHER.tokenize`` raised ``ValueError: no normalization rule for backend
    'redshift'``: the pre-fix ``KeyError`` had simply moved one layer out. An
    assertion that stops at the dispatcher cannot see that.
    """
    token = HASHER.tokenize(make_dt(make_con()))
    assert isinstance(token, str) and token


def test_the_no_probe_fallback_the_error_recommends_also_tokenizes() -> None:
    """The privilege error tells the user to switch to ``ParquetSnapshotCache``.

    That path runs ``SnapshotStrategy.normalize_backend``, which for a backend
    outside ``NAME_ONLY_BACKEND_NAMES`` delegates to ``HASHER.normalize``. It
    hit the identical missing rule, so the actionable error was handing the
    user an action that raised too.
    """
    assert SnapshotStrategy.normalize_backend(make_con())


def test_catalog_counts_are_coerced_from_decimal() -> None:
    """``tbl_rows`` is ``numeric(38,0)``, which psycopg returns as ``Decimal``.

    dasher's encoder takes only str/int/float/bool/bytes/None, so an
    un-coerced ``Decimal`` raises at tokenize time -- after the probe has
    already succeeded. A fake serving Python ``int`` cannot show this, so the
    fake serves what the driver actually serves.
    """
    con = make_con(rows=((Decimal(12345), Decimal(12000)),))
    counts = get_redshift_row_counts(make_dt(con))
    assert counts == (12345, 12000)
    assert all(type(count) is int for count in counts)
    assert HASHER.tokenize(make_dt(make_con(rows=((Decimal(1), Decimal(1)),))))


def test_the_key_carries_a_freshness_component() -> None:
    """The key must change when the underlying row count changes.

    This is what separates the fix from routing Redshift to the identity-only
    ``normalize_remote_databasetable`` that trino and gizmosql use. That would
    also issue no DDL and also return a tuple -- and would let a SourceStorage
    cache serve stale numbers forever, converting a loud failure into a quiet
    wrong answer.
    """
    before = HASHER.tokenize(make_dt(make_con(rows=((100, 100),))))
    after = HASHER.tokenize(make_dt(make_con(rows=((200, 200),))))
    assert before != after, (
        "cache key is insensitive to the source row count, so a cached "
        "result would never be invalidated by upstream changes"
    )


def test_the_key_moves_when_rows_are_deleted_but_not_vacuumed() -> None:
    """``tbl_rows`` alone cannot see a DELETE.

    AWS documents it as including rows marked for deletion but not yet
    vacuumed, so a delete leaves it where it was; ``estimated_visible_rows``
    drops. Keying on only the first would serve a cache that a DELETE never
    invalidates -- the same silent staleness as ``reltuples``, reached by a
    different route.
    """
    before = HASHER.tokenize(make_dt(make_con(rows=((100, 100),))))
    after = HASHER.tokenize(make_dt(make_con(rows=((100, 60),))))
    assert before != after, (
        "cache key ignores estimated_visible_rows, so rows deleted but not "
        "yet vacuumed would never invalidate a cached result"
    )


def test_an_unqualified_table_probes_the_session_schema_not_public() -> None:
    """``con.table("offers")`` carries no schema; ``search_path`` does.

    ``table()`` builds ``Namespace(database=None)`` and resolves nothing, while
    ``_post_connect`` has already set ``search_path`` from the ``schema=``
    connect kwarg. Defaulting the probe to ``"public"`` therefore queried a
    schema the caller never named: no row matched, the probe returned ``None``
    forever, and a SourceStorage cache never invalidated.
    """
    con = make_con(current_schema=SESSION_SCHEMA)
    dt = make_dt(con, database=None)

    assert resolve_redshift_schema(dt) == SESSION_SCHEMA
    assert get_redshift_row_counts(dt) == (12345, 12345)
    assert any("current_schema" in s.lower() for s in con.con.statements), (
        "the unqualified table was probed without resolving search_path"
    )


def test_unqualified_tables_in_different_schemas_get_different_keys() -> None:
    """The resolved schema has to be *in* the key, not merely used to build it.

    Two connections scoped to different schemas, each holding a table of the
    same name with the same row count, are different tables. With only
    ``dt.namespace`` in the key -- ``None`` for both -- they collide.
    """
    left = make_dt(make_con(current_schema="analytics"), database=None)
    right = make_dt(make_con(current_schema="staging"), database=None)
    assert HASHER.tokenize(left) != HASHER.tokenize(right)


def test_the_freshness_query_reads_a_redshift_catalog_view() -> None:
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


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("CHECKPOINT", id="checkpoint"),
        pytest.param("ANALYZE", id="analyze"),
    ],
)
def test_the_postgres_probe_helpers_are_not_reachable_from_redshift(token: str) -> None:
    """Guards the regression path specifically.

    ``get_postgres_n_reltuples`` is selected by backend *name*. A future rename
    would route the redshift-named backend straight back into the DDL-issuing
    probe. This pins that it does not reach it today.

    It says nothing about a *postgres*-named profile pointed at Redshift: that
    backend is not this one, and the module docstring explains why it is out of
    scope here.
    """
    con = make_con()
    _databasetable_dispatcher(make_dt(con))
    assert not any(token in s.upper() for s in con.con.statements)


class _RaisingCursor(_RecordingCursor):
    """Records the statement, then fails the way a driver does."""

    def __init__(self, con: RecordingConnection, exc: BaseException) -> None:
        super().__init__(con)
        self._exc = exc

    def execute(self, sql: object, *args: object, **kwargs: object) -> _RecordingCursor:
        super().execute(sql, *args, **kwargs)
        raise self._exc


class RaisingConnection(RecordingConnection):
    def __init__(self, exc: BaseException, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._exc = exc

    def cursor(self) -> _RaisingCursor:
        return _RaisingCursor(self, self._exc)


def make_insufficient_privilege() -> Exception:
    """Stands in for ``psycopg.errors.InsufficientPrivilege``.

    Matched on ``sqlstate`` in ``redshift_utils``, which is what psycopg sets on
    the instance, so a stand-in carrying the code exercises the real path
    without importing psycopg here.
    """
    exc = Exception("permission denied for relation svv_table_info")
    exc.sqlstate = "42501"
    return exc


def make_raising_dt(
    exc: BaseException, database: str | None = "sales"
) -> ops.DatabaseTable:
    con = RedshiftBackend(host="example.invalid", port=DEFAULT_PORT)
    con.con = RaisingConnection(exc)
    return make_dt(con, database=database)


def test_a_read_only_user_gets_an_actionable_error_not_insufficient_privilege() -> None:
    """Verified live, and it is why this test exists.

    A Redshift user with USAGE on the schema and SELECT on its tables -- the
    least-privilege shape reported from the field -- CANNOT read
    ``svv_table_info``:

        InsufficientPrivilege: permission denied for relation svv_table_info

    Offline tests could not have found this; the fix passed every one of them
    while being unusable for the exact user the issue came from. Left alone, a
    raw psycopg error surfaces from inside cache-key computation, which is the
    same genre of unactionable failure as the CHECKPOINT syntax error this
    issue is about.
    """
    with pytest.raises(RedshiftFreshnessUnavailable) as excinfo:
        get_redshift_row_counts(make_raising_dt(make_insufficient_privilege()))

    message = str(excinfo.value)
    assert "svv_table_info" in message
    assert "GRANT SELECT" in message
    assert "ParquetSnapshotCache" in message
    assert "RedshiftFreshnessUnavailable" in message, (
        "the error tells the user to catch a type but not where to import it"
    )


def test_the_error_warns_against_the_reltuples_workaround() -> None:
    """The obvious workaround is readable by that user and silently wrong.

    Measured live: after inserting 5 rows with no ANALYZE, the real count went
    12 -> 17 and svv_table_info tracked it, while ``pg_class.reltuples`` stayed
    at 12. Keying on it yields a cache that never invalidates -- so the error
    names it explicitly rather than leaving the next person to rediscover it.
    """
    with pytest.raises(RedshiftFreshnessUnavailable) as excinfo:
        get_redshift_row_counts(make_raising_dt(make_insufficient_privilege()))
    assert "reltuples" in str(excinfo.value)


def test_a_non_privilege_failure_propagates_unchanged() -> None:
    """The negative control the privilege tests need to mean anything.

    Without it, an ``_is_permission_error`` that returned ``True``
    unconditionally would pass every other test in this file -- and would
    relabel every driver failure, however unrelated, as a missing GRANT.
    """
    boom = Exception("connection is closed")
    with pytest.raises(Exception) as excinfo:
        get_redshift_row_counts(make_raising_dt(boom))
    # Identity, not a message match: RedshiftFreshnessUnavailable subclasses
    # Exception and interpolates the original into its own text, so
    # `pytest.raises(Exception, match=...)` was satisfied by the WRAPPER and
    # passed even with _is_permission_error stubbed to always return True --
    # the precise regression this test claims to guard.
    assert excinfo.value is boom


def test_a_filesystem_permission_error_is_not_relabelled_as_a_missing_grant() -> None:
    """``PermissionError`` says "Permission denied" and has nothing to do with GRANT.

    An unreadable ssl key or pgpass file raises one on the way to the
    warehouse. Matching the *message* caught it and told the reader to grant
    SELECT on a catalog view, sending them somewhere there is nothing to find;
    matching SQLSTATE 42501 does not.
    """
    with pytest.raises(PermissionError):
        get_redshift_row_counts(
            make_raising_dt(PermissionError(13, "Permission denied"))
        )


def test_an_ambiguous_catalog_answer_names_the_table() -> None:
    """Two rows means the count to key on is a guess.

    The previous unpack raised ``ValueError: too many values to unpack`` with
    no table, no schema and no cause -- true, and useless to act on.
    """
    con = make_con(rows=((100, 100), (200, 200)))
    with pytest.raises(RedshiftFreshnessUnavailable) as excinfo:
        get_redshift_row_counts(make_dt(con))
    message = str(excinfo.value)
    assert "offers" in message
    assert "should not be reachable" in message


def test_a_genuinely_absent_table_still_returns_none_rather_than_raising() -> None:
    """Absence and unreadability must not be conflated.

    Verified live: a table that exists but has had no data written does not
    appear in svv_table_info at all. That is a legitimate empty table, not a
    privilege problem, and a cache key is the wrong place to fail on it.
    """
    con = make_con(rows=())
    assert get_redshift_row_counts(make_dt(con)) is None


def test_the_probe_binds_the_resolved_schema_and_name() -> None:
    """Asserting on the emitted SQL is not asserting on what was BOUND.

    The statement text is a constant, so a probe that ignored its ``schema``
    argument, or that swapped ``name`` and ``schema``, emitted exactly the same
    SQL and passed every other test here. Only the parameters distinguish them.
    """
    con = make_con(current_schema=SESSION_SCHEMA)
    get_redshift_row_counts(make_dt(con, database=None))
    assert con.con.params_for("svv_table_info") == {
        "name": "offers",
        "schema": SESSION_SCHEMA,
    }


def test_a_relation_svv_table_info_never_tracks_raises_rather_than_freezing() -> None:
    """A view or external table is absent from the view for a different reason.

    ``svv_table_info`` lists tables and materialized views holding at least one
    row, so a plain view, a late-binding view, a Spectrum external table and a
    session-temp table are all permanently absent. Returning ``None`` for those
    is not "empty", it is "unknowable" -- and it yields a key that can never
    change no matter what the data does, which is the silent staleness this
    module exists to prevent.
    """
    con = make_con(rows=(), relkind=(("v",),))
    with pytest.raises(RedshiftFreshnessUnavailable) as excinfo:
        get_redshift_row_counts(make_dt(con))
    message = str(excinfo.value)
    assert "offers" in message
    assert "ParquetSnapshotCache" in message


def test_a_name_absent_from_pg_class_also_raises() -> None:
    """No relkind row at all: dropped, or not visible in the probed schema.

    Distinguished from the empty-table case by the same read, and it must not
    be allowed to masquerade as one.
    """
    con = make_con(rows=(), relkind=())
    with pytest.raises(RedshiftFreshnessUnavailable):
        get_redshift_row_counts(make_dt(con))


def test_an_empty_ordinary_table_is_still_none_after_the_relkind_check() -> None:
    """The disambiguation must not turn the legitimate empty case into an error.

    Measured live: an ordinary table with no rows written is absent from
    svv_table_info and reports ``relkind='r'``. ``None`` is the honest key
    component for it -- it changes as soon as data lands.
    """
    con = make_con(rows=(), relkind=(("r",),))
    assert get_redshift_row_counts(make_dt(con)) is None


def test_a_catalog_qualified_table_is_refused_rather_than_mismeasured() -> None:
    """svv_table_info describes the connected database only.

    Probing ``otherdb.sales.t`` from this connection would silently score the
    local ``sales.t``, or nothing at all. Both are wrong answers with no error,
    so the probe refuses instead.
    """
    con = make_con()
    dt = ops.DatabaseTable(
        name="offers",
        schema=sch.Schema({"id": "int64"}),
        source=con,
        namespace=ops.Namespace(catalog="otherdb", database="sales"),
    )
    with pytest.raises(RedshiftFreshnessUnavailable) as excinfo:
        get_redshift_row_counts(dt)
    assert "otherdb" in str(excinfo.value)
    assert not con.con.calls, "refusal must come before any statement is issued"
