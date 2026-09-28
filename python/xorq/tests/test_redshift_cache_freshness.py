"""Redshift cache keys: freshness refused, snapshot identity complete.

Sited in ``python/xorq/tests/`` rather than under
``python/xorq/backends/redshift/tests/`` for the reason given in
``test_redshift_backend.py``: ``backends/conftest.py`` auto-applies a backend
marker by path and CI selects by marker, so a test under the backend directory
would run only in the credential-gated workflow. Nothing here needs credentials.

The defect this started from has two shapes depending on which backend reaches
Redshift. **Only the second is covered here:**

* Through the **postgres** backend -- reaching Redshift over an SSH tunnel as a
  Postgres profile, which is how it was first reported -- dasher's per-backend
  dispatch calls ``get_postgres_n_reltuples``, which issues ``CHECKPOINT`` and
  then ``ANALYZE "<table>"``. **This path is out of scope here**: dispatch keys
  on ``dt.source.name``, a postgres-named profile still reports ``postgres``,
  and separating a Redshift endpoint from a PostgreSQL one behind that name
  needs a signal nothing in this change establishes.
* Through the dedicated **redshift** backend, dasher's dispatch is a bare dict
  lookup with no ``redshift`` key, so computing a cache key raised
  ``KeyError: 'redshift'`` before any SQL was sent.

The freshness key is now refused outright (see ``redshift_utils`` for why no
Redshift catalog read can serve), so "it issues no DDL" is true trivially and
proves nothing on its own. What these tests pin is the pair of positive
properties: the freshness refusal names the cache that works, and that cache's
key -- asserted **at the tokenize boundary**, where a well-formed tuple the
encoder refuses would show -- tells apart every pair of relations a reader
could confuse.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

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
from xorq.backends.redshift import Backend as RedshiftBackend  # noqa: E402
from xorq.caching.strategy import (  # noqa: E402
    ModificationTimeStrategy,
    SnapshotStrategy,
)
from xorq.common.exceptions import RedshiftFreshnessUnavailable  # noqa: E402
from xorq.common.utils.dasher import HASHER  # noqa: E402
from xorq.common.utils.dasher._relations import _databasetable_dispatcher  # noqa: E402
from xorq.common.utils.redshift_utils import resolve_redshift_schema  # noqa: E402
from xorq.vendor.ibis.expr.datatypes import Unknown  # noqa: E402


SESSION_SCHEMA = "analytics"
TEMP_SCHEMA = "pg_temp_3"
PORT = 5439


class _ConnectionInfo:
    """The psycopg ``info`` surface ``normalize_redshift_backend`` reads."""

    port = PORT

    @staticmethod
    def get_parameters() -> dict[str, str]:
        return {"host": "example.invalid", "dbname": "dev"}


class RecordingConnection:
    """Records every statement and answers only the ones the key may issue.

    Deliberately *not* a mock that accepts anything. A statement this fake does
    not recognise raises, because answering it is how a postgres-only call
    reached a live warehouse before: the fake served ``pg_my_temp_schema()``,
    Redshift does not have it, and every unqualified key failed live.
    """

    def __init__(
        self,
        current_schema: str = SESSION_SCHEMA,
        temp_schema: str | None = None,
        temp_names: tuple[str, ...] = (),
    ) -> None:
        # (sql, params) rather than sql alone, so the name actually BOUND is
        # asserted and not only the statement shape.
        self.calls: list[tuple[str, object]] = []
        self.info = _ConnectionInfo()
        self._current_schema = current_schema
        # The session's ``pg_temp_<N>`` schema and the temp tables in it. None
        # by default, as on a real connection that has created no temp table.
        self._temp_schema = temp_schema
        self._temp_names = temp_names

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
        lowered = self._last.lower()
        if "pg_my_temp_schema" in lowered:
            raise make_undefined_function("pg_my_temp_schema()")
        if "current_schema" not in lowered and "svv_columns" not in lowered:
            raise AssertionError(f"unexpected statement sent to Redshift: {sql!r}")
        return self

    def fetchall(self) -> list[tuple]:
        lowered = self._last.lower()
        if "current_schema" in lowered:
            return [(self._con._current_schema,)]
        # svv_columns, the only other statement execute() lets through. Routed
        # on the name actually BOUND: an open temp schema holding other tables
        # must not make every unqualified name temporary.
        _, params = self._con.calls[-1]
        temp = self._con._temp_schema
        if temp is not None and params["name"] in self._con._temp_names:
            return [(temp,)]
        return []

    def fetchone(self) -> tuple | None:
        rows = self.fetchall()
        return rows[0] if rows else None


def make_undefined_function(name: str) -> Exception:
    """Stands in for ``psycopg.errors.UndefinedFunction``, as Redshift raises it."""
    exc = Exception(f"function {name} does not exist")
    exc.sqlstate = "42883"
    return exc


def make_con(
    current_schema: str = SESSION_SCHEMA,
    temp_schema: str | None = None,
    temp_names: tuple[str, ...] = (),
) -> RedshiftBackend:
    con = RedshiftBackend(host="example.invalid", port=PORT)
    con.con = RecordingConnection(
        current_schema=current_schema,
        temp_schema=temp_schema,
        temp_names=temp_names,
    )
    return con


def make_dt(
    con: RedshiftBackend,
    name: str = "offers",
    database: str | None = "sales",
    schema: sch.Schema | None = None,
) -> ops.DatabaseTable:
    return ops.DatabaseTable(
        name=name,
        schema=schema or sch.Schema({"id": "int64", "amt": "float64"}),
        source=con,
        namespace=ops.Namespace(catalog=None, database=database),
    )


def snapshot_token(dt: ops.DatabaseTable) -> str:
    return SnapshotStrategy().declared_hasher().tokenize(dt)


@pytest.mark.parametrize(
    "database",
    (pytest.param("sales", id="qualified"), pytest.param(None, id="unqualified")),
)
def test_a_freshness_key_is_refused_before_any_statement(database: str | None) -> None:
    """No freshness signal exists, so nothing is read in search of one.

    Refusing before the first round trip is also what keeps the postgres
    probe's ``CHECKPOINT`` and ``ANALYZE`` unreachable from this backend.
    """
    con = make_con()
    with pytest.raises(RedshiftFreshnessUnavailable):
        _databasetable_dispatcher(make_dt(con, database=database))
    assert not con.con.calls


def test_the_freshness_refusal_reaches_the_cache_strategy() -> None:
    """``ParquetCache``'s strategy, not only the rule, must refuse.

    ``ModificationTimeStrategy`` keys on ``expr.ls.tokenized``, the global
    hasher; a refusal the strategy did not reach would leave the actual
    ``.cache()`` path keying on something else.
    """
    expr = make_dt(make_con()).to_expr()
    with pytest.raises(RedshiftFreshnessUnavailable):
        ModificationTimeStrategy().calc_key(expr)


def test_the_freshness_refusal_says_why_and_names_the_cache_that_works() -> None:
    con = make_con()
    with pytest.raises(RedshiftFreshnessUnavailable) as excinfo:
        HASHER.tokenize(make_dt(con))
    message = str(excinfo.value)
    assert "'offers'" in message
    assert "ANALYZE" in message
    assert "ParquetSnapshotCache" in message
    # ``xorq run-cached`` defaults to ParquetCache, so a CLI user hits this
    # refusal first and needs the flag, not the class.
    assert "--cache-type snapshot" in message


def test_the_snapshot_key_computes_through_the_strategy() -> None:
    """The cache the refusal recommends must itself produce a key.

    ``SnapshotStrategy.normalize_backend`` delegates to ``HASHER.normalize`` for
    a backend outside ``NAME_ONLY_BACKEND_NAMES``, which hit dasher's raising
    default until the Redshift backend rule was registered, so the refusal was
    handing the user an action that raised too.
    """
    expr = make_dt(make_con()).to_expr()
    assert SnapshotStrategy().calc_key(expr)
    assert SnapshotStrategy.normalize_backend(make_con())


def test_unqualified_tables_in_different_schemas_get_different_snapshot_keys() -> None:
    """``a.offers`` and ``b.offers`` must never share a snapshot.

    Unqualified, both carry an empty namespace, and the connection identity is
    host, port and database only. Before the resolved schema was added, these
    two tables -- same name, same columns, same cluster, different schemas --
    had one key, and one schema's snapshot was served for the other.
    """
    keys = {
        snapshot_token(make_dt(make_con(current_schema=schema), database=None))
        for schema in ("a", "b")
    }
    assert len(keys) == 2


def test_the_same_unqualified_table_keeps_its_snapshot_key() -> None:
    """Resolution must not make the key vary for one relation."""
    keys = {
        snapshot_token(make_dt(make_con(), database=None)),
        snapshot_token(make_dt(make_con(), database=None)),
    }
    assert len(keys) == 1


def test_a_qualified_table_costs_no_statement_for_its_snapshot_key() -> None:
    """The namespace already names the schema; resolving it again is waste."""
    con = make_con()
    assert snapshot_token(make_dt(con))
    assert not con.con.calls


def test_an_unqualified_table_resolves_the_session_schema() -> None:
    """The session schema, not ``public``, and one temp lookup bound by name."""
    con = make_con()
    assert resolve_redshift_schema(make_dt(con, database=None)) == SESSION_SCHEMA
    assert con.con.params_for("svv_columns") == {"name": "offers"}


def test_a_name_resolving_into_the_session_temp_schema_is_refused() -> None:
    """``table()`` accepts a temp-only name; the snapshot key must not.

    Keying it under ``current_schema()`` would describe a different, permanent
    relation that merely shares the name, and keying the temp table itself
    would describe something invisible to every other session. The refusal
    names the schema, and does not send the user to a snapshot cache that
    refuses the same table.
    """
    con = make_con(temp_schema=TEMP_SCHEMA, temp_names=("offers",))
    with pytest.raises(RedshiftFreshnessUnavailable) as excinfo:
        snapshot_token(make_dt(con, database=None))
    message = str(excinfo.value)
    assert TEMP_SCHEMA in message
    assert "'offers'" in message
    assert "ParquetSnapshotCache" not in message


def test_a_permanent_table_is_unaffected_by_an_open_temp_schema() -> None:
    """Having *a* temp schema must not refuse every unqualified table."""
    con = make_con(temp_schema=TEMP_SCHEMA, temp_names=("staging",))
    assert resolve_redshift_schema(make_dt(con, database=None)) == SESSION_SCHEMA


@pytest.mark.parametrize(
    "temp_names",
    (pytest.param((), id="no-temp-table"), pytest.param(("offers",), id="temp-table")),
)
def test_the_snapshot_key_never_calls_pg_my_temp_schema(
    temp_names: tuple[str, ...],
) -> None:
    """Redshift has no ``pg_my_temp_schema()``; calling it fails every key.

    The inherited postgres ``_session_temp_db`` issues it. The fake raises on
    it as Redshift does, so either outcome here, a key or the temp-table
    refusal, proves the call was never made.
    """
    con = make_con(temp_schema=TEMP_SCHEMA, temp_names=temp_names)
    dt = make_dt(con, database=None)
    if temp_names:
        with pytest.raises(RedshiftFreshnessUnavailable, match=TEMP_SCHEMA):
            snapshot_token(dt)
    else:
        assert snapshot_token(dt)
    assert not any("pg_my_temp_schema" in s.lower() for s in con.con.statements)


@pytest.mark.parametrize(
    "nullable",
    (pytest.param(True, id="nullable"), pytest.param(False, id="not-null")),
)
def test_a_table_with_an_unmappable_column_gets_a_stable_snapshot_key(
    nullable: bool,
) -> None:
    """A Redshift table binds even when a column has no xorq type (SUPER, say).

    Such a column sits in the schema as ``Unknown``, and the schema is part of
    the key, so tokenizing must accept it and give the same answer every time.
    """
    schema = sch.Schema({"id": "int64", "payload": Unknown(nullable=nullable)})
    tokens = {snapshot_token(make_dt(make_con(), schema=schema)) for _ in range(2)}
    assert len(tokens) == 1


def test_an_unmappable_column_does_not_share_a_snapshot_key_with_a_mapped_one() -> None:
    """Two tables that differ only in whether a column could be mapped differ."""
    keys = {
        snapshot_token(
            make_dt(make_con(), schema=sch.Schema({"id": "int64", "payload": dtype}))
        )
        for dtype in (Unknown(), Unknown(nullable=False), "string")
    }
    assert len(keys) == 3
