"""Offline tests for the Redshift backend.

Sited here, not under ``python/xorq/backends/redshift/tests/``, on purpose.
``xorq/backends/conftest.py`` auto-applies ``pytest.mark.<backend>`` by path and
adds ``core`` only outside ``backends/``, and every CI job selects by marker, so
a test placed under the backend directory would be Redshift-marked and would
run only in the credential-gated workflow. These need no credentials and should
run in the default sweep.

Every trap these cover is silent: each one produces a working-looking backend
that is wrong, so the assertions are on the specific observable, not on
"it connects".
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from types import ModuleType

import pyarrow as pa
import pytest
import sqlglot as sg
import sqlglot.expressions as sge


# Must run BEFORE the xorq imports below. Neither driver named here is a core
# dependency -- each ships only in an extra -- and each is imported unguarded
# at module scope on the path ``import xorq.backends.redshift`` takes, so
# neither can be deferred to the tests that care:
#
#   adbc_driver_manager   backends/postgres/__init__.py:12
#   psycopg               vendor/ibis/backends/postgres/__init__.py:11
#
# Measured one name at a time: blocking either one alone breaks the import.
# ``adbc_driver_postgresql`` is a third driver on the same family tree and is
# deliberately NOT guarded here. Nothing above reaches it: the only importer
# is ``xorq.common.utils.postgres_utils``, which five tests below need and the
# ``postgres_utils`` fixture imports for them. Two further tests need it
# merely *installed*, for the probe's own ``find_spec``, and guard themselves
# with ``importorskip``. Six tests, rather than the whole module.
#
# CI selects by marker with no path filter, so every job COLLECTS this file;
# without the guard the jobs lacking the extras failed collection outright
# rather than deselecting -- a red build reading ModuleNotFoundError, not a
# skip. ``backends/conftest.py`` guards ``backends/<name>/`` paths only, and
# this file is deliberately sited outside them (see the docstring above), so
# the guard has to be here. The E402s are that guard running first, not import
# sloppiness -- and the two blank lines above this comment are load-bearing:
# with one, ruff raises I001 and ``--fix`` hoists the imports back above the
# guard. Same shape as ``python/xorq/tests/test_redshift_cache_freshness.py``,
# which arrives with PR #2335.
pytest.importorskip("adbc_driver_manager")
psycopg = pytest.importorskip("psycopg")

import xorq  # noqa: E402
import xorq.api as xo  # noqa: E402
import xorq.backends.redshift as redshift_module  # noqa: E402
import xorq.common.exceptions as exc  # noqa: E402
import xorq.vendor.ibis.expr.datatypes as dt  # noqa: E402
from xorq.backends.postgres import Backend as PostgresBackend  # noqa: E402
from xorq.backends.redshift import Backend as RedshiftBackend  # noqa: E402
from xorq.vendor.ibis.backends.profiles import (  # noqa: E402
    Profile,
    check_for_exposed_secrets,
    con_name_to_secret_keys,
)


def test_name_is_redshift_not_inherited_postgres():
    """A subclass inherits ``name = "postgres"``, which would make every
    Redshift profile serialize as postgres."""
    assert RedshiftBackend.name == "redshift"


def test_profile_from_con_records_redshift():
    """The observable that actually matters for the name.

    ``Profile.from_con`` keys on ``con.name``. The secret-key mirror test
    resolves via entry point rather than ``.name``, so it passes even when the
    name is wrong -- this is the assertion that does not.
    """
    con = RedshiftBackend()
    assert con._profile.con_name == "redshift"


def test_secret_keys_match_postgres_and_the_mirror():
    """Declaring nothing would not skip the mirror test -- ``getattr`` finds
    the inherited tuple -- and declaring ``()`` would narrow the exposed-secret
    check to just ``password``."""
    assert tuple(RedshiftBackend._secret_keys) == tuple(PostgresBackend._secret_keys)
    assert tuple(con_name_to_secret_keys["redshift"]) == tuple(
        RedshiftBackend._secret_keys
    )


def test_exposed_secret_check_is_not_narrowed():
    """``sslkey``/``passfile`` must raise, not just ``password``."""
    for key in RedshiftBackend._secret_keys:
        try:
            check_for_exposed_secrets("redshift", {key: "a-literal-value"})
        except ValueError:
            continue
        raise AssertionError(f"{key} is declared secret but was not caught")


def test_top_level_methods_are_not_inherited():
    """The postgres backend exposes ``connect_env`` (backed by PostgresConfig)
    and ``connect_examples`` (hardcoded to a public postgres host). Inheriting
    them would put a method on the Redshift namespace that does not connect to
    Redshift."""
    assert PostgresBackend._top_level_methods == ("connect_examples", "connect_env")
    assert RedshiftBackend._top_level_methods == ()


def test_api_namespace_exposes_no_postgres_connect_helpers():
    """The class attribute is only half of trap 5.

    ``_top_level_methods`` is surfaced on the backend namespace by
    ``xorq.api.__getattr__``, so this is the observable a user would actually
    hit: ``xo.redshift.connect_env`` must not exist, because it is backed by
    ``PostgresConfig`` and would connect to postgres, and
    ``xo.redshift.connect_examples`` must not exist, because it is hardcoded to
    a public postgres host.
    """
    assert hasattr(xo.postgres, "connect_env")
    assert hasattr(xo.postgres, "connect_examples")

    assert hasattr(xo.redshift, "connect")
    assert not hasattr(xo.redshift, "connect_env")
    assert not hasattr(xo.redshift, "connect_examples")


def test_plain_xorq_import_does_not_expose_the_backend():
    """The proxy is ``xorq.api``; plain ``import xorq`` raises, as it does for
    every other backend."""
    with pytest.raises(AttributeError):
        xorq.redshift


def test_current_schema_is_called_with_parentheses():
    """Redshift rejects bare ``CURRENT_SCHEMA`` with ``UndefinedColumn``.

    Asserted on emitted SQL rather than by executing, because the failure is a
    *server-side* error on SQL that compiles cleanly -- and on the SQL *this
    override* emits rather than on how sqlglot renders
    ``sg.func("current_schema")``. That rendering is third-party behaviour and
    it changed inside the range this project declares it supports: measured,
    ``sg.func("current_schema")`` renders ``SELECT CURRENT_SCHEMA()`` at
    sqlglot 23.6.3 -- the floor ``uv lock --resolution lowest-direct`` picks
    under ``sqlglot>=23.4`` -- and ``SELECT CURRENT_SCHEMA`` at 28.6.0, under
    the Postgres and Redshift dialects alike. Asserting the bare form pinned
    the whole lowest-direct matrix to one sqlglot. ``Anonymous`` parenthesises
    at both versions and under every dialect measured, so the property below
    is version-independent.

    Still the negative control it was written to be, and on the version that
    matters: delete the override and the inherited implementation at
    ``vendor/ibis/backends/postgres/__init__.py:401`` runs, emitting
    ``SELECT CURRENT_SCHEMA`` under any sqlglot new enough to have dropped the
    parentheses, and this fails. Under one old enough to keep them it does not
    -- because there the override is genuinely redundant and there is nothing
    for a test to detect.
    """
    con = make_offline_con()
    con.con = _FakeConnection(rows=[("public",)])

    assert con.current_database == "public"
    assert executed(con) == ["SELECT CURRENT_SCHEMA()"]


def test_current_catalog_needs_no_override():
    """``CURRENT_DATABASE()`` already renders parenthesised, so only
    ``current_database`` (which selects the *schema*) needed overriding."""
    dialect = RedshiftBackend.compiler.dialect
    assert (
        sg.select(sg.func("current_database")).sql(dialect)
        == "SELECT CURRENT_DATABASE()"
    )


def test_client_encoding_defaults_without_entering_the_build_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``client_encoding`` is mandatory -- Redshift reports the PG 8.x alias
    ``UNICODE``, absent from psycopg3's codec map, so every query otherwise
    raises ``NotSupportedError``.

    It is defaulted inside ``do_connect`` rather than by the caller precisely
    so it stays out of ``_con_kwargs``, which is captured from the caller's
    arguments and feeds the build hash.

    ``_con_kwargs`` alone cannot see the first half: it is populated by
    ``BaseBackend.__init__``, so it holds with ``do_connect`` deleted. The
    kwarg is caught where it lands, at ``psycopg.connect``.
    """
    recorded = {}

    def fake_connect(**kwargs):
        recorded.update(kwargs)
        return _FakeConnection()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(RedshiftBackend, "_post_connect", lambda self: None)

    con = RedshiftBackend()
    con.do_connect(host="example.invalid", user="u", password="p", database="d")

    # Reached the driver ...
    assert recorded["client_encoding"] == "utf8"
    assert recorded["port"] == redshift_module.DEFAULT_PORT
    # ... and did not reach the build hash.
    assert "client_encoding" not in con._con_kwargs


def test_client_encoding_is_not_inherited_from_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative control for the test above: the postgres backend must
    *not* set it, or the Redshift override would be indistinguishable from
    doing nothing."""
    recorded = {}

    def fake_connect(**kwargs):
        recorded.update(kwargs)
        return _FakeConnection()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(PostgresBackend, "_post_connect", lambda self: None)

    con = PostgresBackend()
    con.do_connect(host="example.invalid", user="u", password="p", database="d")

    assert "client_encoding" not in recorded


def test_default_port_is_redshifts():
    assert redshift_module.DEFAULT_PORT == 5439
    defaults = dict(
        zip(
            RedshiftBackend.do_connect.__code__.co_varnames[1:],
            RedshiftBackend.do_connect.__defaults__,
        )
    )
    assert defaults["port"] == redshift_module.DEFAULT_PORT


def test_profile_roundtrips():
    con = RedshiftBackend()
    type(con).__init__(con, host="example.invalid", port=redshift_module.DEFAULT_PORT)
    restored = Profile(**con._profile.as_dict())
    assert restored.con_name == "redshift"
    assert restored.hash_name == con._profile.hash_name


# ---------------------------------------------------------------------------
# Ingest dispatch: psycopg baseline, ADBC accelerator.
#
# The postgres backend's ``read_record_batches`` is unconditional ADBC, and
# inheriting it made this backend claim a psycopg baseline it did not have.
# These tests are all offline: they drive the real SQL generation against a
# recording connection, because the failure being guarded against is *which
# statements get emitted*, not whether a socket opens.
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Records executed SQL. Mimics psycopg3's chaining ``execute``."""

    def __init__(self, log: list, rows: tuple = ()) -> None:
        self.log = log
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, sql, *args, **kwargs):
        self.log.append(("execute", sql))
        return self

    def executemany(self, sql, rows):
        self.log.append(("executemany", sql, list(rows)))
        return self

    def fetchall(self) -> list:
        return list(self.rows)


class _FakeConnection:
    def __init__(self, rows: tuple = ()) -> None:
        self.log: list = []
        self.rows = rows

    def cursor(self, *args, **kwargs):
        return _FakeCursor(self.log, self.rows)

    def transaction(self):
        return contextlib.nullcontext()


def make_offline_con(**con_kwargs):
    """A Redshift backend with ``_con_kwargs`` populated but nothing dialled.

    ``type(con).__init__`` is the same idiom the profile tests above use: it
    runs ``BaseBackend.__init__``, which captures ``_con_kwargs`` and builds
    the profile, without ``do_connect``.
    """
    con = RedshiftBackend()
    type(con).__init__(
        con, host="example.invalid", port=redshift_module.DEFAULT_PORT, **con_kwargs
    )
    con.con = _FakeConnection()
    con.table = lambda name: ("table", name)
    return con


def make_reader(*batches, schema=None):
    if schema is None:
        schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    return pa.RecordBatchReader.from_batches(
        schema, [pa.RecordBatch.from_pydict(batch, schema=schema) for batch in batches]
    )


def executed(con):
    return [sql for (kind, sql, *_) in con.con.log if kind == "execute"]


def raise_get_conn(self, **kwargs):
    raise RuntimeError("FATAL: password authentication failed for user")


def test_psycopg_ingest_creates_and_inserts():
    """The baseline the ADR promises and the inherited method did not provide.

    ``INSERT`` rather than ``COPY`` is not a shortcut: Redshift has no
    ``COPY ... FROM STDIN``, so a ``COPY`` baseline would need an S3 bucket and
    an assumable role, which is the deferred ``redshift.ingest.bucket`` work.
    """
    con = make_offline_con()

    result = con.read_record_batches(
        make_reader({"a": [1, 2], "b": ["x", "y"]}), table_name="t"
    )

    assert con.con.log == [
        ("execute", 'CREATE TABLE "t" ("a" BIGINT, "b" VARCHAR)'),
        (
            "executemany",
            'INSERT INTO "t" ("a", "b") VALUES (%s, %s)',
            [(1, "x"), (2, "y")],
        ),
    ]
    assert result == ("table", "t")


def test_psycopg_ingest_consumes_every_batch():
    """A reader is a stream, and the obvious wrong implementation -- reading
    ``next(reader)`` or materialising ``.read_all()`` into one statement --
    silently drops or reshapes rows."""
    con = make_offline_con()

    con.read_record_batches(
        make_reader(
            {"a": [1], "b": ["x"]},
            {"a": [2, 3], "b": ["y", "z"]},
        ),
        table_name="t",
    )

    inserted = [entry[2] for entry in con.con.log if entry[0] == "executemany"]
    assert inserted == [[(1, "x")], [(2, "y"), (3, "z")]]


def test_psycopg_ingest_of_an_empty_batch_still_creates_the_table():
    """``executemany`` with no rows is skipped, but the schema still lands --
    an empty parquet file must produce an empty table, not no table."""
    con = make_offline_con()

    con.read_record_batches(make_reader({"a": [], "b": []}), table_name="t")

    assert executed(con) == ['CREATE TABLE "t" ("a" BIGINT, "b" VARCHAR)']
    assert not [entry for entry in con.con.log if entry[0] == "executemany"]


def test_psycopg_ingest_accepts_a_table_like_the_adbc_branch_does():
    """``adbc_ingest`` takes a ``pa.Table``, and iterating one yields *columns*
    -- so the naive psycopg loop would fail on a missing ``num_rows`` for an
    input the accelerator handles. Which branch runs has to stay an
    implementation detail."""
    con = make_offline_con()

    con.read_record_batches(pa.table({"a": [1], "b": ["x"]}), table_name="t")

    assert con.con.log == [
        ("execute", 'CREATE TABLE "t" ("a" BIGINT, "b" VARCHAR)'),
        ("executemany", 'INSERT INTO "t" ("a", "b") VALUES (%s, %s)', [(1, "x")]),
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("create", ['CREATE TABLE "t" ("a" BIGINT, "b" VARCHAR)']),
        ("append", []),
        (
            "replace",
            [
                'DROP TABLE IF EXISTS "t"',
                'CREATE TABLE "t" ("a" BIGINT, "b" VARCHAR)',
            ],
        ),
        (
            "create_append",
            ['CREATE TABLE IF NOT EXISTS "t" ("a" BIGINT, "b" VARCHAR)'],
        ),
    ],
)
def test_psycopg_ingest_modes_match_their_adbc_meanings(mode, expected):
    """Which branch runs has to stay an implementation detail, and it stops
    being one the moment the two disagree about what ``mode`` means:
    ``append`` must not create, ``create`` must not tolerate an existing table,
    ``replace`` must drop it, ``create_append`` must tolerate it."""
    con = make_offline_con()

    con.read_record_batches(
        make_reader({"a": [1], "b": ["x"]}), table_name="t", mode=mode
    )

    assert executed(con) == expected


def test_psycopg_ingest_creates_the_temp_table_directly():
    """The ADBC path creates a permanent table and converts it afterwards with
    ``make_table_temporary``. That is not overhead ADBC failed to avoid -- it
    connects separately, so a temp table created there would be invisible.
    Sharing the psycopg connection is what makes the direct form correct, so
    assert no rename-and-copy appears."""
    con = make_offline_con()

    con.read_record_batches(
        make_reader({"a": [1], "b": ["x"]}), table_name="t", temporary=True
    )

    assert executed(con) == ['CREATE TEMPORARY TABLE "t" ("a" BIGINT, "b" VARCHAR)']


def test_ingest_never_dispatches_to_adbc_even_when_it_is_available(monkeypatch):
    """The test this replaces asserted the opposite, and the opposite was wrong.

    Neither ADBC driver can ingest into Redshift -- both ingest by ``COPY``,
    and Redshift's ``COPY`` reads from S3 only -- so a driver that is installed
    AND credentialed must change nothing here. ``_adbc_unavailable_reason()``
    answering ``None`` is the case that used to select the branch that cannot
    run; this pins that it no longer selects anything.

    Every other ingest test describes the psycopg branch and would keep passing
    if a dispatch were reintroduced, so this is the only one that would fail.
    """
    con = make_offline_con(password="static")
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: None)
    monkeypatch.setattr(
        PostgresBackend,
        "read_record_batches",
        lambda *args, **kwargs: pytest.fail("ingest delegated to the ADBC branch"),
    )

    result = con.read_record_batches(
        make_reader({"a": [1], "b": ["x"]}), table_name="t", mode="append"
    )

    assert result == ("table", "t")
    assert con.con.log == [
        ("executemany", 'INSERT INTO "t" ("a", "b") VALUES (%s, %s)', [(1, "x")]),
    ]


def test_ingest_rejects_a_missing_table_name():
    """Inherited, ``table_name=None`` reached ``adbc_ingest`` and failed
    somewhere inside the driver."""
    con = make_offline_con()

    with pytest.raises(ValueError, match="table_name"):
        con.read_record_batches(make_reader({"a": [1], "b": ["x"]}))


def test_ingest_validates_mode_before_issuing_any_sql(monkeypatch):
    """An unknown mode must fail before anything is created or inserted, and
    must fail identically whether or not a driver happens to be installed --
    probed with one available, since that is the case that used to divert."""
    con = make_offline_con(password="static")
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: None)

    with pytest.raises(ValueError, match="mode must be one of"):
        con.read_record_batches(
            make_reader({"a": [1], "b": ["x"]}), table_name="t", mode="upsert"
        )

    assert con.con.log == []


# ---------------------------------------------------------------------------
# Driver availability, and telling driver-absent from auth-failed.
# ---------------------------------------------------------------------------


@pytest.fixture
def postgres_utils() -> ModuleType:
    """``xorq.common.utils.postgres_utils``, imported per test rather than at
    module scope.

    It does ``import adbc_driver_postgresql.dbapi`` on its first line, and that
    driver is in the ``postgres``/``examples`` extras only. Nothing else in
    this file reaches it -- the backend itself needs ``adbc_driver_manager``
    and ``psycopg``, which are guarded at module scope -- so importing it here
    is what keeps the other thirty-odd tests running in jobs that have the
    backend but not this driver.
    """
    return pytest.importorskip("xorq.common.utils.postgres_utils")


def test_adbc_is_unavailable_without_the_driver(monkeypatch):
    con = make_offline_con(password="static")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            None
            if name == "adbc_driver_postgresql"
            else real_find_spec(name, *args, **kwargs)
        ),
    )

    assert "adbc_driver_postgresql" in con._adbc_unavailable_reason()


@pytest.mark.parametrize("con_kwargs", [{}, {"password": None}])
def test_adbc_is_unavailable_without_a_password(con_kwargs):
    """``PgADBC.password`` reads ``_con_kwargs["password"]`` and interpolates
    it into a URI, so both cases disqualify -- and the ``None`` case is the one
    worth pinning: it does not raise, it formats as the literal string
    ``"None"`` and fails later as an auth error against a password nobody
    set."""
    # The probe checks the driver before the password, so with the driver
    # absent this would pass or fail on the wrong clause. Measured: without
    # ``adbc_driver_postgresql`` installed the reason is
    # "adbc_driver_postgresql is not installed" and the assertion below fails.
    pytest.importorskip("adbc_driver_postgresql")

    con = make_offline_con(**con_kwargs)
    assert "password" in con._adbc_unavailable_reason()


def test_adbc_is_available_when_installed_and_credentialed():
    """A ``None`` reason means installed *and* credentialed -- not that the
    driver is known to work against Redshift. Whether
    ``adbc_driver_postgresql`` speaks to Redshift at all is untested and needs
    a live endpoint; see the alternative recorded in ADR-2332."""
    pytest.importorskip("adbc_driver_postgresql")
    con = make_offline_con(password="static")
    assert con._adbc_unavailable_reason() is None


def test_auth_failure_is_not_swallowed_as_a_missing_driver(
    monkeypatch: pytest.MonkeyPatch, postgres_utils: ModuleType
) -> None:
    """The discrimination the inherited ``except Exception`` cannot make.

    A rejected temporary credential and an absent driver arrive at the probe as
    the same exception. Postgres can afford to conflate them -- a static
    password that fails ADBC while psycopg works means a ``.pgpass``
    connection, and falling back is right. Under a rotating IAM credential the
    same silence hides an expired password behind a working query.
    """
    con = make_offline_con(password="rotating")
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: None)
    monkeypatch.setattr(postgres_utils.PgADBC, "get_conn", raise_get_conn)

    with pytest.raises(RuntimeError, match="password authentication failed"):
        con._open_adbc_conn_or_none()


def test_an_unavailable_driver_is_not_dialled_at_all(
    monkeypatch: pytest.MonkeyPatch, postgres_utils: ModuleType
) -> None:
    """Availability is decided from local facts *before* connecting, which is
    what makes the test above possible: every exception from the connect is
    then a real failure."""
    con = make_offline_con()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")
    monkeypatch.setattr(postgres_utils.PgADBC, "get_conn", raise_get_conn)

    assert con._open_adbc_conn_or_none() is None


def test_the_postgres_seam_keeps_swallowing(
    monkeypatch: pytest.MonkeyPatch, postgres_utils: ModuleType
) -> None:
    """The probe was extracted from ``to_pyarrow_batches`` so Redshift could
    override it. Postgres's own behaviour must be unchanged by that -- its
    catch-all is deliberate, and users connecting without a password in
    ``_con_kwargs`` depend on the quiet fallback."""
    con = PostgresBackend()
    type(con).__init__(con, host="example.invalid")
    monkeypatch.setattr(postgres_utils.PgADBC, "get_conn", raise_get_conn)

    assert con._open_adbc_conn_or_none() is None


def test_ingest_modes_are_the_adbc_ingest_modes():
    assert redshift_module.INGEST_MODES == (
        "create",
        "append",
        "replace",
        "create_append",
    )


def test_ingest_ddl_pins_two_unverified_redshift_type_widths():
    """Not a passing feature -- a tripwire on a live-session checklist item.

    The ``CREATE`` is rendered under the postgres dialect, and two of its types
    are documented Redshift divergences that no offline test can settle:

    * bare ``VARCHAR`` is unbounded in PostgreSQL but documented as
      ``VARCHAR(256)`` in Redshift, where ``TEXT`` is also an alias for it --
      so a string longer than 256 would fail the insert, not the create.
    * ``TIMESTAMP(6)`` carries a precision modifier that PostgreSQL accepts and
      Redshift is not documented to.

    Both are *suspected*, on documentation rather than observation. This pins
    what is emitted today so that fixing either is a visible change, and so
    the live session has a checklist entry rather than a discovery.
    """
    con = make_offline_con()

    schema = pa.schema([("s", pa.string()), ("ts", pa.timestamp("us"))])
    con.read_record_batches(
        make_reader({"s": ["x"], "ts": [None]}, schema=schema), table_name="t"
    )

    (create,) = executed(con)
    assert create == 'CREATE TABLE "t" ("s" VARCHAR, "ts" TIMESTAMP(6))'


# ---------------------------------------------------------------------------
# Introspection: ``con.table`` and ``con.sql`` on Redshift.
#
# Both defects are server-side errors on SQL that compiles cleanly under the
# postgres dialect, so -- as with ``CURRENT_SCHEMA`` above -- the assertions
# are on the *statements emitted*, plus on the schema built from a
# Redshift-shaped catalog response. Nothing here dials a warehouse.
# ---------------------------------------------------------------------------


class _IntrospectionCursor(_FakeCursor):
    """``_FakeCursor`` that can also answer a query.

    Records ``params`` alongside the ``sql`` its base class already logs, so a
    test can assert that the schema and table name are *bound*, not
    interpolated into the statement.
    """

    def __init__(
        self,
        con: _IntrospectionConnection,
        rows: tuple,
        description: tuple | None,
        temp_rows: tuple = (),
    ) -> None:
        super().__init__(con.log)
        self._con = con
        self._rows = rows
        self._temp_rows = temp_rows
        self._last = None
        self.description = description

    def execute(
        self, sql: str, params: dict | None = None, **kwargs: object
    ) -> _IntrospectionCursor:
        super().execute(sql)
        self._con.params.append(params)
        self._last = sql
        return self

    def fetchall(self) -> list:
        # The permanent and temporary lookups hit different views, so the fake
        # answers them separately: a test that supplies only ``temp_rows`` must
        # not see them returned for a ``svv_all_columns`` query, or the
        # fallback would look exercised when it was not.
        if "svv_columns" in (self._last or ""):
            return list(self._temp_rows)
        return list(self._rows)


class _IntrospectionConnection(_FakeConnection):
    def __init__(
        self,
        rows: tuple = (),
        description: tuple | None = None,
        temp_rows: tuple = (),
    ) -> None:
        super().__init__()
        self.params = []
        self._rows = rows
        self._temp_rows = temp_rows
        self._description = description

    def cursor(self, *args: object, **kwargs: object) -> _IntrospectionCursor:
        return _IntrospectionCursor(
            self, self._rows, self._description, self._temp_rows
        )


class _FakeColumn:
    """The subset of ``psycopg.Column`` the query path reads.

    ``type_display`` is psycopg's own rendering of the OID *and* its type
    modifier -- ``integer``, ``numeric(8,2)``, ``date[]``. The spellings used
    in these tests are pinned against the real registry by
    ``test_fake_column_type_displays_match_psycopg``, so the fakes cannot drift
    from what a live cursor would report.

    The default mirrors what psycopg does for an OID it cannot name: it renders
    the bare number.
    """

    def __init__(
        self, name: str, type_code: int, type_display: str | None = None
    ) -> None:
        self.name = name
        self.type_code = type_code
        self.type_display = type_display if type_display is not None else str(type_code)


def make_introspection_con(
    rows: tuple = (),
    description: tuple | None = None,
    temp_rows: tuple = (),
) -> RedshiftBackend:
    con = make_offline_con()
    con.con = _IntrospectionConnection(
        rows=rows, description=description, temp_rows=temp_rows
    )
    return con


def issued(con: RedshiftBackend) -> list[str]:
    return executed(con)


def last_call(con: RedshiftBackend) -> tuple[str, dict]:
    """The ``(sql, params)`` of the most recent statement."""
    return (executed(con)[-1], con.con.params[-1])


# Shaped after the worked example in the SVV_ALL_COLUMNS reference. Note that
# ``numeric_precision``/``numeric_scale`` are populated for *integer* columns
# too -- buyerid carries 32/0 -- which is exactly the trap that makes "append
# the precision whenever it is there" produce the unparsable ``integer(32)``.
SVV_ROWS = (
    # column_name, data_type, is_nullable, num_precision, num_scale
    ("buyerid", "integer", "NO", 32, 0),
    ("commission", "numeric", "YES", 8, 2),
    ("dateid", "smallint", "NO", 16, 0),
    ("eventname", "character varying", "YES", None, None),
    ("saletime", "timestamp without time zone", "YES", None, None),
)


def test_get_schema_reads_svv_all_columns_and_never_pg_catalog():
    """The defect: the inherited implementation joins ``pg_catalog.pg_enum``
    purely to label enum columns, and Redshift has no ``pg_enum`` -- so every
    ``con.table`` raises ``UndefinedTable``.

    ``pg_catalog`` as a whole is asserted absent, not just ``pg_enum``: the
    inherited query also reads ``pg_attribute``/``pg_class``/``pg_namespace``,
    and a fix that dropped only the enum arm would still not be reading
    Redshift's own catalog.
    """
    con = make_introspection_con(rows=SVV_ROWS)
    con.get_schema("sales", database="public")

    statements = issued(con)
    assert statements, "no statement was issued at all"
    joined = "\n".join(statements).lower()
    assert "svv_all_columns" in joined
    assert "pg_enum" not in joined
    assert "pg_catalog" not in joined


def test_get_schema_builds_the_schema_the_catalog_describes():
    """Names, types and nullability -- the whole observable of ``con.table``.

    ``is_nullable`` is a ``varchar(3)`` carrying ``YES``/``NO``, not a boolean,
    so a fix that passed it straight through as ``nullable=`` would make every
    column nullable (both strings are truthy) and silently lose every NOT NULL.
    """
    con = make_introspection_con(rows=SVV_ROWS)
    schema = con.get_schema("sales", database="public")

    assert schema.names == (
        "buyerid",
        "commission",
        "dateid",
        "eventname",
        "saletime",
    )
    assert schema["buyerid"] == dt.Int32(nullable=False)
    assert schema["dateid"] == dt.Int16(nullable=False)
    assert schema["eventname"] == dt.String(nullable=True)
    assert schema["saletime"] == dt.Timestamp(scale=6, nullable=True)


def test_get_schema_keeps_decimal_precision_and_scale():
    """``svv_all_columns.data_type`` is the unparameterised name: a
    ``numeric(8,2)`` column reports ``numeric`` with the precision and scale in
    *separate* columns. Reading ``data_type`` alone silently widens every
    decimal to an unconstrained one.
    """
    con = make_introspection_con(rows=SVV_ROWS)
    schema = con.get_schema("sales", database="public")

    assert schema["commission"] == dt.Decimal(precision=8, scale=2, nullable=True)


def test_get_schema_does_not_parameterise_a_type_that_takes_no_modifier():
    """The other half of the same trap. ``buyerid`` is an ``integer`` whose
    ``numeric_precision`` is 32; appending it would build ``integer(32)``,
    which is not a type."""
    con = make_introspection_con(rows=(("buyerid", "integer", "NO", 32, 0),))
    schema = con.get_schema("sales", database="public")

    assert schema["buyerid"] == dt.Int32(nullable=False)


@pytest.mark.parametrize(
    ("flag", "nullable"),
    [
        pytest.param("YES", True, id="upper-yes"),
        pytest.param("yes", True, id="lower-yes"),
        pytest.param("NO", False, id="upper-no"),
        pytest.param("no", False, id="lower-no"),
        pytest.param(" ", True, id="blank-means-no-information"),
        pytest.param("", True, id="empty-means-no-information"),
    ],
)
def test_get_schema_reads_is_nullable_case_insensitively(
    flag: str, nullable: bool
) -> None:
    """The reference documents the values as lowercase ``yes``/``no`` and its
    own worked example prints them uppercase. Neither is worth betting a
    nullability flag on.

    The blank cases are the third documented value, not defensive padding: the
    ``SVV_REDSHIFT_COLUMNS`` reference gives the possible values as ``yes``,
    ``no`` and ``" "`` -- "no information" -- which external and datashare rows
    carry. Deciding that case by asking ``== "yes"`` answers ``NOT NULL``, and a
    column wrongly marked ``NOT NULL`` makes the first batch carrying a null
    fail the pyarrow cast in ``project_and_cast_reader``. Unknown has to widen.
    """
    con = make_introspection_con(rows=(("c", "integer", flag, 32, 0),))
    schema = con.get_schema("t", database="public")

    assert schema["c"].nullable is nullable


def test_get_schema_binds_the_schema_and_table_rather_than_interpolating():
    con = make_introspection_con(rows=SVV_ROWS)
    con.get_schema("sales", database="analytics")

    (sql, params) = last_call(con)
    assert "sales" not in sql
    assert "analytics" not in sql
    assert params["table"] == "sales"
    assert params["schema"] == "analytics"


def test_get_schema_emits_no_array_predicate():
    """The inherited query filters with ``n.nspname = ANY(%(dbs)s)``. Redshift
    has no array type, so the array form has to go even though it compiles."""
    con = make_introspection_con(rows=SVV_ROWS)
    con.get_schema("sales", database="public")

    (sql, _params) = last_call(con)
    assert "ANY(" not in sql.upper()
    assert "ARRAY" not in sql.upper()


def test_get_schema_scopes_by_catalog_when_one_is_given():
    """``svv_all_columns`` spans databases, so an unscoped query can match a
    same-named table in another one."""
    con = make_introspection_con(rows=SVV_ROWS)
    con.get_schema("sales", catalog="warehouse", database="public")

    (sql, params) = last_call(con)
    assert "database_name" in sql
    assert params["catalog"] == "warehouse"


def test_get_schema_scopes_by_the_current_catalog_when_none_is_given() -> None:
    """The predicate is unconditional, and the no-catalog case is the one that
    matters.

    ``catalog`` is ``None`` on every ordinary ``con.table("t")`` and
    ``con.table("t", database="s")`` -- only a 2-tuple or a dotted three-part
    name ever supplies one -- so scoping *only* when one was passed left the
    common path unscoped. ``svv_all_columns`` is documented to include the
    columns from datashares provided by remote clusters, so an unscoped lookup
    can match ``public.sales`` in two databases at once; ``ORDER BY
    ordinal_position`` has no tiebreaker across them, so the rows interleave and
    same-named columns overwrite each other, while the compiled query still
    reads from the *current* database. The result is a schema describing a
    different table than the one queried, with no error anywhere.
    """
    con = make_introspection_con(rows=SVV_ROWS)
    con.get_schema("sales", database="public")

    (sql, params) = last_call(con)
    assert "database_name" in sql
    # The default is applied server-side, so the bound value is None and the
    # statement carries the fallback. Reading current_database() in Python
    # first would be two extra round trips per con.table().
    assert params["catalog"] is None
    assert "COALESCE(%(catalog)s, current_database())" in sql


def test_get_schema_raises_rather_than_collapsing_duplicate_column_names() -> None:
    """Last-write-wins on a duplicate name is how an unscoped lookup lost a
    column silently. The scoping above is the fix; this is the backstop, and it
    has to be loud rather than quietly short a column."""
    rows = (
        ("id", "integer", "NO", 32, 0),
        ("id", "character varying", "YES", None, None),
    )
    con = make_introspection_con(rows=rows)

    with pytest.raises(exc.IntegrityError, match="id"):
        con.get_schema("sales", database="public")


def test_get_schema_defaults_the_schema_to_the_current_one() -> None:
    """The customer's failure reproduced on a plain single-schema
    ``con.table("t")``, so the no-``database`` path is the one that has to
    work, not only the three-part-named one."""
    con = make_introspection_con(rows=SVV_ROWS)
    con.get_schema("sales")

    (sql, params) = last_call(con)
    assert params["schema"] is None
    assert "COALESCE(%(schema)s, current_schema())" in sql


def test_get_schema_raises_table_not_found_for_a_missing_table():
    """An empty catalog response is an absent table, not a table with no
    columns. Returning an empty schema would turn a typo into a query that
    compiles and returns nothing."""
    con = make_introspection_con(rows=())

    with pytest.raises(exc.TableNotFound):
        con.get_schema("nope", database="public")


def test_get_schema_sql_parses_and_selects_from_svv_all_columns():
    """Read the statement, not just substrings of it.

    A structurally invalid query can still contain every string the assertions
    above look for, so this reparses what was emitted and checks the shape:
    one SELECT, from ``svv_all_columns``, ordered by ``ordinal_position``.
    """
    con = make_introspection_con(rows=SVV_ROWS)
    con.get_schema("sales", database="public")

    (sql, _params) = last_call(con)
    parsed = sg.parse_one(sql, read=con.dialect)

    assert isinstance(parsed, sge.Select)
    assert parsed.find(sge.Table).name == "svv_all_columns"
    assert "ordinal_position" in parsed.args["order"].sql().lower()


# --- con.sql: the temporary-view defect ------------------------------------


def test_get_schema_using_query_issues_no_ddl_at_all():
    """The defect: the inherited implementation infers a query's schema by
    building a ``CREATE TEMPORARY VIEW``. Redshift has temporary *tables* but
    not temporary *views*, so ``con.sql`` dies with a syntax error at ``VIEW``.

    The exception is suppressed so this fails on the *emitted DDL* rather than
    on whatever the fake connection makes of it; the ``statements`` assertion
    keeps it from passing vacuously when nothing runs.
    """
    con = make_introspection_con(description=(_FakeColumn("a", 23, "int4"),))
    with contextlib.suppress(Exception):
        con._get_schema_using_query("SELECT 1 AS a")

    statements = issued(con)
    assert statements, "no statement was issued at all"
    upper = "\n".join(statements).upper()
    assert "CREATE" not in upper
    assert "VIEW" not in upper
    assert "DROP" not in upper


def test_get_schema_using_query_builds_the_schema_from_the_cursor():
    """No DDL means the schema has to come from the result description.

    ``type_code`` is a PostgreSQL type OID -- Redshift is a PostgreSQL 8.0
    derivative and reports the same OIDs for the standard types -- so this
    reaches the same ``type_mapper`` the catalog path uses by another route.
    """
    con = make_introspection_con(
        description=(
            _FakeColumn("a", 23, "int4"),
            _FakeColumn("b", 1043, "varchar"),
            _FakeColumn("c", 1700, "numeric(8,2)"),
            _FakeColumn("d", 1114, "timestamp"),
        )
    )
    schema = con._get_schema_using_query("SELECT a, b, c, d FROM t")

    assert schema.names == ("a", "b", "c", "d")
    assert schema["a"] == dt.Int32(nullable=True)
    assert schema["b"] == dt.String(nullable=True)
    assert schema["c"] == dt.Decimal(precision=8, scale=2, nullable=True)
    assert schema["d"] == dt.Timestamp(scale=6, nullable=True)


def test_get_schema_using_query_bounds_the_probe_to_no_rows():
    """``cursor.execute`` buffers the whole result client-side, so the probe
    has to return nothing -- otherwise introspecting a query scans the table
    it selects from."""
    con = make_introspection_con(description=(_FakeColumn("a", 23, "int4"),))
    con._get_schema_using_query("SELECT a FROM big")

    (sql,) = issued(con)
    parsed = sg.parse_one(sql, read=con.dialect)
    assert parsed.args["limit"].expression.this == "0"
    assert parsed.find(sge.Subquery).alias == "redshift_probe"


def test_get_schema_using_query_wraps_rather_than_appends():
    """Appending ``LIMIT 0`` to a query that already ends in a ``LIMIT`` would
    be a syntax error, and appending it to a ``UNION`` would bind to the last
    branch only. Wrapping is what makes the probe total."""
    con = make_introspection_con(description=(_FakeColumn("a", 23, "int4"),))
    con._get_schema_using_query("SELECT a FROM t LIMIT 5")

    (sql,) = issued(con)
    parsed = sg.parse_one(sql, read=con.dialect)

    # The outer LIMIT is the probe's; the inner one is the caller's, still
    # bound to the subquery rather than replaced or hoisted.
    assert parsed.args["limit"].expression.this == "0"
    inner = parsed.find(sge.Subquery).this
    assert inner.args["limit"].expression.this == "5"


def test_get_schema_using_query_rejects_an_unmappable_oid_loudly():
    """Redshift's own types (SUPER, VARBYTE, GEOMETRY) carry OIDs psycopg does
    not know. Mapping them to ``unknown`` would hand back a schema that looks
    fine and is wrong; the OID is named so a live session can add it."""
    con = make_introspection_con(description=(_FakeColumn("s", 4000),))

    with pytest.raises(exc.UnsupportedBackendType, match="4000"):
        con._get_schema_using_query("SELECT s FROM t")


def test_neither_introspection_path_creates_a_temporary_view():
    """The single statement the customer transcript shows failing, asserted
    across both entry points at once."""
    table_con = make_introspection_con(rows=SVV_ROWS)
    table_con.get_schema("sales", database="public")

    query_con = make_introspection_con(description=(_FakeColumn("a", 23, "int4"),))
    query_con._get_schema_using_query("SELECT 1 AS a")

    for con in (table_con, query_con):
        for sql in issued(con):
            assert "TEMPORARY VIEW" not in sql.upper()


def test_postgres_introspection_is_left_alone():
    """The overrides are Redshift's. Postgres still reads ``pg_catalog`` and
    still uses a temporary view -- both are correct there, and a change to the
    shared implementation would reach every postgres user."""
    assert PostgresBackend.get_schema is not RedshiftBackend.get_schema
    assert (
        PostgresBackend._get_schema_using_query
        is not RedshiftBackend._get_schema_using_query
    )


# --- the temporary-table path ----------------------------------------------


def test_get_schema_reads_a_temporary_table_from_svv_columns() -> None:
    """The defect this covers is in *this backend's own ingest path*.

    ``read_parquet``/``read_csv``/``read_record_batches`` with no
    ``table_name`` force ``temporary=True``, create a ``TEMPORARY`` table and
    end in ``self.table(name)``, which forwards ``catalog=None,
    database=None``. The inherited postgres ``get_schema`` covered that by
    folding ``_session_temp_db`` into its schema list; dropping the fold made
    every temporary ingest raise ``TableNotFound`` *after* writing the data.

    Restoring the fold could never have worked: ``svv_all_columns`` does not
    list temporary tables at all (measured live -- 0 rows for one that
    exists), so no ``schema_name`` predicate finds them. ``svv_columns`` does,
    in a ``pg_temp_<N>`` schema and with ``is_nullable`` intact, which is why
    this path keeps ``NOT NULL`` where a ``cursor.description`` probe would
    widen every column.
    """
    con = make_introspection_con(
        rows=(),
        temp_rows=(
            ("a", "integer", "NO", 32, 0),
            ("b", "character varying", "YES", None, None),
        ),
    )
    schema = con.get_schema("xorq_temp_abc123")

    assert schema.names == ("a", "b")
    assert schema["a"] == dt.Int32(nullable=False)

    statements = issued(con)
    assert any("svv_all_columns" in sql for sql in statements)
    (temp_sql,) = [sql for sql in statements if "svv_columns" in sql]
    # The scoping is the whole point: without it the fallback means "anything
    # this session can see" rather than "a temporary table".
    assert "pg^_temp^_%" in temp_sql
    assert "ESCAPE '^'" in temp_sql


def test_get_schema_does_not_resolve_through_search_path() -> None:
    """The previous fallback was ``SELECT * FROM "t" LIMIT 0``, which resolves
    through ``search_path`` -- on Redshift ``"$user", public`` by default.

    An unqualified ``con.table("t")`` that missed the catalog would then find a
    *permanent* ``alice.t`` and report it with every column widened to
    nullable: a different table than the caller named, silently, and the exact
    outcome ``get_schema``'s own docstring says the design trades away. The
    fallback must consult a catalog scoped to temp schemas, never the session's
    name resolution.
    """
    con = make_introspection_con(rows=(), temp_rows=())

    with pytest.raises(exc.TableNotFound):
        con.get_schema("t")

    for sql in issued(con):
        assert "LIMIT 0" not in sql
        assert "redshift_probe" not in sql


def test_get_schema_does_not_look_for_a_temp_table_when_the_lookup_was_explicit() -> (
    None
):
    """A lookup that named a schema and got nothing back means that table is
    not there. Temporary tables have no addressable schema name to pass."""
    con = make_introspection_con(rows=(), temp_rows=(("a", "integer", "NO", 32, 0),))

    with pytest.raises(exc.TableNotFound):
        con.get_schema("nope", database="public")

    assert not any("svv_columns" in sql for sql in issued(con))


# --- types neither path may quietly get wrong -------------------------------


def test_both_paths_agree_on_a_char_column() -> None:
    """``CHAR(n)`` is reported as ``character`` by the catalog and as
    ``bpchar`` -- OID 1042 -- by a result description. sqlglot's postgres
    dialect parses the first and not the second, so before the alias the same
    column was ``string`` through ``con.table`` and ``unknown`` through
    ``con.sql``: a mismatch that surfaces only when an operation touches the
    column, with an error naming neither Redshift nor ``CHAR``.
    """
    catalog_con = make_introspection_con(
        rows=(("code", "character", "NO", None, None),)
    )
    query_con = make_introspection_con(
        description=(_FakeColumn("code", 1042, "bpchar(3)"),)
    )

    from_catalog = catalog_con.get_schema("dim", database="public")["code"]
    from_query = query_con._get_schema_using_query("SELECT code FROM dim")["code"]

    assert from_catalog == dt.String(nullable=False)
    assert from_query == dt.String(nullable=True)


@pytest.mark.parametrize(
    "data_type",
    [
        pytest.param("super", id="super"),
        # The spelling svv_all_columns actually emits for VARBYTE(16),
        # measured live: the view definition rewrites the DDL keyword.
        pytest.param("binary varying", id="binary-varying"),
        pytest.param("varbyte", id="varbyte-as-written-by-a-user"),
        pytest.param("hllsketch", id="hllsketch"),
        pytest.param("geometry", id="geometry"),
        pytest.param("geography", id="geography"),
    ],
)
def test_get_schema_rejects_a_redshift_only_type_loudly(data_type: str) -> None:
    """The catalog path had no unsupported-type guard at all, while the query
    path raised -- opposite policies from two entry points onto one table.

    ``geometry`` and ``geography`` are the reason this cannot be left to a
    generic "unknown" check: they *parse*, into ibis ``GeoSpatial`` types, and
    this backend still compiles as PostgreSQL, so a geo operation on such a
    column emits a PostGIS call Redshift does not implement -- a server-side
    error on SQL that compiled cleanly, which is the failure class these
    overrides exist to remove.
    """
    con = make_introspection_con(rows=(("c", data_type, "NO", None, None),))

    with pytest.raises(exc.UnsupportedBackendType, match=data_type):
        con.get_schema("t", database="public")


def test_query_path_rejects_an_oid_psycopg_names_but_cannot_map() -> None:
    """The guard tested only ``info is None`` -- whether psycopg could *name*
    the OID -- and not whether the type mapper could use the name.

    ``oid`` (26) and its ``regclass``/``regproc`` relatives are named and make
    the upstream mapper raise a bare ``AttributeError: 'str' object has no
    attribute 'name'`` from inside ``to_ibis``. Redshift does expose
    ``pg_catalog`` views, so ``SELECT oid FROM pg_class`` is reachable.
    """
    con = make_introspection_con(description=(_FakeColumn("oid", 26, "oid"),))

    with pytest.raises(exc.UnsupportedBackendType, match="oid"):
        con._get_schema_using_query("SELECT oid FROM pg_class")


def test_unsupported_type_keeps_the_nullability_it_was_given() -> None:
    """``SqlglotType.from_string`` drops ``nullable=`` on its ``dt.unknown``
    fallback, so a ``SUPER NOT NULL`` column used to come back with the wrong
    type *and* the wrong nullability. Nothing may reach that branch now."""
    mapper = RedshiftBackend.compiler.type_mapper

    assert mapper.from_string("integer", nullable=False) == dt.Int32(nullable=False)
    assert mapper.from_string("integer", nullable=True) == dt.Int32(nullable=True)
    with pytest.raises(exc.UnsupportedBackendType):
        mapper.from_string("super", nullable=False)


def test_fake_column_type_displays_match_psycopg() -> None:
    """Pins the fakes above against psycopg's real registry.

    ``_FakeColumn`` takes ``type_display`` as a literal, which is only safe
    while those literals are what a live cursor would actually report.
    """
    psycopg = pytest.importorskip("psycopg")

    expected = {
        (23, -1): "int4",
        (1043, -1): "varchar",
        (1700, ((8 << 16) | 2) + 4): "numeric(8,2)",
        (1114, -1): "timestamp",
        (1042, 3 + 4): "bpchar(3)",
        (26, -1): "oid",
        # The array case, and the reason ``type_display`` replaced
        # ``info.name``: the latter renders OID 1182 as ``date``, which
        # would silently turn every array column into its element type.
        (1182, -1): "date[]",
    }
    for (oid, fmod), display in expected.items():
        info = psycopg.postgres.types.get(oid)
        assert info is not None, oid
        assert info.get_type_display(oid=oid, fmod=fmod) == display


# --- statement forms the probe has to survive -------------------------------


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("VALUES (1, 2)", id="values"),
        pytest.param("TABLE t", id="table"),
        pytest.param("", id="empty"),
        pytest.param("-- just a comment", id="comment-only"),
    ],
)
def test_get_schema_using_query_refuses_statements_with_no_result_shape(
    query: str,
) -> None:
    """These parse to something that is not ``sge.Query``, and an earlier
    version wrapped them by hand so ``.subquery()`` would not raise
    ``AttributeError``.

    That was effort spent on statements Redshift cannot run: measured live,
    ``VALUES (1, 2)``, ``SELECT * FROM (VALUES (1, 2)) AS p LIMIT 0``,
    ``TABLE t`` and its wrapped form are all syntax errors there, and
    ``TABLE t`` parses to ``Alias(Column(TABLE), t)`` so the probe rendered
    ``SELECT * FROM (TABLE AS t)``. Refusing locally turns a round trip that
    was going to fail into a message, and costs no statement at all.
    """
    con = make_introspection_con(description=(_FakeColumn("a", 23, "int4"),))

    with pytest.raises(exc.XorqError):
        con._get_schema_using_query(query)

    assert not issued(con), "a statement was sent for a query that cannot work"


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("SELECT a FROM t; -- trailing note", id="trailing-comment"),
        pytest.param("SELECT a FROM t;", id="trailing-semicolon"),
        pytest.param("SELECT a FROM t;;", id="doubled-semicolon"),
    ],
)
def test_get_schema_using_query_accepts_one_statement_with_trailing_noise(
    query: str,
) -> None:
    """A trailing separator or comment is not a second statement.

    ``sg.parse`` yields an element for each: ``SELECT 1; -- note`` parses to
    ``[Select, Semicolon]`` and ``SELECT 1;;`` to ``[Select, None]``. Counting
    those rejected valid SQL -- verified live, ``con.sql("SELECT event FROM
    xorq_test.events; -- note")`` raised while the same query without the
    comment worked -- and it was a regression against both the inherited
    ``parse_one`` path and this backend's own first implementation.
    """
    con = make_introspection_con(description=(_FakeColumn("a", 23, "int4"),))
    schema = con._get_schema_using_query(query)

    assert schema.names == ("a",)


def test_get_schema_using_query_rejects_more_than_one_statement() -> None:
    """``parse_one`` silently probes the first statement while
    ``ops.SQLQueryResult`` stores and executes the whole string, so the schema
    would describe a different statement than the one that runs."""
    con = make_introspection_con(description=(_FakeColumn("a", 23, "int4"),))

    with pytest.raises(exc.XorqError, match="single statement"):
        con._get_schema_using_query("SELECT 1 AS a; SELECT 2 AS b")

    assert not issued(con)


def test_get_schema_using_query_keeps_duplicate_result_column_names_loud() -> None:
    """``SELECT t1.id, t2.id`` -- or just ``SELECT 1, 2``, whose columns are
    both ``?column?`` -- gives a description with a repeated name. Keying a
    dict on it returned a schema one column short of the cursor, which
    ``_fetch_from_cursor`` then misaligns rather than rejecting. The inherited
    temporary-view path failed loudly here and this keeps that.
    """
    con = make_introspection_con(
        description=(
            _FakeColumn("id", 23, "int4"),
            _FakeColumn("id", 1043, "varchar"),
        )
    )

    with pytest.raises(exc.IntegrityError, match="id"):
        con._get_schema_using_query("SELECT t1.id, t2.id FROM t1, t2")


def test_an_unmappable_array_element_does_not_escape_the_guard() -> None:
    """The postcondition used to inspect only the top-level dtype.

    ``bpchar[]`` is a real spelling -- psycopg names OID 1014 exactly that, and
    ``svv_all_columns`` shows ``"char"[]``/``integer[]`` for ``pg_catalog``
    relations -- and it mapped to ``Array(value_type=Unknown)``, an ``Unknown``
    the top-level check walked straight past.
    """
    mapper = RedshiftBackend.compiler.type_mapper

    # The alias applies through the array suffix, so this one is mappable.
    assert mapper.from_string("bpchar[]", nullable=True) == dt.Array(
        dt.String(nullable=True), nullable=True
    )
    # This one is not, and must not come back as Array(Unknown).
    with pytest.raises(exc.UnsupportedBackendType):
        mapper.from_string("xid[]", nullable=True)


def test_geo_types_raise_however_they_are_spelled() -> None:
    """``geometry``/``geography`` are on the name list, but ``point``/``line``/
    ``polygon`` reach a geo type through ``PostgresType.unknown_type_strings``
    and never touch it. Rejecting structurally covers both.

    This compiler is still the PostgreSQL one, so a geo operation on such a
    column emits a PostGIS call Redshift does not implement.
    """
    mapper = RedshiftBackend.compiler.type_mapper

    for spelling in ("geometry", "geography", "point", "line", "polygon"):
        with pytest.raises(exc.UnsupportedBackendType):
            mapper.from_string(spelling, nullable=True)


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("append", id="append-creates-nothing-to-mark"),
        pytest.param("create_append", id="create-append-shadows-via-pg-temp"),
    ],
)
def test_temporary_is_refused_for_the_append_modes(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """``append`` emits no ``CREATE`` for ``TEMPORARY`` to mark, and
    ``create_append`` would render ``CREATE TEMPORARY TABLE IF NOT EXISTS``,
    which resolves against ``pg_temp`` and shadows a permanent table.

    Probed with a driver available, which is the configuration that used to
    divert to ADBC: the rejection must come from this method, not from
    whichever branch a predicate picked."""
    con = make_offline_con(password="static")
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: None)

    with pytest.raises(ValueError, match="temporary=True is not supported"):
        con.read_record_batches(
            make_reader({"a": [1], "b": ["x"]}),
            table_name="t",
            temporary=True,
            mode=mode,
        )

    assert con.con.log == []


def test_temporary_is_refused_for_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    """``replace`` emits an unqualified ``DROP TABLE IF EXISTS`` before the
    ``CREATE``, and it resolves through ``search_path``. With no temporary
    table of that name in the session yet, the DROP lands on the PERMANENT
    one, and what replaces it disappears at disconnect.

    Separate from the append-mode guard above because the harm differs in kind,
    not degree: that one refuses SHADOWING, which ends with the session, and
    this one refuses DESTRUCTION, which does not.

    Probed with a driver available, which is the configuration that used to
    divert to ADBC: the rejection must come from this method."""
    con = make_offline_con(password="static")
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: None)

    with pytest.raises(ValueError, match="temporary=True is not supported"):
        con.read_record_batches(
            make_reader({"a": [1], "b": ["x"]}),
            table_name="t",
            temporary=True,
            mode="replace",
        )

    # The DROP is the whole point: nothing may reach the server.
    assert con.con.log == []


def test_clone_does_not_carry_client_encoding(
    postgres_utils: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``do_connect`` keeps ``client_encoding`` out of ``_con_kwargs``, the
    profile and the build hash by defaulting it below the caller's kwargs.

    ``clone`` rebuilds its kwargs from the LIVE connection rather than from
    ``_con_kwargs``, and ``get_parameters`` reports every setting the
    connection actually has -- so without ``_clone_drop_dsn_params`` the clone
    reacquires a setting nobody passed, and a cloned-then-built artifact hashes
    differently from one built off the source.

    Hash equality with the source is deliberately NOT asserted here: this fake
    DSN reports ``port`` as a string, as libpq does, so an equality assertion
    would pin unrelated coercions rather than this guard. What is asserted is
    the claim the guard makes -- the setting is absent -- plus that the drop is
    narrow: ``options`` carries ``search_path`` and must survive it.
    """

    class _FakeInfo:
        def __init__(self, parameters: dict) -> None:
            self._parameters = parameters

        def get_parameters(self) -> dict:
            return dict(self._parameters)

    con = make_offline_con(password="static", user="u", database="d")
    con.con.info = _FakeInfo(
        {
            "host": "example.invalid",
            "port": str(redshift_module.DEFAULT_PORT),
            "user": "u",
            "dbname": "d",
            "options": "-c search_path=myschema",
            "client_encoding": "UNICODE",
        }
    )
    con.con.autocommit = True

    recorded = {}

    def fake_connect(**kwargs):
        recorded.update(kwargs)
        return _FakeConnection()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(RedshiftBackend, "_post_connect", lambda self: None)

    clone = con.clone()

    # ``do_connect`` still defaults it, so the wire is configured ...
    assert recorded["client_encoding"] == "utf8"
    # ... and the clone did not inherit the DSN's value as a caller argument.
    assert "client_encoding" not in clone._con_kwargs
    assert "client_encoding" not in clone._profile.kwargs_dict
    # The drop is narrow: a DSN setting the caller does depend on survives.
    assert clone._con_kwargs["options"] == "-c search_path=myschema"


def test_null_typed_columns_are_refused_before_any_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A null column renders as the column type ``NULL``, which no server
    accepts. The vendored ``_register_in_memory_table`` guards this; the
    psycopg ingest was written without it.

    Probed with a driver available, which is the configuration that used to
    divert to ADBC: the guard must run before any statement is issued."""
    schema = pa.schema([("n", pa.null()), ("a", pa.int64())])
    con = make_offline_con(password="static")
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: None)

    with pytest.raises(exc.XorqTypeError, match="null. typed columns"):
        con.read_record_batches(
            make_reader({"n": [None], "a": [1]}, schema=schema), table_name="t"
        )

    assert con.con.log == []


def test_an_unavailable_driver_is_not_even_imported(
    monkeypatch: pytest.MonkeyPatch, postgres_utils
) -> None:
    """``postgres_utils`` imports the driver at module scope, so an import
    above the probe raises in exactly the case the probe detects.

    Asserting it is not re-imported is what separates this from
    ``test_an_unavailable_driver_is_not_dialled_at_all``, which cannot see the
    ordering. Requesting the ``postgres_utils`` fixture is what puts it in
    ``sys.modules`` for the ``delitem`` below to take back out: the module
    deliberately does not import it at the top, so without the fixture this
    test would depend on an earlier test having imported it."""
    con = make_offline_con()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")
    monkeypatch.delitem(sys.modules, "xorq.common.utils.postgres_utils")

    assert con._open_adbc_conn_or_none() is None
    assert "xorq.common.utils.postgres_utils" not in sys.modules


def test_redshift_exposes_no_module_level_connect() -> None:
    """Redshift deliberately has no module-level ``connect`` of its own: the
    loader builds ``xo.redshift.connect`` from the bound ``Backend.connect``.

    The postgres module-level ``connect`` and ``clone`` themselves are covered
    in ``backends/postgres/tests/test_connect_and_clone.py``."""
    assert not hasattr(redshift_module, "connect")
    assert redshift_module.__all__ == ["Backend"]


def test_clone_keeps_a_client_encoding_the_caller_passed(
    postgres_utils: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sibling test above covers the IMPLICIT case -- nobody passed one, so
    the DSN's value must not be reacquired. This is the other half, and it was
    broken: ``_clone_drop_dsn_params`` was dissoc-ed from the MERGED dict,
    which already held ``_con_kwargs``, so a caller who *did* ask for
    ``latin1`` got a clone silently dialling ``utf8``.

    That is the failure the mechanism exists to prevent, arrived at from the
    other direction: source and clone disagree about a connection setting, so
    a cloned-then-built artifact hashes differently from one built off the
    source. ``test_clone_keeps_a_hostaddr_the_caller_passed`` in the postgres
    suite states exactly this invariant for the DSN dissoc; nothing restated
    it for the second, later mechanism.
    """

    class _FakeInfo:
        def __init__(self, parameters: dict) -> None:
            self._parameters = parameters

        def get_parameters(self) -> dict:
            return dict(self._parameters)

    con = make_offline_con(
        password="static", user="u", database="d", client_encoding="latin1"
    )
    con.con.info = _FakeInfo(
        {
            "host": "example.invalid",
            "port": str(redshift_module.DEFAULT_PORT),
            "user": "u",
            "dbname": "d",
            "client_encoding": "UNICODE",
        }
    )
    con.con.autocommit = True

    recorded = {}

    def fake_connect(**kwargs):
        recorded.update(kwargs)
        return _FakeConnection()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(RedshiftBackend, "_post_connect", lambda self: None)

    clone = con.clone()

    # The caller asked for it, so it is theirs to keep -- on the wire ...
    assert recorded["client_encoding"] == "latin1"
    # ... and in the clone's own kwargs, so the profile agrees with the source.
    assert clone._con_kwargs["client_encoding"] == "latin1"
    # The DSN's ``UNICODE`` is still what gets dropped, not the caller's value.
    assert clone._con_kwargs["client_encoding"] != "UNICODE"


def test_clone_refuses_rather_than_borrowing_the_postgres_env_password(
    postgres_utils: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Redshift backend with no password in ``_con_kwargs`` -- which is every
    one built by ``from_connection`` -- inherited postgres's
    ``make_credential_defaults()``, i.e. ``$POSTGRES_PASSWORD``.

    Two bad outcomes, and the second is the dangerous one: on a machine with
    no ``POSTGRES_PASSWORD`` it refused with a message naming a service the
    caller never used, and on a developer machine where that variable happens
    to be set it dialled the *warehouse* with a local postgres password.

    Asserted with the variable POPULATED, because that is the case the old
    code passed silently. A test that only unset it would see an error either
    way and could not tell the two apart.
    """
    monkeypatch.setenv("POSTGRES_PASSWORD", "a-local-postgres-password")

    class _FakeInfo:
        def __init__(self, parameters: dict) -> None:
            self._parameters = parameters

        def get_parameters(self) -> dict:
            return dict(self._parameters)

    con = RedshiftBackend()
    type(con).__init__(con)
    con.con = _FakeConnection()
    con.con.info = _FakeInfo(
        {
            "host": "example.invalid",
            "port": str(redshift_module.DEFAULT_PORT),
            "user": "u",
            "dbname": "d",
        }
    )
    con.con.autocommit = True

    dialled = {}

    def fake_connect(**kwargs):
        dialled.update(kwargs)
        return _FakeConnection()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(RedshiftBackend, "_post_connect", lambda self: None)

    with pytest.raises(ValueError, match="password is required"):
        con.clone()

    # The message names redshift and not some other service's env var.
    with pytest.raises(ValueError) as excinfo:
        con.clone()
    assert "redshift" in str(excinfo.value)
    assert "POSTGRES_PASSWORD" not in str(excinfo.value)
    # And nothing was dialled with the borrowed credential.
    assert dialled == {}


def test_postgres_clone_still_falls_back_to_its_own_env_password(
    postgres_utils: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook must not change postgres. Redshift returning ``None`` is an
    override, not a removal of the base behaviour."""
    con = PostgresBackend()
    assert con._clone_credential_default_password() == "$POSTGRES_PASSWORD"
