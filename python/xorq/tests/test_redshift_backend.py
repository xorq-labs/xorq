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
import inspect

import pyarrow as pa
import pytest
import sqlglot as sg
import sqlglot.expressions as sge

import xorq
import xorq.api as xo
import xorq.common.exceptions as exc
import xorq.common.utils.postgres_utils as postgres_utils
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.backends.postgres import Backend as PostgresBackend
from xorq.backends.redshift import DEFAULT_PORT, INGEST_MODES
from xorq.backends.redshift import Backend as RedshiftBackend
from xorq.vendor.ibis.backends.profiles import (
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

    Asserted on the rendered string rather than by executing, because the
    failure is a *server-side* error on SQL that compiles cleanly. The bare
    form is what both the postgres and redshift sqlglot dialects produce, so
    this also pins that no dialect swap silently reintroduces it.
    """
    dialect = RedshiftBackend.compiler.dialect

    assert sg.select(sg.func("current_schema")).sql(dialect) == "SELECT CURRENT_SCHEMA"
    assert (
        sg.select(sge.Anonymous(this="current_schema")).sql(dialect)
        == "SELECT CURRENT_SCHEMA()"
    )


def test_current_catalog_needs_no_override():
    """``CURRENT_DATABASE()`` already renders parenthesised, so only
    ``current_database`` (which selects the *schema*) needed overriding."""
    dialect = RedshiftBackend.compiler.dialect
    assert (
        sg.select(sg.func("current_database")).sql(dialect)
        == "SELECT CURRENT_DATABASE()"
    )


def test_client_encoding_defaults_without_entering_the_build_hash():
    """``client_encoding`` is mandatory -- Redshift reports the PG 8.x alias
    ``UNICODE``, absent from psycopg3's codec map, so every query otherwise
    raises ``NotSupportedError``.

    It is defaulted inside ``do_connect`` rather than by the caller precisely
    so it stays out of ``_con_kwargs``, which is captured from the caller's
    arguments and feeds the build hash.
    """
    con = RedshiftBackend()
    type(con).__init__(con, host="example.invalid", port=DEFAULT_PORT)

    assert "client_encoding" not in con._con_kwargs
    assert con._con_kwargs == {"host": "example.invalid", "port": DEFAULT_PORT}


def test_default_port_is_redshifts():
    assert DEFAULT_PORT == 5439
    defaults = dict(
        zip(
            RedshiftBackend.do_connect.__code__.co_varnames[1:],
            RedshiftBackend.do_connect.__defaults__,
        )
    )
    assert defaults["port"] == DEFAULT_PORT


def test_profile_roundtrips():
    con = RedshiftBackend()
    type(con).__init__(con, host="example.invalid", port=DEFAULT_PORT)
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

    def __init__(self, log):
        self.log = log

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


class _FakeConnection:
    def __init__(self):
        self.log = []

    def cursor(self, *args, **kwargs):
        return _FakeCursor(self.log)

    def transaction(self):
        return contextlib.nullcontext()


def make_offline_con(**con_kwargs):
    """A Redshift backend with ``_con_kwargs`` populated but nothing dialled.

    ``type(con).__init__`` is the same idiom the profile tests above use: it
    runs ``BaseBackend.__init__``, which captures ``_con_kwargs`` and builds
    the profile, without ``do_connect``.
    """
    con = RedshiftBackend()
    type(con).__init__(con, host="example.invalid", port=DEFAULT_PORT, **con_kwargs)
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


def test_psycopg_ingest_creates_and_inserts(monkeypatch):
    """The baseline the ADR promises and the inherited method did not provide.

    ``INSERT`` rather than ``COPY`` is not a shortcut: Redshift has no
    ``COPY ... FROM STDIN``, so a ``COPY`` baseline would need an S3 bucket and
    an assumable role, which is the deferred ``redshift.ingest.bucket`` work.
    """
    con = make_offline_con()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")

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


def test_psycopg_ingest_consumes_every_batch(monkeypatch):
    """A reader is a stream, and the obvious wrong implementation -- reading
    ``next(reader)`` or materialising ``.read_all()`` into one statement --
    silently drops or reshapes rows."""
    con = make_offline_con()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")

    con.read_record_batches(
        make_reader(
            {"a": [1], "b": ["x"]},
            {"a": [2, 3], "b": ["y", "z"]},
        ),
        table_name="t",
    )

    inserted = [entry[2] for entry in con.con.log if entry[0] == "executemany"]
    assert inserted == [[(1, "x")], [(2, "y"), (3, "z")]]


def test_psycopg_ingest_of_an_empty_batch_still_creates_the_table(monkeypatch):
    """``executemany`` with no rows is skipped, but the schema still lands --
    an empty parquet file must produce an empty table, not no table."""
    con = make_offline_con()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")

    con.read_record_batches(make_reader({"a": [], "b": []}), table_name="t")

    assert executed(con) == ['CREATE TABLE "t" ("a" BIGINT, "b" VARCHAR)']
    assert not [entry for entry in con.con.log if entry[0] == "executemany"]


def test_psycopg_ingest_accepts_a_table_like_the_adbc_branch_does(monkeypatch):
    """``adbc_ingest`` takes a ``pa.Table``, and iterating one yields *columns*
    -- so the naive psycopg loop would fail on a missing ``num_rows`` for an
    input the accelerator handles. Which branch runs has to stay an
    implementation detail."""
    con = make_offline_con()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")

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
def test_psycopg_ingest_modes_match_their_adbc_meanings(monkeypatch, mode, expected):
    """Which branch runs has to stay an implementation detail, and it stops
    being one the moment the two disagree about what ``mode`` means:
    ``append`` must not create, ``create`` must not tolerate an existing table,
    ``replace`` must drop it, ``create_append`` must tolerate it."""
    con = make_offline_con()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")

    con.read_record_batches(
        make_reader({"a": [1], "b": ["x"]}), table_name="t", mode=mode
    )

    assert executed(con) == expected


def test_psycopg_ingest_creates_the_temp_table_directly(monkeypatch):
    """The ADBC path creates a permanent table and converts it afterwards with
    ``make_table_temporary``. That is not overhead ADBC failed to avoid -- it
    connects separately, so a temp table created there would be invisible.
    Sharing the psycopg connection is what makes the direct form correct, so
    assert no rename-and-copy appears."""
    con = make_offline_con()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")

    con.read_record_batches(
        make_reader({"a": [1], "b": ["x"]}), table_name="t", temporary=True
    )

    assert executed(con) == ['CREATE TEMPORARY TABLE "t" ("a" BIGINT, "b" VARCHAR)']


def test_ingest_dispatches_to_adbc_when_it_is_available(monkeypatch):
    """The other half of the dispatch. Without this, a psycopg-only
    implementation would pass every test above and silently discard the
    accelerator."""
    con = make_offline_con(password="static")
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: None)

    calls = []
    monkeypatch.setattr(
        PostgresBackend,
        "read_record_batches",
        lambda self, record_batches, **kwargs: calls.append(kwargs) or "delegated",
    )

    result = con.read_record_batches(
        make_reader({"a": [1], "b": ["x"]}), table_name="t", mode="append"
    )

    assert result == "delegated"
    assert calls == [
        {"table_name": "t", "password": None, "temporary": False, "mode": "append"}
    ]
    # nothing was ingested twice
    assert con.con.log == []


def test_ingest_rejects_a_missing_table_name(monkeypatch):
    """Inherited, ``table_name=None`` reached ``adbc_ingest`` and failed
    somewhere inside the driver."""
    con = make_offline_con()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")

    with pytest.raises(ValueError, match="table_name"):
        con.read_record_batches(make_reader({"a": [1], "b": ["x"]}))


def test_ingest_validates_mode_before_choosing_a_branch(monkeypatch):
    """Validation belongs above the dispatch: an unknown mode must fail
    identically whether or not a driver happens to be installed."""
    con = make_offline_con(password="static")
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: None)
    monkeypatch.setattr(
        PostgresBackend,
        "read_record_batches",
        lambda *args, **kwargs: pytest.fail("dispatched on an invalid mode"),
    )

    with pytest.raises(ValueError, match="mode must be one of"):
        con.read_record_batches(
            make_reader({"a": [1], "b": ["x"]}), table_name="t", mode="upsert"
        )


# ---------------------------------------------------------------------------
# Driver availability, and telling driver-absent from auth-failed.
# ---------------------------------------------------------------------------


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
    con = make_offline_con(**con_kwargs)
    assert "password" in con._adbc_unavailable_reason()


def test_adbc_is_available_when_installed_and_credentialed():
    """A ``None`` reason means installed *and* credentialed -- not that the
    driver is known to work against Redshift. Whether
    ``adbc_driver_postgresql`` speaks to Redshift at all is untested and needs
    a live endpoint; see the alternative recorded in
    ADR-redshift-psycopg-baseline-adbc-optional."""
    pytest.importorskip("adbc_driver_postgresql")
    con = make_offline_con(password="static")
    assert con._adbc_unavailable_reason() is None


def test_auth_failure_is_not_swallowed_as_a_missing_driver(monkeypatch):
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


def test_an_unavailable_driver_is_not_dialled_at_all(monkeypatch):
    """Availability is decided from local facts *before* connecting, which is
    what makes the test above possible: every exception from the connect is
    then a real failure."""
    con = make_offline_con()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")
    monkeypatch.setattr(postgres_utils.PgADBC, "get_conn", raise_get_conn)

    assert con._open_adbc_conn_or_none() is None


def test_the_postgres_seam_keeps_swallowing(monkeypatch):
    """The probe was extracted from ``to_pyarrow_batches`` so Redshift could
    override it. Postgres's own behaviour must be unchanged by that -- its
    catch-all is deliberate, and users connecting without a password in
    ``_con_kwargs`` depend on the quiet fallback."""
    con = PostgresBackend()
    type(con).__init__(con, host="example.invalid")
    monkeypatch.setattr(postgres_utils.PgADBC, "get_conn", raise_get_conn)

    assert con._open_adbc_conn_or_none() is None


def test_ingest_modes_are_the_adbc_ingest_modes():
    assert INGEST_MODES == ("create", "append", "replace", "create_append")


def test_ingest_ddl_pins_two_unverified_redshift_type_widths(monkeypatch):
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
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")

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

    Records ``(sql, params)`` on the connection so a test can assert that the
    schema and table name are *bound*, not interpolated into the statement.
    """

    def __init__(self, con, rows, description):
        super().__init__(con.log)
        self._con = con
        self._rows = rows
        self.description = description

    def execute(self, sql, params=None, **kwargs):
        super().execute(sql)
        self._con.calls.append((sql, params))
        return self

    def fetchall(self):
        return list(self._rows)


class _IntrospectionConnection(_FakeConnection):
    def __init__(self, rows=(), description=None):
        super().__init__()
        self.calls = []
        self._rows = rows
        self._description = description

    def cursor(self, *args, **kwargs):
        return _IntrospectionCursor(self, self._rows, self._description)


class _FakeColumn:
    """The subset of ``psycopg.Column`` the query path reads."""

    def __init__(self, name, type_code, precision=None, scale=None):
        self.name = name
        self.type_code = type_code
        self.precision = precision
        self.scale = scale


def make_introspection_con(rows=(), description=None):
    con = make_offline_con()
    con.con = _IntrospectionConnection(rows=rows, description=description)
    return con


def issued(con):
    return [sql for (sql, _params) in con.con.calls]


# Shaped after the worked example in the SVV_ALL_COLUMNS reference. Note that
# ``numeric_precision``/``numeric_scale`` are populated for *integer* columns
# too -- buyerid carries 32/0 -- which is exactly the trap that makes "append
# the precision whenever it is there" produce the unparsable ``integer(32)``.
SVV_ROWS = (
    # column_name, data_type, is_nullable, char_max_len, num_precision, num_scale
    ("buyerid", "integer", "NO", None, 32, 0),
    ("commission", "numeric", "YES", None, 8, 2),
    ("dateid", "smallint", "NO", None, 16, 0),
    ("eventname", "character varying", "YES", 256, None, None),
    ("saletime", "timestamp without time zone", "YES", None, None, None),
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
    con = make_introspection_con(rows=(("buyerid", "integer", "NO", None, 32, 0),))
    schema = con.get_schema("sales", database="public")

    assert schema["buyerid"] == dt.Int32(nullable=False)


@pytest.mark.parametrize(
    ("flag", "nullable"),
    [("YES", True), ("yes", True), ("NO", False), ("no", False)],
)
def test_get_schema_reads_is_nullable_case_insensitively(flag, nullable):
    """The reference documents the values as lowercase ``yes``/``no`` and its
    own worked example prints them uppercase. Neither is worth betting a
    nullability flag on."""
    con = make_introspection_con(rows=(("c", "integer", flag, None, 32, 0),))
    schema = con.get_schema("t", database="public")

    assert schema["c"].nullable is nullable


def test_get_schema_binds_the_schema_and_table_rather_than_interpolating():
    con = make_introspection_con(rows=SVV_ROWS)
    con.get_schema("sales", database="analytics")

    (sql, params) = con.con.calls[-1]
    assert "sales" not in sql
    assert "analytics" not in sql
    assert params["table"] == "sales"
    assert params["schema"] == "analytics"


def test_get_schema_emits_no_array_predicate():
    """The inherited query filters with ``n.nspname = ANY(%(dbs)s)``. Redshift
    has no array type, so the array form has to go even though it compiles."""
    con = make_introspection_con(rows=SVV_ROWS)
    con.get_schema("sales", database="public")

    (sql, _params) = con.con.calls[-1]
    assert "ANY(" not in sql.upper()
    assert "ARRAY" not in sql.upper()


def test_get_schema_scopes_by_catalog_when_one_is_given():
    """``svv_all_columns`` spans databases, so an unscoped query can match a
    same-named table in another one."""
    con = make_introspection_con(rows=SVV_ROWS)
    con.get_schema("sales", catalog="dev", database="public")

    (sql, params) = con.con.calls[-1]
    assert "database_name" in sql
    assert params["catalog"] == "dev"


def test_get_schema_omits_the_catalog_predicate_when_none_is_given():
    con = make_introspection_con(rows=SVV_ROWS)
    con.get_schema("sales", database="public")

    (sql, params) = con.con.calls[-1]
    assert "database_name" not in sql
    assert "catalog" not in params


def test_get_schema_defaults_the_schema_to_the_current_one(monkeypatch):
    """The customer's failure reproduced on a plain single-schema
    ``con.table("t")``, so the no-``database`` path is the one that has to
    work, not only the three-part-named one."""
    con = make_introspection_con(rows=SVV_ROWS)
    monkeypatch.setattr(
        type(con), "current_database", property(lambda self: "reporting")
    )
    con.get_schema("sales")

    (_sql, params) = con.con.calls[-1]
    assert params["schema"] == "reporting"


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

    (sql, _params) = con.con.calls[-1]
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
    con = make_introspection_con(description=(_FakeColumn("a", 23),))
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
            _FakeColumn("a", 23),  # int4
            _FakeColumn("b", 1043),  # varchar
            _FakeColumn("c", 1700, precision=8, scale=2),  # numeric(8,2)
            _FakeColumn("d", 1114),  # timestamp
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
    con = make_introspection_con(description=(_FakeColumn("a", 23),))
    con._get_schema_using_query("SELECT a FROM big")

    (sql,) = issued(con)
    assert sql == "SELECT * FROM (SELECT a FROM big) AS redshift_probe LIMIT 0"


def test_get_schema_using_query_wraps_rather_than_appends():
    """Appending ``LIMIT 0`` to a query that already ends in a ``LIMIT`` would
    be a syntax error, and appending it to a ``UNION`` would bind to the last
    branch only. Wrapping is what makes the probe total."""
    con = make_introspection_con(description=(_FakeColumn("a", 23),))
    con._get_schema_using_query("SELECT a FROM t LIMIT 5")

    (sql,) = issued(con)
    assert sql == "SELECT * FROM (SELECT a FROM t LIMIT 5) AS redshift_probe LIMIT 0"
    assert sg.parse_one(sql, read=con.dialect)


def test_get_schema_using_query_rejects_an_unmappable_oid_loudly():
    """Redshift's own types (SUPER, VARBYTE, GEOMETRY) carry OIDs psycopg does
    not know. Mapping them to ``unknown`` would hand back a schema that looks
    fine and is wrong; the OID is named so a live session can add it."""
    con = make_introspection_con(description=(_FakeColumn("s", 999999),))

    with pytest.raises(exc.UnsupportedBackendType, match="999999"):
        con._get_schema_using_query("SELECT s FROM t")


def test_neither_introspection_path_creates_a_temporary_view():
    """The single statement the customer transcript shows failing, asserted
    across both entry points at once."""
    table_con = make_introspection_con(rows=SVV_ROWS)
    table_con.get_schema("sales", database="public")

    query_con = make_introspection_con(description=(_FakeColumn("a", 23),))
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
    assert "pg_enum" in inspect.getsource(PostgresBackend.get_schema)
