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
import sys
from types import ModuleType

import pyarrow as pa
import pytest
import sqlglot as sg


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
# is ``xorq.common.utils.postgres_utils``, which four tests below need and the
# ``postgres_utils`` fixture imports for them. One further test needs it
# merely *installed*, for the probe's own ``find_spec``, and guards itself.
# Six tests of forty, rather than the whole module.
#
# CI selects by marker with no path filter, so every job COLLECTS this file;
# without the guard the jobs lacking the extras failed collection outright
# rather than deselecting -- a red build reading ModuleNotFoundError, not a
# skip. ``backends/conftest.py`` guards ``backends/<name>/`` paths only, and
# this file is deliberately sited outside them (see the docstring above), so
# the guard has to be here. The E402s are that guard running first, not import
# sloppiness -- and the two blank lines above this comment are load-bearing:
# with one, ruff raises I001 and ``--fix`` hoists the imports back above the
# guard. Same shape as ``test_redshift_cache_freshness.py``.
pytest.importorskip("adbc_driver_manager")
psycopg = pytest.importorskip("psycopg")

import xorq  # noqa: E402
import xorq.api as xo  # noqa: E402
import xorq.backends.postgres as postgres_module  # noqa: E402
import xorq.backends.redshift as redshift_module  # noqa: E402
import xorq.common.exceptions as exc  # noqa: E402
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
    """``append`` emits no ``CREATE`` for the psycopg branch to mark while the
    ADBC branch marks unconditionally; ``create_append`` would render
    ``CREATE TEMPORARY TABLE IF NOT EXISTS``, which resolves against
    ``pg_temp`` and shadows a permanent table. Probed over both reasons so the
    rejection is not itself a divergence."""
    for reason in ("no driver", None):
        con = make_offline_con(password="static")
        monkeypatch.setattr(
            con, "_adbc_unavailable_reason", lambda reason=reason: reason
        )

        with pytest.raises(ValueError, match="temporary=True is not supported"):
            con.read_record_batches(
                make_reader({"a": [1], "b": ["x"]}),
                table_name="t",
                temporary=True,
                mode=mode,
            )

        assert con.con.log == []


def test_null_typed_columns_are_refused_on_both_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A null column renders as the column type ``NULL``, which no server
    accepts. The vendored ``_register_in_memory_table`` guards this; the
    psycopg ingest was written without it. Guarded above the dispatch, so
    probed over both reasons."""
    schema = pa.schema([("n", pa.null()), ("a", pa.int64())])

    for reason in ("no driver", None):
        con = make_offline_con(password="static")
        monkeypatch.setattr(
            con, "_adbc_unavailable_reason", lambda reason=reason: reason
        )

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


def test_clone_returns_the_subclass_not_postgres() -> None:
    """``clone`` resolved the postgres module's ``connect`` through
    ``__globals__``, so a Redshift caller got a postgres backend back.
    Asserted on the source rather than by cloning, which needs a live
    ``con.info``."""
    source = inspect.getsource(PostgresBackend.clone)
    assert "return self.connect(" in source
    assert "return connect(" not in source


def test_module_level_connect_builds_a_connected_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Backend.connect(**kwargs)`` was an unbound call that raised
    ``TypeError``; ``clone`` was its only caller. Redshift's copy is gone --
    the loader builds ``xo.redshift.connect`` from the bound method."""
    monkeypatch.setattr(psycopg, "connect", lambda **kwargs: _FakeConnection())
    monkeypatch.setattr(PostgresBackend, "_post_connect", lambda self: None)

    con = postgres_module.connect(host="example.invalid", user="u", database="d")
    assert isinstance(con, PostgresBackend)
    assert con.con is not None

    assert not hasattr(redshift_module, "connect")
    assert redshift_module.__all__ == ["Backend"]
