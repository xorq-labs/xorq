"""Offline tests for the Redshift backend.

Sited here, not under ``python/xorq/backends/redshift/tests/``, on purpose.
``xorq/backends/conftest.py`` auto-applies ``pytest.mark.<backend>`` by path and
adds ``core`` only outside ``backends/``, and every CI job selects by marker, so
a test placed under the backend directory would be Redshift-marked, and no CI
job selects that marker -- it would run nowhere. These need no credentials and
should run in the default sweep.

Every trap these cover is silent: each one produces a working-looking backend
that is wrong, so the assertions are on the specific observable, not on
"it connects".
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import subprocess
import sys
import typing
from types import ModuleType

import pyarrow as pa
import pytest


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

import adbc_driver_manager.dbapi  # noqa: E402

import xorq  # noqa: E402
import xorq.api as xo  # noqa: E402
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


def test_exposed_secret_check_catches_every_postgres_secret_key() -> None:
    """``sslkey``/``passfile`` must raise, not just ``password``.

    ``check_for_exposed_secrets`` reads the static mirror, not the class, so a
    narrowed ``_secret_keys`` declaration is caught by the test above, not
    here. This one iterates the postgres keys so that it fails if the
    mirror's redshift entry is narrowed."""
    for key in PostgresBackend._secret_keys:
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


def test_current_catalog_needs_no_override() -> None:
    """The inherited ``current_catalog`` already emits ``CURRENT_DATABASE()``
    parenthesised, so only ``current_database`` (which selects the *schema*)
    needed overriding. Asserted on the SQL the backend sends, like the test
    above, so it fails if the inherited query ever renders bare."""
    con = make_offline_con()
    con.con = _FakeConnection(rows=[("d",)])

    assert con.current_catalog == "d"
    assert executed(con) == ["SELECT CURRENT_DATABASE()"]


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
    kwarg is caught where it lands, at ``psycopg.connect``. The second half is
    asserted on the profile, which ``Profile.from_con`` fills from
    ``do_connect``'s signature defaults as well as the caller's arguments --
    so moving the default into the signature fails here.
    """
    recorded = {}

    def fake_connect(**kwargs):
        recorded.update(kwargs)
        return _FakeConnection()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(PostgresBackend, "_post_connect", lambda self: None)

    con = RedshiftBackend().connect(
        host="example.invalid", user="u", password="p", database="d"
    )

    # Reached the driver ...
    assert recorded["client_encoding"] == "utf8"
    assert recorded["port"] == redshift_module.DEFAULT_PORT
    # ... and did not reach the build hash.
    assert "client_encoding" not in con._profile.kwargs_dict


def test_prepare_threshold_defaults_off_without_entering_the_build_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redshift rejects ``DEALLOCATE ALL``, which psycopg sends on rollback to
    clear the statements it has prepared; measured live, the error rolled back
    a ``drop_table`` and left an ``into_backend`` placeholder in the schema.
    With the threshold ``None`` nothing is prepared, so nothing is sent.

    Set on the connection by ``_post_connect``, the hook every construction
    path reaches, so it never enters the caller's kwargs or the profile."""
    recorded = {}

    def fake_connect(**kwargs):
        recorded.update(kwargs)
        return _FakeConnection()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(PostgresBackend, "_post_connect", lambda self: None)

    con = RedshiftBackend().connect(
        host="example.invalid", user="u", password="p", database="d"
    )

    assert con.con.prepare_threshold is None
    assert "prepare_threshold" not in recorded
    assert "prepare_threshold" not in con._profile.kwargs_dict


def test_a_callers_prepare_threshold_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = {}

    def fake_connect(**kwargs):
        recorded.update(kwargs)
        # psycopg applies the kwarg to the connection it returns.
        return _FakeConnection(prepare_threshold=kwargs.get("prepare_threshold", 5))

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(PostgresBackend, "_post_connect", lambda self: None)

    con = RedshiftBackend().connect(
        host="example.invalid",
        user="u",
        password="p",
        database="d",
        prepare_threshold=3,
    )

    assert recorded["prepare_threshold"] == 3
    assert con.con.prepare_threshold == 3


class _FakeEncodingInfo:
    def __init__(self, encoding: str | None) -> None:
        self._encoding = encoding

    @property
    def encoding(self) -> str:
        if self._encoding is None:
            # What psycopg raises for Redshift's ``UNICODE``.
            raise psycopg.NotSupportedError("codec not available in Python: 'UNICODE'")
        return self._encoding


def test_from_connection_turns_off_statement_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_connection`` skips ``do_connect``; ``_post_connect``, which it
    does reach, sets the threshold on the connection it was given."""
    monkeypatch.setattr(PostgresBackend, "_post_connect", lambda self: None)
    raw = _FakeConnection(prepare_threshold=5)

    con = RedshiftBackend.from_connection(raw)

    assert con.con is raw
    assert raw.prepare_threshold is None


def test_from_connection_refuses_an_undecodable_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection opened without ``client_encoding`` reports Redshift's
    ``UNICODE``, which psycopg cannot decode, so every query on it would fail.
    It is refused before any SQL, naming the setting."""
    post_connected = []
    monkeypatch.setattr(
        PostgresBackend, "_post_connect", lambda self: post_connected.append(self)
    )
    raw = _FakeConnection()
    raw.info = _FakeEncodingInfo(None)

    with pytest.raises(ValueError, match="client_encoding='utf8'"):
        RedshiftBackend.from_connection(raw)

    assert post_connected == []
    assert raw.log == []


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
    assert "prepare_threshold" not in recorded


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
    def __init__(self, rows: tuple = (), prepare_threshold: int | None = 5) -> None:
        self.log: list = []
        self.rows = rows
        # psycopg's defaults: a decodable encoding, and a threshold of 5.
        self.info = _FakeEncodingInfo("utf-8")
        self.prepare_threshold = prepare_threshold

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


def raise_get_conn(*args, **kwargs):
    raise RuntimeError("FATAL: password authentication failed for user")


class _FakeLibpqInfo:
    """What ``PgADBC.params`` reads off the live psycopg connection."""

    user = "u"
    host = "example.invalid"
    port = 5439
    dbname = "d"


def capture_adbc_uris(
    monkeypatch: pytest.MonkeyPatch, postgres_utils: ModuleType
) -> list[str]:
    uris: list[str] = []

    def connect(uri):
        uris.append(uri)
        return "adbc-conn"

    monkeypatch.setattr(postgres_utils.adbc_driver_postgresql.dbapi, "connect", connect)
    return uris


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

    The other ingest tests that patch the predicate to ``None`` would also
    fail on a reintroduced dispatch, but only incidentally (their fake
    connection has no ``info`` for ``PgADBC`` to read); this is the one that
    names it.
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
    """A ``None`` reason means installed *and* credentialed -- a local fact,
    not a check that the driver works against Redshift. That was measured
    against a live endpoint and is recorded in ADR-2332; no offline test can
    reach it."""
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
    con.con.info = _FakeLibpqInfo()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: None)
    monkeypatch.setattr(
        postgres_utils.adbc_driver_postgresql.dbapi, "connect", raise_get_conn
    )

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
    monkeypatch.setattr(
        postgres_utils.adbc_driver_postgresql.dbapi, "connect", raise_get_conn
    )

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


_BASE_URI = "postgresql://u:static@example.invalid:5439/d"


@pytest.mark.parametrize(
    ("schema", "options"),
    [
        pytest.param("xorq_test", "-csearch_path%3Dxorq_test", id="plain"),
        pytest.param("s1,public", "-csearch_path%3Ds1%2Cpublic", id="list"),
        pytest.param("a b", "-csearch_path%3Da%5C%20b", id="space-escaped"),
        pytest.param('"a b"', "-csearch_path%3D%22a%5C%20b%22", id="quoted"),
        pytest.param("x\\y", "-csearch_path%3Dx%5C%5Cy", id="backslash-escaped"),
    ],
)
def test_adbc_read_connection_carries_the_schema(
    monkeypatch: pytest.MonkeyPatch,
    postgres_utils: ModuleType,
    schema: str,
    options: str,
) -> None:
    """psycopg gets ``schema`` from ``_post_connect``'s ``set_config``; the
    ADBC read connection is a second connection and used to get nothing, so it
    ran with ``'$user, public'``. The compiler emits unqualified table names,
    so every table-bound read raised "relation does not exist" there and was
    re-run on psycopg by the execute-stage catch -- right rows, and an
    accelerator that never served a table-bound read.

    The escaping cases are libpq's rule for ``options`` (whitespace splits
    arguments unless backslash-escaped); each value's effective
    ``search_path`` was measured against a local postgres to match what
    ``set_config`` gives the same string.
    """
    con = make_offline_con(password="static", schema=schema)
    con.con.info = _FakeLibpqInfo()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: None)
    uris = capture_adbc_uris(monkeypatch, postgres_utils)

    assert con._open_adbc_conn_or_none() == "adbc-conn"
    assert uris == [f"{_BASE_URI}?options={options}"]


def test_adbc_read_connection_without_a_schema_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, postgres_utils: ModuleType
) -> None:
    con = make_offline_con(password="static")
    con.con.info = _FakeLibpqInfo()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: None)
    uris = capture_adbc_uris(monkeypatch, postgres_utils)

    con._open_adbc_conn_or_none()

    assert uris == [_BASE_URI]


def test_postgres_adbc_read_connection_is_given_the_schema_too(
    monkeypatch: pytest.MonkeyPatch, postgres_utils: ModuleType
) -> None:
    """The search path is ``PgADBC``'s, so postgres gets it as well: it had
    the same gap, and its table-bound reads on a ``schema=`` connection fell
    back to psycopg in the same way."""
    con = PostgresBackend()
    type(con).__init__(con, host="example.invalid", password="static", schema="s")
    con.con = _FakeConnection()
    con.con.info = _FakeLibpqInfo()
    uris = capture_adbc_uris(monkeypatch, postgres_utils)

    con._open_adbc_conn_or_none()

    assert uris == [f"{_BASE_URI}?options=-csearch_path%3Ds"]


def test_ingest_modes_are_the_adbc_ingest_modes() -> None:
    """Compared against what ``adbc_ingest`` declares, so a driver release
    that adds or drops a mode fails here rather than drifting silently."""
    annotation = (
        inspect.signature(adbc_driver_manager.dbapi.Cursor.adbc_ingest)
        .parameters["mode"]
        .annotation
    )

    assert set(redshift_module.INGEST_MODES) == set(typing.get_args(annotation))


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


def test_clone_carries_only_the_settings_the_caller_passed(
    postgres_utils: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``clone`` reads the live connection's ``get_parameters``, which reports
    every non-default libpq setting, asked for or not: ``client_encoding`` that
    ``do_connect`` defaulted, ``sslcertmode`` that libpq 17+ reports unasked,
    and anything libpq took from the environment. Carrying those made a clone
    hash differently from its source, and put them in the clone's ADBC URI.

    So for a source opened with kwargs, the caller's values win and the DSN
    fills in only address keys the caller left out. The fake reports the keys
    libpq 18 reports that matter here.
    """

    class _FakeInfo:
        def __init__(self, parameters: dict) -> None:
            self._parameters = parameters

        def get_parameters(self) -> dict:
            return dict(self._parameters)

    con = make_offline_con(
        password="static", user="u", database="d", options="-c search_path=mine"
    )
    con.con.info = _FakeInfo(
        {
            "host": "example.invalid",
            "port": str(redshift_module.DEFAULT_PORT),
            "user": "u",
            "dbname": "d",
            "options": "-c search_path=mine",
            # Defaulted by ``do_connect``; libpq echoes the client's value.
            "client_encoding": "utf8",
            # Reported by libpq 17+ although nobody set it.
            "sslcertmode": "allow",
            # As if from ``PGAPPNAME``: libpq re-derives it for the clone.
            "application_name": "from-the-environment",
        }
    )
    con.con.autocommit = True

    recorded = {}

    def fake_connect(**kwargs):
        recorded.update(kwargs)
        return _FakeConnection()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(PostgresBackend, "_post_connect", lambda self: None)

    clone = con.clone()

    # ``do_connect`` still defaults the encoding, so the wire is configured ...
    assert recorded["client_encoding"] == "utf8"
    # ... ``_post_connect`` still turns preparation off ...
    assert clone.con.prepare_threshold is None
    # ... what the caller passed survives ...
    assert clone._con_kwargs["options"] == "-c search_path=mine"
    # ... and nothing the caller did not pass comes back as a caller argument.
    for key in ("client_encoding", "sslcertmode", "application_name"):
        assert key not in clone._con_kwargs
        assert key not in clone._profile.kwargs_dict


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


@pytest.mark.parametrize(
    ("blocked", "imports"),
    [
        pytest.param("adbc_driver_postgresql", True, id="without-the-driver"),
        pytest.param("adbc_driver_manager", False, id="without-the-manager"),
    ],
)
def test_import_needs_the_driver_manager_but_not_the_driver(
    blocked: str, imports: bool
) -> None:
    """The psycopg baseline is only real if the backend imports without the
    accelerator. A fresh process, because this one has already imported both;
    ``None`` in ``sys.modules`` makes the import raise as an absent package
    does. The manager half pins the limit the ADR records: the postgres
    backend imports it at module scope."""
    code = f"import sys; sys.modules[{blocked!r}] = None; import xorq.backends.redshift"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert (result.returncode == 0) is imports, result.stderr


def test_clone_keeps_a_client_encoding_the_caller_passed(
    postgres_utils: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sibling test above covers the IMPLICIT case -- nobody passed one, so
    the DSN's value must not be reacquired. This is the other half, and it was
    broken once: a drop list was dissoc-ed from the MERGED dict, which already
    held ``_con_kwargs``, so a caller who *did* ask for ``latin1`` got a clone
    silently dialling ``utf8``.

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
            # A sentinel distinct from the caller's ``latin1``, so the
            # assertions below can tell which dict a value came from. Live
            # libpq would echo ``latin1`` here.
            "client_encoding": "UNICODE",
        }
    )
    con.con.autocommit = True

    recorded = {}

    def fake_connect(**kwargs):
        recorded.update(kwargs)
        return _FakeConnection()

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(PostgresBackend, "_post_connect", lambda self: None)

    clone = con.clone()

    # The caller asked for it, so it is theirs to keep -- on the wire ...
    assert recorded["client_encoding"] == "latin1"
    # ... and in the clone's own kwargs, so the profile agrees with the source.
    assert clone._con_kwargs["client_encoding"] == "latin1"
    # The DSN's ``UNICODE`` is still what gets dropped, not the caller's value.
    assert clone._con_kwargs["client_encoding"] != "UNICODE"


@pytest.mark.parametrize(
    "caller_kwargs",
    [
        pytest.param({}, id="defaults"),
        pytest.param({"schema": "s"}, id="schema"),
        pytest.param({"client_encoding": "latin1"}, id="caller-encoding"),
    ],
)
def test_clone_hashes_equal_to_its_source(
    monkeypatch: pytest.MonkeyPatch, caller_kwargs: dict
) -> None:
    """The invariant the two tests above guard piecewise: a cloned-then-built
    artifact hashes the same as one built off the source.

    Both connections go through the real ``connect`` -> ``do_connect`` path,
    so ``_con_kwargs`` is what a caller actually gets, and the fake reports
    what libpq's ``get_parameters`` does: ``port`` as a string, ``schema``
    absent (``_post_connect`` applies it with ``set_config``, not libpq), and
    ``client_encoding`` echoing the value the client sent, and ``sslcertmode``
    reported unasked, as libpq 17+ does. The string ``port`` is normalised by
    ``Profile.from_con``, so it is part of the path under test, not noise. If
    ``clone`` carries DSN keys the caller did not pass, every case fails here
    on ``sslcertmode``, and the first two on ``client_encoding`` as well.
    ``test_clone_hashes_equal_with_a_real_libpq`` checks the same against a
    server.
    """
    connect_kwargs = {
        "host": "example.invalid",
        "user": "u",
        "password": "static",
        "database": "d",
        **caller_kwargs,
    }
    dsn = {
        "host": "example.invalid",
        "port": str(redshift_module.DEFAULT_PORT),
        "user": "u",
        "dbname": "d",
        "client_encoding": caller_kwargs.get("client_encoding", "utf8"),
        "sslcertmode": "allow",
    }

    class _FakeInfo:
        encoding = "utf-8"

        def get_parameters(self) -> dict:
            return dict(dsn)

    def fake_connect(**kwargs):
        con = _FakeConnection()
        con.info = _FakeInfo()
        con.autocommit = kwargs["autocommit"]
        return con

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(PostgresBackend, "_post_connect", lambda self: None)

    source = RedshiftBackend().connect(**connect_kwargs)
    clone = source.clone()

    assert clone._profile.kwargs_dict == source._profile.kwargs_dict
    assert clone._profile.content_hash == source._profile.content_hash


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
        port = redshift_module.DEFAULT_PORT

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
    monkeypatch.setattr(PostgresBackend, "_post_connect", lambda self: None)

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
