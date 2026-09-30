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
import json
import re
import subprocess
import sys
import typing
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
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

import adbc_driver_manager.dbapi  # noqa: E402

# psycopg's own client-side placeholder binder, for the fake cursors below.
# ``psycopg._queries`` is private, and is used anyway because it refuses
# exactly what a live ``cursor.execute`` would -- a lone ``%``, a placeholder
# with no parameter -- and needs no connection to do it.
from psycopg._queries import PostgresQuery  # noqa: E402
from psycopg.adapt import Transformer  # noqa: E402

import xorq  # noqa: E402
import xorq.api as xo  # noqa: E402
import xorq.backends.redshift as redshift_module  # noqa: E402
import xorq.common.exceptions as exc  # noqa: E402
import xorq.ibis_yaml.translate  # noqa: E402, F401 -- registers the yaml rules
import xorq.vendor.ibis.expr.datatypes as dt  # noqa: E402
from xorq.backends.postgres import Backend as PostgresBackend  # noqa: E402
from xorq.backends.redshift import Backend as RedshiftBackend  # noqa: E402
from xorq.ibis_yaml.common import TranslationContext  # noqa: E402
from xorq.vendor.ibis.backends.profiles import (  # noqa: E402
    Profile,
    check_for_exposed_secrets,
    con_name_to_secret_keys,
)
from xorq.vendor.ibis.expr import types as ir  # noqa: E402


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


class _FakeAdapters:
    def register_loader(self, oid, loader) -> None:
        pass


class _FakeConnection:
    def __init__(self, rows: tuple = (), prepare_threshold: int | None = 5) -> None:
        self.log: list = []
        self.rows = rows
        # psycopg's defaults: a decodable encoding, and a threshold of 5.
        self.info = _FakeEncodingInfo("utf-8")
        self.prepare_threshold = prepare_threshold
        # ``_post_connect`` registers the ``VARBYTE`` loader here.
        self.adapters = _FakeAdapters()

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
        ("execute", 'CREATE TABLE "t" ("a" BIGINT, "b" VARCHAR(65535))'),
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

    assert executed(con) == ['CREATE TABLE "t" ("a" BIGINT, "b" VARCHAR(65535))']
    assert not [entry for entry in con.con.log if entry[0] == "executemany"]


def test_psycopg_ingest_accepts_a_table_like_the_adbc_branch_does():
    """``adbc_ingest`` takes a ``pa.Table``, and iterating one yields *columns*
    -- so the naive psycopg loop would fail on a missing ``num_rows`` for an
    input the accelerator handles. Which branch runs has to stay an
    implementation detail."""
    con = make_offline_con()

    con.read_record_batches(pa.table({"a": [1], "b": ["x"]}), table_name="t")

    assert con.con.log == [
        ("execute", 'CREATE TABLE "t" ("a" BIGINT, "b" VARCHAR(65535))'),
        ("executemany", 'INSERT INTO "t" ("a", "b") VALUES (%s, %s)', [(1, "x")]),
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        pytest.param(
            "create",
            ['CREATE TABLE "t" ("a" BIGINT, "b" VARCHAR(65535))'],
            id="create-creates",
        ),
        pytest.param("append", [], id="append-creates-nothing"),
        pytest.param(
            "replace",
            [
                'DROP TABLE IF EXISTS "t"',
                'CREATE TABLE "t" ("a" BIGINT, "b" VARCHAR(65535))',
            ],
            id="replace-drops-then-creates",
        ),
        pytest.param(
            "create_append",
            ['CREATE TABLE IF NOT EXISTS "t" ("a" BIGINT, "b" VARCHAR(65535))'],
            id="create-append-tolerates-existing",
        ),
    ],
)
def test_psycopg_ingest_modes_match_their_adbc_meanings(
    mode: str, expected: list[str]
) -> None:
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

    assert executed(con) == [
        'CREATE TEMPORARY TABLE "t" ("a" BIGINT, "b" VARCHAR(65535))'
    ]


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


def test_ingest_ddl_emits_the_measured_redshift_spellings() -> None:
    """The two divergences this used to pin as *suspected* are now measured.

    It was a tripwire, deliberately asserting the broken value so that fixing
    it would be a visible change -- ``VARCHAR`` and ``TIMESTAMP(6)``, both on
    documentation rather than observation. A live warehouse settled them on
    2026-09-25 and both are rejections, not stylistic differences:
    ``TIMESTAMP(6)`` raises ``FeatureNotSupported: timestamp column does not
    support precision``, and a bare ``VARCHAR`` is ``VARCHAR(256)``, so the
    257th byte fails the ``INSERT``.

    So this is now an acceptance assertion rather than a tripwire: the exact
    string below was executed against Redshift and accepted. The tripwire had
    done its job -- but note the lesson in how long it took to be read, since
    the live session held the same day ingested ``bigint`` and ``'a'`` and
    checked neither suspect.
    """
    con = make_offline_con()

    schema = pa.schema([("s", pa.string()), ("ts", pa.timestamp("us"))])
    con.read_record_batches(
        make_reader({"s": ["x"], "ts": [None]}, schema=schema), table_name="t"
    )

    (create,) = executed(con)
    assert create == 'CREATE TABLE "t" ("s" VARCHAR(65535), "ts" TIMESTAMP)'


# Every Arrow type a caller can put through ingest, and the Redshift spelling
# it must render to. The right-hand column is not derived -- each entry was
# executed against a live Redshift Serverless warehouse on 2026-09-25, as a
# ``CREATE`` inside a rolled-back transaction, and accepted.
#
# The table is what was measured on the warehouse, not the mapper's whole
# range; the divergence tests in ``test_redshift_dialect.py`` pin the rest of
# the range against ``PostgresType`` offline. The defect this replaces was not
# that one mapping was wrong; it was that only two types had any assertion at
# all, so five rejections had no tripwire, not even a wrong one.
_MEASURED_REDSHIFT_DDL_TYPES = (
    (pa.string(), "VARCHAR(65535)"),
    (pa.timestamp("us"), "TIMESTAMP"),
    (pa.timestamp("us", tz="UTC"), "TIMESTAMP WITH TIME ZONE"),
    (pa.date32(), "DATE"),
    (pa.time64("us"), "TIME"),
    (pa.decimal128(10, 2), "DECIMAL(10, 2)"),
    (pa.binary(), "VARBYTE"),
    (pa.uint8(), "SMALLINT"),
    (pa.uint16(), "INTEGER"),
    (pa.uint32(), "BIGINT"),
    (pa.uint64(), "DECIMAL(20, 0)"),
    (pa.int64(), "BIGINT"),
    (pa.float64(), "DOUBLE PRECISION"),
    (pa.bool_(), "BOOLEAN"),
)


@pytest.mark.parametrize(
    ("arrow_type", "expected"),
    _MEASURED_REDSHIFT_DDL_TYPES,
    ids=[str(t) for t, _ in _MEASURED_REDSHIFT_DDL_TYPES],
)
def test_ingest_ddl_renders_each_type_as_redshift_accepts_it(
    monkeypatch: pytest.MonkeyPatch, arrow_type: pa.DataType, expected: str
) -> None:
    con = make_offline_con()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")

    schema = pa.schema([("c", arrow_type)])
    con.read_record_batches(make_reader({"c": [None]}, schema=schema), table_name="t")

    (create,) = executed(con)
    assert create == f'CREATE TABLE "t" ("c" {expected})'


@pytest.mark.parametrize(
    ("arrow_type", "match"),
    [
        pytest.param(pa.list_(pa.int64()), "no array type", id="list"),
        pytest.param(pa.struct([("a", pa.int64())]), "no struct type", id="struct"),
        pytest.param(pa.map_(pa.string(), pa.string()), "no map type", id="map"),
    ],
)
def test_nested_types_raise_before_any_sql(
    monkeypatch: pytest.MonkeyPatch, arrow_type: pa.DataType, match: str
) -> None:
    """Redshift has no array, map or struct type, and ``SUPER`` is not the
    answer: measured, ``CREATE TABLE (c SUPER)`` is accepted but the psycopg
    ingest binds ``batch.to_pydict()`` values directly and Redshift rejects a
    bound Python list with ``DatatypeMismatch``.

    So emitting SUPER would move the failure from CREATE to INSERT and make it
    less legible. The mapper raises instead, naming the type and what to do --
    and, like the null-column guard, before any statement is issued.
    """
    con = make_offline_con()
    monkeypatch.setattr(con, "_adbc_unavailable_reason", lambda: "no driver")

    schema = pa.schema([("c", arrow_type)])
    with pytest.raises(exc.UnsupportedBackendType, match=match):
        con.read_record_batches(
            make_reader({"c": [None]}, schema=schema), table_name="t"
        )

    assert con.con.log == []


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

    A statement with ``params`` is bound through psycopg's own placeholder
    binder first, as a live cursor would bind it, and the result -- the
    statement with ``$n`` placeholders, and the parameter values in their
    order -- is recorded too. Without that, a ``%`` psycopg refuses or a placeholder
    with no parameter passed every test here and failed every live lookup.
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
        if params is not None:
            query = PostgresQuery(Transformer())
            query.convert(sql, params)
            self._con.bound.append(
                (query.query.decode(), [p and bytes(p) for p in query.params])
            )
        self._last = sql
        return self

    def fetchall(self) -> list:
        # The permanent and temporary lookups hit different views, so the fake
        # answers them separately: a test that supplies only ``temp_rows`` must
        # not see them returned for a ``svv_all_columns`` query, or the
        # fallback would look exercised when it was not. The temp-schema
        # lookup reads ``svv_columns`` too, but selects only ``table_schema``,
        # and is answered from the connection's ``temp_schemas``, which a test
        # may change between calls.
        last = self._last or ""
        if "SELECT table_schema" in last:
            return [(schema,) for schema in self._con.temp_schemas]
        if "svv_columns" in last:
            return list(self._temp_rows)
        return list(self._rows)

    def fetchone(self) -> tuple | None:
        return next(iter(self.fetchall()), None)


class _IntrospectionConnection(_FakeConnection):
    def __init__(
        self,
        rows: tuple = (),
        description: tuple | None = None,
        temp_rows: tuple = (),
        temp_schemas: tuple = (),
    ) -> None:
        super().__init__()
        self.temp_schemas = list(temp_schemas)
        self.params = []
        self.bound = []
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
    temp_schemas: tuple = (),
) -> RedshiftBackend:
    con = make_offline_con()
    con.con = _IntrospectionConnection(
        rows=rows,
        description=description,
        temp_rows=temp_rows,
        temp_schemas=temp_schemas,
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
SVV_ROWS_NAMES = tuple(row[0] for row in SVV_ROWS)


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
    # And each placeholder is one psycopg fills from those params.
    (bound, values) = con.con.bound[-1]
    assert "COALESCE($1, current_database())" in bound
    assert "COALESCE($2, current_schema())" in bound
    assert "table_name = $3" in bound
    assert values == [None, b"analytics", b"sales"]


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
    work, not only the three-part-named one.

    The fake answers ``current_schema()`` like any other SQL, so it cannot
    catch a function Redshift refuses in this position -- the reference lists
    ``CURRENT_SCHEMA`` among the leader-node-only functions. That is covered by
    the live runs instead: an unqualified ``con.table`` passed against a
    warehouse, as an administrator and as a read-only user.
    """
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
    # The statement carries psycopg's named placeholders, which psycopg binds
    # and sqlglot 23.6.3 -- the supported floor -- cannot parse. Each becomes a
    # literal so the shape is what gets checked; that the values are bound
    # rather than interpolated is asserted separately.
    parsed = sg.parse_one(re.sub(r"%\((\w+)\)s", r"'\1'", sql), read=con.dialect)

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
    """The probe wraps the caller's query instead of appending ``LIMIT 0``.

    Appending through the AST would not be wrong -- sqlglot's ``.limit(0)``
    replaces an existing ``LIMIT`` and binds to a whole ``UNION`` -- but the
    wrap gives every probe one shape. This pins that the caller's own
    ``LIMIT`` survives inside it, neither replaced nor hoisted."""
    con = make_introspection_con(description=(_FakeColumn("a", 23, "int4"),))
    con._get_schema_using_query("SELECT a FROM t LIMIT 5")

    (sql,) = issued(con)
    parsed = sg.parse_one(sql, read=con.dialect)

    # The outer LIMIT is the probe's; the inner one is the caller's, still
    # bound to the subquery rather than replaced or hoisted.
    assert parsed.args["limit"].expression.this == "0"
    inner = parsed.find(sge.Subquery).this
    assert inner.args["limit"].expression.this == "5"


@pytest.mark.parametrize(
    ("oid", "expected"),
    [
        pytest.param(4000, dt.NamedUnknown(raw_type="super"), id="super"),
        pytest.param(6551, dt.Binary(), id="varbyte"),
        pytest.param(3000, dt.NamedUnknown(raw_type="geometry"), id="geometry"),
        pytest.param(3001, dt.NamedUnknown(raw_type="geography"), id="geography"),
        pytest.param(2935, dt.NamedUnknown(raw_type="hllsketch"), id="hllsketch"),
        # What a live description reports for a GEOMETRY value, not pg_type's.
        pytest.param(3999, dt.NamedUnknown(raw_type="geometry"), id="geometry-value"),
        pytest.param(
            1188,
            dt.NamedUnknown(raw_type="intervaly2m"),
            id="interval-year-to-month",
        ),
        pytest.param(
            1190,
            dt.NamedUnknown(raw_type="intervald2s"),
            id="interval-day-to-second",
        ),
    ],
)
def test_query_path_maps_redshift_type_oids_as_the_catalog_path_maps_names(
    oid: int, expected: dt.DataType
) -> None:
    """A result description names Redshift's own types only by OID, so the
    query path needs the OIDs to agree with the catalog path: VARBYTE is binary
    on both, and SUPER binds under its name on both."""
    con = make_introspection_con(description=(_FakeColumn("c", oid),))

    assert con._get_schema_using_query("SELECT c FROM t")["c"] == expected


class _AdaptingConnection(_FakeConnection):
    """``_FakeConnection`` carrying a real psycopg adapters map, as a live
    connection does, so a loader registered on it can be read back."""

    def __init__(self) -> None:
        super().__init__()
        self.adapters = psycopg.adapt.AdaptersMap(psycopg.adapters)


def test_varbyte_values_are_decoded_from_hex_on_the_psycopg_path() -> None:
    """psycopg does not know OID 6551, so it returned ``VARBYTE`` as the hex
    text Redshift sends, and the binary cast made that text's ASCII bytes:
    measured live, ``b"\\xab"`` came back as ``b"ab"`` with ADBC off."""
    con = make_offline_con()
    con.con = _AdaptingConnection()

    con._post_connect()

    loader = con.con.adapters.get_loader(
        redshift_module.VARBYTE_OID, psycopg.pq.Format.TEXT
    )
    assert loader(redshift_module.VARBYTE_OID).load(b"ab00ff") == b"\xab\x00\xff"


def test_redshift_type_oids_are_unknown_to_psycopg() -> None:
    """The table is consulted first, so an OID psycopg also knew would be
    silently renamed. Pins that none of them is in its registry."""
    for oid in RedshiftBackend._REDSHIFT_TYPE_OIDS:
        assert psycopg.postgres.types.get(oid) is None, oid


def test_get_schema_using_query_binds_an_unknown_oid_by_its_number() -> None:
    """An OID neither psycopg nor this backend knows used to fail the whole
    query for one such column, as the catalog path once did. It binds instead,
    named by its number, and is refused where it is read."""
    con = make_introspection_con(
        description=(_FakeColumn("id", 23, "int4"), _FakeColumn("s", 99999))
    )

    schema = con._get_schema_using_query("SELECT id, s FROM t")

    assert schema["id"] == dt.Int32(nullable=True)
    assert schema["s"] == dt.NamedUnknown(raw_type="type OID 99999", nullable=True)


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
    shared implementation would reach every postgres user.

    Asserted on the statements postgres emits, not on the methods being
    different objects, which any override satisfies. The fake cannot answer
    what follows the first statement, so each call's failure is suppressed
    and only what it sent is read.
    """
    con = PostgresBackend()
    con.con = _IntrospectionConnection(rows=(("a", "integer", True),))

    con.get_schema("t", database="public")
    with contextlib.suppress(Exception):
        con._get_schema_using_query("SELECT 1 AS a")

    (catalog_sql, view_sql, *_rest) = executed(con)
    assert "pg_catalog.pg_attribute" in catalog_sql
    assert view_sql.upper().startswith("CREATE TEMPORARY VIEW")


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
    # Found in the temporary catalog, so the permanent one is never asked.
    assert not any("svv_all_columns" in sql for sql in statements)
    (temp_sql,) = [sql for sql in statements if "svv_columns" in sql]
    # The scoping is the whole point: without it the fallback means "anything
    # this session can see" rather than "a temporary table". Asserted on the
    # text psycopg binds, where ``%%`` has become the one ``%`` Redshift sees;
    # a single ``%`` in the source is refused by psycopg before that.
    ((bound, values),) = [b for b in con.con.bound if "svv_columns" in b[0]]
    assert "LIKE 'pg^_temp^_%' ESCAPE '^'" in bound
    assert "table_name = $1" in bound
    assert values == [b"xorq_temp_abc123"]


def test_table_reaches_the_temp_fallback_unqualified() -> None:
    """The ingest path ends in ``self.table(name)``, not ``get_schema``, so the
    fallback is exercised through the real ``table``: an override that
    defaulted ``database`` there would disable it while every direct call to
    ``get_schema`` stayed green."""
    con = make_introspection_con(rows=(), temp_rows=(("a", "integer", "NO", 32, 0),))

    t = RedshiftBackend.table(con, "xorq_temp_abc123")

    assert t.schema() == xo.schema({"a": dt.Int32(nullable=False)})
    assert any("svv_columns" in sql for sql in issued(con))


def test_a_temporary_ingest_binds_the_table_it_created() -> None:
    """The round trip the two tests above cover in halves: a
    ``temporary=True`` ingest ends in ``self.table(name)``, and that bind must
    find the table the ingest just created.

    ``make_offline_con`` stubs ``table`` so the ingest tests never reach the
    bind; this one removes the stub, so the real ``table`` runs against the
    same connection the ``CREATE TEMPORARY TABLE`` went over. The catalog rows
    are how ``svv_columns`` reports the two columns that ``CREATE`` declares.
    """
    con = make_introspection_con(
        rows=(),
        temp_rows=(
            ("a", "bigint", "YES", 64, 0),
            ("b", "character varying", "YES", None, None),
        ),
    )
    del con.table

    t = con.read_record_batches(
        make_reader({"a": [1], "b": ["x"]}), table_name="t", temporary=True
    )

    assert t.get_name() == "t"
    assert t.schema() == xo.schema({"a": dt.Int64(), "b": dt.String()})
    create, insert, lookup = [(kind, sql) for (kind, sql, *_) in con.con.log]
    assert create == (
        "execute",
        'CREATE TEMPORARY TABLE "t" ("a" BIGINT, "b" VARCHAR(65535))',
    )
    assert insert[0] == "executemany"
    # Resolved in the temporary catalog, for the name the ingest created.
    assert "svv_columns" in lookup[1]
    ((_bound, values),) = con.con.bound
    assert values == [b"t"]


@pytest.mark.parametrize(
    "database",
    [
        pytest.param(("warehouse", "analytics"), id="tuple"),
        pytest.param("warehouse.analytics", id="dotted"),
    ],
)
def test_table_routes_a_catalog_qualified_name_to_the_catalog_query(
    database: tuple[str, str] | str,
) -> None:
    """A 2-tuple or dotted name is the only way ``catalog`` is ever supplied,
    so the scoping ``get_schema`` documents depends on ``table`` splitting it."""
    con = make_introspection_con(rows=SVV_ROWS)

    RedshiftBackend.table(con, "sales", database=database)

    (_sql, params) = last_call(con)
    assert params["catalog"] == "warehouse"
    assert params["schema"] == "analytics"


def test_an_unqualified_lookup_binds_the_temporary_table_that_shadows() -> None:
    """Measured on a live warehouse: with a temporary ``offers`` and a
    permanent ``xorq_test.offers`` both present, unqualified SQL reads the
    temporary one, and ``con.table("offers")`` bound the permanent one's
    eleven columns. The compiled query names the table unqualified, so the
    schema described a different table than the one it read. A qualified
    lookup still binds the permanent table.
    """
    con = make_introspection_con(
        rows=SVV_ROWS, temp_rows=(("temp_only_col", "integer", "YES", 32, 0),)
    )

    assert con.get_schema("sales").names == ("temp_only_col",)
    assert con.get_schema("sales", database="public").names == SVV_ROWS_NAMES


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


# One row per Redshift type the warehouse actually holds: how svv_all_columns
# spells it (with the precision and scale it reports), how a result
# description reports the same column (OID and psycopg type_display), and the
# dtype both introspection paths must bind. Every row was read off one real
# column on a live warehouse, both sides from that column, except the two
# interval rows, whose catalog spelling was read from svv_columns for a
# temporary table (svv_all_columns lists none). A type with no row here is
# unmeasured on at least one side; add it only from a measurement.
TYPE_PARITY_ROWS = (
    ("bigint", 64, 0, 20, "int8", dt.Int64(nullable=True)),
    ("integer", 32, 0, 23, "int4", dt.Int32(nullable=True)),
    ("numeric", 12, 4, 1700, "numeric(12,4)", dt.Decimal(12, 4, nullable=True)),
    ("boolean", None, None, 16, "bool", dt.Boolean(nullable=True)),
    ("character varying", None, None, 1043, "varchar(32)", dt.String(nullable=True)),
    ("date", None, None, 1082, "date", dt.Date(nullable=True)),
    (
        "timestamp without time zone",
        None,
        None,
        1114,
        "timestamp",
        dt.Timestamp(scale=6, nullable=True),
    ),
    ("binary varying", None, None, 6551, "6551", dt.Binary(nullable=True)),
    ("super", None, None, 4000, "4000", dt.NamedUnknown(raw_type="super")),
    (
        "intervaly2m",
        None,
        None,
        1188,
        "1188",
        dt.NamedUnknown(raw_type="intervaly2m"),
    ),
    (
        "intervald2s",
        None,
        None,
        1190,
        "1190",
        dt.NamedUnknown(raw_type="intervald2s"),
    ),
)


@pytest.mark.parametrize(
    ("data_type", "precision", "scale", "oid", "type_display", "expected"),
    [pytest.param(*row, id=row[0]) for row in TYPE_PARITY_ROWS],
)
def test_both_introspection_paths_bind_a_measured_type_alike(
    data_type: str,
    precision: int | None,
    scale: int | None,
    oid: int,
    type_display: str,
    expected: dt.DataType,
) -> None:
    """``con.table`` and ``con.sql`` reach one column's dtype by different
    routes -- a catalog spelling on one, an OID and psycopg's name for it on
    the other -- and the two drifted apart once per type this backend learned
    about: ``bpchar``, ``VARBYTE``, an unknown OID, the interval types. One
    table, asserted through both, is what keeps them together."""
    from_catalog = RedshiftBackend._schema_from_catalog_rows(
        [("c", data_type, "YES", precision, scale)]
    )["c"]
    query_con = make_introspection_con(
        description=(_FakeColumn("c", oid, type_display),)
    )
    from_query = query_con._get_schema_using_query("SELECT c FROM t")["c"]

    assert from_catalog == expected
    assert from_query == expected


QUERY_PATH_TYPE_MAPPING = json.loads(
    (
        Path(__file__).parent / "fixtures" / "redshift_query_path_type_mapping.json"
    ).read_text()
)


@pytest.mark.parametrize(
    ("oid", "type_display", "expected"),
    [
        pytest.param(*entry, id=f"{entry[0]}-{entry[1]}")
        for entry in QUERY_PATH_TYPE_MAPPING["entries"]
    ],
)
def test_the_query_path_maps_every_pinned_type_alike_on_every_sqlglot(
    oid: int, type_display: str, expected: str
) -> None:
    """Every name psycopg's registry can put in a result description, pinned
    with the dtype it binds as. The inputs are pinned too, not read from the
    installed psycopg, so only sqlglot and this backend can move a result --
    which is the point: the lowest-direct CI job runs this at sqlglot 23.6.3,
    where ``int8`` used to bind as an 8-bit integer while the locked version
    got it right, and nothing else looked at every name.

    A failure here is a changed mapping. If the change is intended,
    regenerate the fixture and review its diff line by line::

        for info in psycopg.postgres.types:
            for oid in (info.oid, info.array_oid):
                display = info.get_type_display(oid=oid, fmod=-1)
                ...repr(RedshiftBackend._column_dtype_from_description(column))

    plus one entry per ``RedshiftBackend._REDSHIFT_TYPE_OIDS`` OID, whose
    display psycopg renders as the bare number.
    """
    column = _FakeColumn("c", oid, type_display)

    assert repr(RedshiftBackend._column_dtype_from_description(column)) == expected


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


def test_query_path_maps_the_single_byte_char_type() -> None:
    """``"char"`` -- OID 18, with the quotes -- is PostgreSQL's one-byte type,
    which ``pg_catalog`` columns such as ``pg_class.relkind`` carry. Without
    its alias it binds as unknown, and ``SELECT relkind FROM pg_class``
    through ``con.sql`` is refused on read."""
    con = make_introspection_con(
        description=(
            _FakeColumn("relkind", 18, '"char"'),
            _FakeColumn("kinds", 1002, '"char"[]'),
        )
    )

    schema = con._get_schema_using_query("SELECT relkind, kinds FROM pg_class")

    assert schema["relkind"] == dt.String(nullable=True)
    assert schema["kinds"] == dt.Array(dt.String(nullable=True), nullable=True)


@pytest.mark.parametrize(
    "data_type",
    [
        pytest.param("super", id="super"),
        pytest.param("hllsketch", id="hllsketch"),
        pytest.param("geometry", id="geometry"),
        pytest.param("geography", id="geography"),
        # How svv_columns spells the interval column types (measured); their
        # values arrive through psycopg as text.
        pytest.param("intervaly2m", id="interval-year-to-month"),
        pytest.param("intervald2s", id="interval-day-to-second"),
        pytest.param("interval year to month", id="interval-ym-as-ddl"),
    ],
)
@pytest.mark.parametrize(
    ("is_nullable", "nullable"),
    [pytest.param("NO", False, id="not-null"), pytest.param("YES", True, id="null")],
)
def test_get_schema_binds_a_redshift_only_type_as_unknown(
    data_type: str, is_nullable: str, nullable: bool
) -> None:
    """One unmappable column used to fail the whole table, so a table carrying
    a ``SUPER`` column could not be bound at all. It now binds, with that
    column present and typed ``unknown`` under its catalog spelling; touching
    it is what fails.

    ``geometry`` and ``geography`` are here although they *parse*, into ibis
    ``GeoSpatial`` types: this backend still compiles as PostgreSQL, so a geo
    operation on such a column would emit a PostGIS call Redshift does not
    implement.

    Nullability is asserted both ways because the mapper's own ``unknown``
    fallback drops ``nullable=``, so a ``SUPER NOT NULL`` column taken from it
    would come back nullable.
    """
    con = make_introspection_con(
        rows=(
            ("id", "integer", "NO", 32, 0),
            ("c", data_type, is_nullable, None, None),
        )
    )

    schema = con.get_schema("t", database="public")

    assert schema.names == ("id", "c")
    assert schema["id"] == dt.Int32(nullable=False)
    assert schema["c"] == dt.NamedUnknown(raw_type=data_type, nullable=nullable)


@pytest.mark.parametrize(
    "data_type",
    [
        # The spelling svv_all_columns actually emits for VARBYTE(16),
        # measured live: the view definition rewrites the DDL keyword.
        pytest.param("binary varying", id="binary-varying"),
        pytest.param("varbyte", id="varbyte-as-written-by-a-user"),
    ],
)
def test_get_schema_maps_varbyte_to_binary(data_type: str) -> None:
    """``VARBYTE`` is variable-length binary data, which ``dt.Binary`` is.
    sqlglot's postgres dialect parses neither spelling, so without the alias
    both would bind as ``unknown``."""
    con = make_introspection_con(rows=(("c", data_type, "NO", None, None),))

    assert con.get_schema("t", database="public")["c"] == dt.Binary(nullable=False)


def test_query_path_binds_an_oid_psycopg_names_but_cannot_map() -> None:
    """The guard tested only ``info is None`` -- whether psycopg could *name*
    the OID -- and not whether the type mapper could use the name.

    ``oid`` (26) and its ``regclass``/``regproc`` relatives are named and make
    the upstream mapper raise a bare ``AttributeError: 'str' object has no
    attribute 'name'`` from inside ``to_ibis``. Redshift does expose
    ``pg_catalog`` views, so ``SELECT oid FROM pg_class`` is reachable.
    """
    con = make_introspection_con(description=(_FakeColumn("oid", 26, "oid"),))

    schema = con._get_schema_using_query("SELECT oid FROM pg_class")

    assert schema["oid"] == dt.NamedUnknown(raw_type="oid", nullable=True)


def test_unsupported_type_keeps_the_nullability_it_was_given() -> None:
    """``SqlglotType.from_string`` drops ``nullable=`` on its ``dt.unknown``
    fallback, so a ``SUPER NOT NULL`` column used to come back with the wrong
    type *and* the wrong nullability. Nothing may reach that branch now."""
    mapper = RedshiftBackend.compiler.type_mapper

    assert mapper.from_string("integer", nullable=False) == dt.Int32(nullable=False)
    assert mapper.from_string("integer", nullable=True) == dt.Int32(nullable=True)
    with pytest.raises(exc.UnsupportedBackendType):
        mapper.from_string("super", nullable=False)
    assert mapper.from_string("varbyte(16)", nullable=False) == dt.Binary(
        nullable=False
    )


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        # psycopg's name for OID 20, so every BIGINT through ``con.sql``.
        pytest.param("int8", dt.Int64(nullable=True), id="int8"),
        pytest.param("int8[]", dt.Array(dt.Int64(nullable=True)), id="int8-array"),
        pytest.param("float", dt.Float64(nullable=True), id="float"),
    ],
)
def test_integer_and_float_widths_are_redshift_s_on_every_sqlglot(
    spelling: str, expected: dt.DataType
) -> None:
    """At sqlglot 23.6.3 -- what the lowest-direct job installs -- ``int8`` is
    an 8-bit integer and ``float`` a 32-bit one. On Redshift they are
    ``BIGINT`` and ``DOUBLE PRECISION``. Unaliased, ``COUNT(*)`` through
    ``con.sql`` bound as int8 there, overflowing above 127."""
    mapper = RedshiftBackend.compiler.type_mapper

    assert mapper.from_string(spelling, nullable=True) == expected


def test_query_path_binds_bigint_as_int64() -> None:
    con = make_introspection_con(description=(_FakeColumn("n", 20, "int8"),))

    schema = con._get_schema_using_query("SELECT COUNT(*) AS n FROM t")

    assert schema["n"] == dt.Int64(nullable=True)


@pytest.mark.parametrize(
    "spelling",
    [
        # psycopg's name for OID 1186 with a precision.
        pytest.param("interval(6)", id="interval-with-precision"),
        pytest.param("numeric(0,0)", id="zero-precision"),
        pytest.param("numeric(10,20)", id="scale-above-precision"),
        pytest.param("timestamp(10)", id="timestamp-scale-out-of-range"),
    ],
)
def test_a_modifier_the_dtype_rejects_binds_the_column_as_unknown(
    spelling: str,
) -> None:
    """The dtype constructors reject these with ``ValueError``,
    ``XorqTypeError`` or a validation error, none of which named the column.
    They escaped ``_column_dtype`` -- the binder both introspection paths
    share -- so one such column failed its whole table. Some
    are version-dependent: ``interval(6)`` maps at sqlglot 23.6.3 and raises
    at 28.6.0, so both outcomes are accepted and only the escape is not."""
    dtype = RedshiftBackend._column_dtype(spelling, nullable=True)

    assert isinstance(dtype, (dt.NamedUnknown, dt.Interval))
    if spelling != "interval(6)":
        assert dtype == dt.NamedUnknown(raw_type=spelling, nullable=True)


def test_a_type_the_mapper_resolves_by_table_keeps_its_nullability() -> None:
    """``PostgresType.unknown_type_strings`` hits return a dtype built with the
    mapper's default nullability, whatever was asked for: upstream,
    ``name NOT NULL`` -- ``pg_class.relname`` -- comes back nullable. The
    mapper restores what it was given."""
    mapper = RedshiftBackend.compiler.type_mapper

    assert mapper.from_string("name", nullable=False) == dt.String(nullable=False)
    assert mapper.from_string("name", nullable=True) == dt.String(nullable=True)


# --- an unmappable column fails where it is used, not where it is bound -----


# ``offers`` carries a SUPER column between two mappable ones, which is the
# shape of the field report: most required tables have one or more.
OFFERS_CATALOG_ROWS = (
    ("id", "integer", "NO", 32, 0),
    ("payload", "super", "NO", None, None),
    ("name", "character varying", "YES", None, None),
)
OFFERS_DATA_ROWS = ((1, "a"), (2, None))


class _ReadCursor(_IntrospectionCursor):
    """``_IntrospectionCursor`` that answers a data query with data rows.

    Catalog queries still get the catalog rows. ``fetchmany`` serves the
    psycopg read path, which drains a named cursor in chunks.
    """

    def __init__(self, con: _ReadConnection, *args: object) -> None:
        super().__init__(con, *args)
        self._data = con.data_rows

    def fetchall(self) -> list:
        if "svv_" in (self._last or ""):
            return super().fetchall()
        rows, self._data = self._data, ()
        return list(rows)

    def fetchmany(self, size: int) -> list:
        return self.fetchall()


class _ReadConnection(_IntrospectionConnection):
    def __init__(self, rows: tuple, data_rows: tuple) -> None:
        super().__init__(rows=rows)
        self.data_rows = data_rows

    def cursor(self, *args: object, **kwargs: object) -> _ReadCursor:
        return _ReadCursor(self, self._rows, self._description, self._temp_rows)


Offers = tuple[RedshiftBackend, ir.Table]


@pytest.fixture
def offers(monkeypatch: pytest.MonkeyPatch) -> Offers:
    """``offers`` bound through ``con.table``, readable over the psycopg path.

    ``make_offline_con`` stubs ``con.table``, so the unbound method is called
    to reach the real ``get_schema``. ADBC is switched off so the read stays
    on the fake connection rather than dialling anything.
    """
    con = make_offline_con()
    con.con = _ReadConnection(rows=OFFERS_CATALOG_ROWS, data_rows=OFFERS_DATA_ROWS)
    monkeypatch.setattr(RedshiftBackend, "_open_adbc_conn_or_none", lambda self: None)
    return con, RedshiftBackend.table(con, "offers", database="public")


def test_a_table_with_an_unmappable_column_binds_and_reads_its_other_columns(
    offers: Offers,
) -> None:
    """The point of binding: the columns that do map are readable through the
    table, without hand-building a source that leaves the SUPER column out."""
    con, t = offers

    assert t.schema()["payload"] == dt.NamedUnknown(raw_type="super", nullable=False)
    result = t.select("id", "name").to_pyarrow()

    assert result.to_pylist() == [{"id": 1, "name": "a"}, {"id": 2, "name": None}]


@pytest.mark.parametrize(
    "read",
    [
        pytest.param(lambda con, t: con.to_pyarrow_batches(t), id="con-batches"),
        pytest.param(lambda con, t: con.to_pyarrow(t), id="con-to-pyarrow"),
        pytest.param(lambda con, t: t.execute(), id="xo-execute"),
        pytest.param(lambda con, t: t.to_pyarrow_batches(), id="xo-batches"),
        pytest.param(lambda con, t: t.to_pyarrow(), id="xo-to-pyarrow"),
        pytest.param(
            lambda con, t: t.into_backend(xo.connect()).execute(), id="into-backend"
        ),
        pytest.param(
            lambda con, t: con.create_table("copy", t), id="create-table-from-expr"
        ),
        pytest.param(
            lambda con, t: con.create_table("copy", schema=t.schema()),
            id="create-table-from-schema",
        ),
        # ``insert`` truncates before it runs the pre-execute hooks, so with
        # ``overwrite`` a refusal from there left the target empty.
        pytest.param(
            lambda con, t: con.insert("copy", t, overwrite=True),
            id="insert-overwrite",
        ),
        pytest.param(lambda con, t: con.insert("copy", t), id="insert"),
        # Redshift casts a SUPER object or array to VARCHAR as NULL (measured),
        # so a cast read would silently return no data.
        pytest.param(
            lambda con, t: t.select("id", p=t.payload.cast("string")).execute(),
            id="cast",
        ),
        # Nothing returned is unmappable, but the struct has to be compiled
        # with the column's type, which fails with a KeyError naming no column.
        pytest.param(
            lambda con, t: (
                t.select(s=xo.struct({"p": t.payload, "i": t.id}))
                .unpack("s")
                .select("i")
                .execute()
            ),
            id="struct-unpacked-and-dropped",
        ),
        # The local side is uploaded into a placeholder on Redshift by a
        # transform pass, which used to run before the refusal did.
        pytest.param(
            lambda con, t: (
                xo.memtable({"id": [1], "z": [3]})
                .into_backend(con, "lz")
                .join(t, "id")
                .execute()
            ),
            id="join-with-a-local-table",
        ),
    ],
)
def test_touching_an_unmappable_column_raises_naming_it(
    offers: Offers, read: Callable[[RedshiftBackend, ir.Table], object]
) -> None:
    """Nothing downstream refuses an ``unknown`` column by name: the Arrow
    conversion raises naming only the type, and DDL fails with a bare
    ``KeyError``. The refusal is therefore explicit, on every entry point, and
    before anything reaches the connection -- a ``TRUNCATE``, an upload, or a
    read.
    """
    con, t = offers
    before = len(con.con.log)

    with pytest.raises(
        exc.UnmappableColumnError, match="'payload' \\(redshift type 'super'\\)"
    ) as excinfo:
        read(con, t)

    assert excinfo.value.columns == ("payload",)
    assert con.con.log[before:] == []


def test_a_column_that_is_referenced_but_not_returned_is_not_refused(
    offers: Offers,
) -> None:
    """The refusal is on what a read *returns*, since that is what would come
    back mistyped. Filtering on the column is evaluated server-side, on
    Redshift's own type, and returns only correctly typed columns."""
    con, t = offers
    con.con.data_rows = ((1,), (2,))

    result = t.filter(t.payload.notnull()).select("id").to_pyarrow()

    assert result.to_pylist() == [{"id": 1}, {"id": 2}]
    assert '"payload" IS NOT NULL' in executed(con)[-1]


def test_an_expression_with_an_unmappable_column_still_compiles(
    offers: Offers,
) -> None:
    """Compiling reads no data, so it must not raise: ``con.compile`` and
    ``xo.to_sql`` are how a caller inspects the SQL for such a table."""
    con, t = offers
    expr = t.select("id", "payload")

    assert '"payload"' in con.compile(expr)
    assert '"payload"' in str(xo.to_sql(expr))


def test_fake_column_type_displays_match_psycopg() -> None:
    """Pins the fakes above against psycopg's real registry.

    ``_FakeColumn`` takes ``type_display`` as a literal, which is only safe
    while those literals are what a live cursor would actually report.
    """
    expected = {
        (23, -1): "int4",
        (1043, -1): "varchar",
        (1700, ((8 << 16) | 2) + 4): "numeric(8,2)",
        (1114, -1): "timestamp",
        (1042, 3 + 4): "bpchar(3)",
        (26, -1): "oid",
        (18, -1): '"char"',
        (20, -1): "int8",
        (1002, -1): '"char"[]',
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
    ("query", "match"),
    [
        # sqlglot's Redshift dialect reads a bare VALUES as a call to a
        # function named VALUES, so it arrives as Anonymous, not Values.
        pytest.param("VALUES (1, 2)", "cannot introspect Anonymous", id="values"),
        pytest.param("TABLE t", "cannot introspect Alias", id="table"),
        pytest.param("", "single statement.*got 0", id="empty"),
        pytest.param("-- just a comment", "single statement.*got 0", id="comment-only"),
    ],
)
def test_get_schema_using_query_refuses_statements_with_no_result_shape(
    query: str, match: str
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

    with pytest.raises(exc.XorqError, match=match):
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
    """``ops.SQLQueryResult`` keeps the whole string, and compiling it takes
    ``parse_one``'s first statement and drops the rest, so a multi-statement
    ``con.sql`` would silently run less than it said. Refused instead."""
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
    # These are not, and must not come back as Array(Unknown) or Array(Point).
    # ``xid[]`` is not among them: it maps to a top-level ``Unknown``, which
    # the top-level check catches without the recursion this test is for.
    for spelling in ("xml[]", "bit[]", "point[]"):
        with pytest.raises(exc.UnsupportedBackendType):
            mapper.from_string(spelling, nullable=True)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dt.Array(dt.Array(dt.unknown)), id="nested-array"),
        pytest.param(dt.Map(dt.string, dt.unknown), id="map-value"),
        pytest.param(dt.Map(dt.unknown, dt.string), id="map-key"),
        pytest.param(dt.Struct({"a": dt.int32, "b": dt.unknown}), id="struct"),
        pytest.param(dt.Array(dt.point), id="geo-element"),
    ],
)
def test_the_guard_finds_an_unmappable_part_anywhere_in_a_type(
    dtype: dt.DataType,
) -> None:
    """No spelling the mapper parses produces these today; the refusal on a
    bound schema walks them all the same, so the walk is pinned directly."""
    part = RedshiftBackend.compiler.type_mapper.unmappable_part(dtype)

    assert isinstance(part, (dt.Unknown, dt.GeoSpatial))


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


def test_temporary_replace_drops_only_the_sessions_temp_table() -> None:
    """An unqualified ``DROP`` resolves through ``search_path``, so the one
    ``replace`` emits for a permanent table would, for a temporary ingest with
    no temporary table of that name yet, land on a PERMANENT table and destroy
    it. A temporary ``replace`` therefore looks the name up in the session's
    temporary schemas and drops only what it finds there, qualified by that
    schema -- the form measured live to remove the temporary table and leave a
    same-named permanent one intact."""
    con = make_introspection_con(temp_schemas=("pg_temp_5",))

    con.read_record_batches(
        make_reader({"a": [1], "b": ["x"]}),
        table_name="t",
        temporary=True,
        mode="replace",
    )

    lookup, drop, create = issued(con)
    assert "svv_columns" in lookup
    assert drop == 'DROP TABLE "pg_temp_5"."t"'
    assert create == 'CREATE TEMPORARY TABLE "t" ("a" BIGINT, "b" VARCHAR(65535))'
    ((_bound, values),) = con.con.bound
    assert values == [b"t"]


def test_temporary_replace_with_no_temp_table_drops_nothing() -> None:
    """The case the old refusal existed for: no temporary table of this name in
    the session, so an unqualified ``DROP`` would have reached a permanent one.
    With nothing to replace, no ``DROP`` is sent at all."""
    con = make_introspection_con(temp_schemas=())

    con.read_record_batches(
        make_reader({"a": [1], "b": ["x"]}),
        table_name="t",
        temporary=True,
        mode="replace",
    )

    lookup, create = issued(con)
    assert "svv_columns" in lookup
    assert create == 'CREATE TEMPORARY TABLE "t" ("a" BIGINT, "b" VARCHAR(65535))'


def test_a_deferred_temporary_read_executes_twice(tmp_path: Path) -> None:
    """``deferred_read_parquet`` defaults ``mode="replace"`` for this backend,
    and every execution of the expression re-runs the read under the same
    generated name. With ``temporary=True`` that used to be refused outright,
    so the standard temporary-read API failed on Redshift alone. The first run
    finds no temporary table and drops nothing; the second drops the one the
    first run created, and only that one."""
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"a": [1], "b": ["x"]}), path)
    con = make_introspection_con(
        temp_rows=(
            ("a", "bigint", "YES", 64, 0),
            ("b", "character varying", "YES", None, None),
        ),
    )
    del con.table
    read = xo.deferred_read_parquet(path, con, temporary=True).op()
    name = read.name

    read.make_dt()
    assert not any(sql.startswith("DROP") for sql in issued(con))

    con.con.log.clear()
    con.con.temp_schemas.append("pg_temp_5")
    dt_ = read.make_dt()

    assert dt_.name == name
    assert f'DROP TABLE "pg_temp_5"."{name}"' in issued(con)
    assert not any(
        sql.startswith("DROP") and "pg_temp_5" not in sql for sql in issued(con)
    )


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


def test_a_query_with_an_unmappable_column_is_refused_where_it_is_read() -> None:
    """``con.sql`` binds such a column as the catalog path does, so the same
    refusal has to catch it."""
    con = make_introspection_con(
        description=(_FakeColumn("id", 23, "int4"), _FakeColumn("s", 4000))
    )
    t = con.sql("SELECT id, s FROM t")

    with pytest.raises(
        exc.UnmappableColumnError, match="'s' \\(redshift type 'super'\\)"
    ):
        con.to_pyarrow_batches(t)


def test_an_unmappable_column_round_trips_through_yaml_with_its_spelling() -> None:
    """A bound table is serialised into every build. The yaml loader resolves
    a dtype by its class name in the datatypes module, so a class defined
    anywhere else would serialise and then fail to load."""
    schema = RedshiftBackend._schema_from_catalog_rows(OFFERS_CATALOG_ROWS)
    context = TranslationContext()

    loaded = context.translate_from_yaml(context.translate_to_yaml(schema))

    assert loaded == schema
    assert loaded["payload"] == dt.NamedUnknown(raw_type="super", nullable=False)


def test_an_unmappable_column_is_named_unknown_and_renders_its_spelling() -> None:
    """``name`` is ``Unknown``'s on purpose: ``SqlglotType.from_ibis``
    dispatches on it, so a backend's ``_from_ibis_Unknown`` reaches this class
    too. The rendering is what a schema shows a user."""
    dtype = dt.NamedUnknown(raw_type="super", nullable=False)

    assert dtype.name == dt.unknown.name == "Unknown"
    assert str(dtype) == "!unknown('super')"


def test_an_unmappable_column_refuses_arrow_conversion_on_its_own() -> None:
    """Behind the explicit refusal, not instead of it: plain ``dt.Unknown``
    converts to Arrow ``string``, which is how a SUPER read would come back
    mistyped. ``NamedUnknown`` has no Arrow mapping at all."""
    schema = RedshiftBackend._schema_from_catalog_rows(OFFERS_CATALOG_ROWS)

    with pytest.raises(NotImplementedError, match="super"):
        schema.to_pyarrow()
