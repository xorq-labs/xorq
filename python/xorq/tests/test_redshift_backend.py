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

import pytest
import sqlglot as sg
import sqlglot.expressions as sge

import xorq
import xorq.api as xo
from xorq.backends.postgres import Backend as PostgresBackend
from xorq.backends.redshift import DEFAULT_PORT
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
