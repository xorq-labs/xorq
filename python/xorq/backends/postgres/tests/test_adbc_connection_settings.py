"""The ADBC read connection must be configured like the psycopg one.

It is a second connection, opened per read from ``PgADBC``'s URI, so anything
the caller or ``_post_connect`` set on psycopg's connection and the URI omits
silently diverges there. These tests open both against the postgres job's
server and compare what each connection actually runs with.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pytest

from xorq.backends.postgres import Backend as PostgresBackend
from xorq.backends.redshift import Backend as RedshiftBackend
from xorq.common.utils.postgres_utils import (
    PgADBC,
    make_connection_defaults,
    make_credential_defaults,
)


SCHEMA = "xorq_adbc_settings_probe"


def connect(
    backend_cls: type[PostgresBackend] = PostgresBackend, **kwargs: Any
) -> PostgresBackend:
    return backend_cls().connect(
        **make_credential_defaults(), **make_connection_defaults(), **kwargs
    )


@pytest.fixture
def schema_with_a_table() -> Iterator[str]:
    admin = connect()
    admin.raw_sql(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')
    admin.raw_sql(f'CREATE TABLE "{SCHEMA}"."t" (a bigint, b text)')
    admin.raw_sql(f"INSERT INTO \"{SCHEMA}\".\"t\" VALUES (1, 'x'), (2, 'y')")
    try:
        yield SCHEMA
    finally:
        admin.raw_sql(f'DROP SCHEMA "{SCHEMA}" CASCADE')
        admin.disconnect()


def adbc_setting(con: PostgresBackend, name: str) -> str:
    with PgADBC(con).get_conn() as adbc, adbc.cursor() as cursor:
        cursor.execute(f"SELECT current_setting('{name}')")
        (value,) = cursor.fetchone()
    return value


def test_caller_libpq_settings_reach_the_adbc_connection() -> None:
    """A caller's ``application_name`` and ``options`` are libpq settings
    psycopg applied; the ADBC connection must run with the same."""
    con = connect(
        application_name="xorq-adbc-probe", options="-c statement_timeout=12345"
    )
    try:
        assert adbc_setting(con, "application_name") == "xorq-adbc-probe"
        assert adbc_setting(con, "statement_timeout") == "12345ms"
    finally:
        con.disconnect()


def test_schema_and_caller_options_are_combined(schema_with_a_table: str) -> None:
    """``schema`` arrives as a ``search_path`` in ``options``, beside the
    caller's own ``options`` rather than replacing them."""
    con = connect(schema=schema_with_a_table, options="-c statement_timeout=12345")
    try:
        assert adbc_setting(con, "search_path") == schema_with_a_table
        assert adbc_setting(con, "statement_timeout") == "12345ms"
    finally:
        con.disconnect()


def test_table_bound_read_on_a_schema_connection_is_served_by_adbc(
    schema_with_a_table: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the schema reached the URI, this read raised "relation does not
    exist" on ADBC and was re-run on psycopg by the execute-stage catch. The
    psycopg fallback opens a named cursor, so counting them counts fallbacks;
    the rows must also match what the psycopg path returns."""
    con = connect(schema=schema_with_a_table)
    try:
        expr = con.sql(
            "SELECT a, b FROM t ORDER BY a", schema={"a": "int64", "b": "string"}
        )

        named_cursors = []
        cursor = con.con.cursor

        def counting_cursor(*args, **kwargs):
            if kwargs.get("name"):
                named_cursors.append(kwargs["name"])
            return cursor(*args, **kwargs)

        monkeypatch.setattr(con.con, "cursor", counting_cursor)
        served_by_adbc = con.to_pyarrow_batches(expr).read_all()
        assert named_cursors == []

        monkeypatch.setattr(con, "_open_adbc_conn_or_none", lambda: None)
        served_by_psycopg = con.to_pyarrow_batches(expr).read_all()
        assert len(named_cursors) == 1

        assert served_by_adbc.equals(served_by_psycopg)
        assert served_by_adbc.to_pydict() == {"a": [1, 2], "b": ["x", "y"]}
        assert served_by_adbc.schema == pa.schema(
            [("a", pa.int64()), ("b", pa.string())]
        )
    finally:
        con.disconnect()


@pytest.mark.parametrize(
    "backend_cls",
    [
        pytest.param(PostgresBackend, id="postgres"),
        pytest.param(RedshiftBackend, id="redshift"),
    ],
)
@pytest.mark.parametrize(
    "caller_kwargs",
    [
        pytest.param({}, id="defaults"),
        pytest.param({"schema": "public"}, id="schema"),
        pytest.param({"client_encoding": "latin1"}, id="caller-encoding"),
        pytest.param({"options": "-c statement_timeout=5000"}, id="caller-options"),
    ],
)
def test_clone_hashes_equal_with_a_real_libpq(
    backend_cls: type[PostgresBackend], caller_kwargs: dict
) -> None:
    """The offline clone tests fake ``get_parameters``; this one does not. The
    libpq psycopg bundles reports settings nobody passed (``sslcertmode`` from
    libpq 17), and a clone that carried them hashed differently from its
    source and handed them to the ADBC URI. Redshift's class runs against the
    postgres server here: connecting and cloning are client-side."""
    source = connect(backend_cls, **caller_kwargs)
    try:
        clone = source.clone()
        try:
            assert clone._profile.content_hash == source._profile.content_hash
            assert PgADBC(clone).settings == PgADBC(source).settings
        finally:
            clone.disconnect()
    finally:
        source.disconnect()


def test_a_from_connection_clone_dials_the_port_it_came_from() -> None:
    """libpq omits a default port from ``get_parameters``, so a clone of a
    ``from_connection`` backend used to fall back to the subclass's
    ``do_connect`` default: Redshift's 5439 for a connection on 5432."""
    raw = connect().con
    source = RedshiftBackend.from_connection(raw)
    try:
        clone = source.clone(password=make_credential_defaults()["password"])
        try:
            assert clone.con.info.port == raw.info.port
        finally:
            clone.disconnect()
    finally:
        source.disconnect()
