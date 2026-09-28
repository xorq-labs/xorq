from __future__ import annotations

import pytest


pytest.importorskip("adbc_driver_postgresql")
psycopg = pytest.importorskip("psycopg")

from xorq.backends.postgres import Backend as PostgresBackend  # noqa: E402
from xorq.common.utils.postgres_utils import PgADBC  # noqa: E402


class _FakeLibpqInfo:
    host = "example.invalid"
    port = 5432
    dbname = "d"

    def __init__(self, user: str) -> None:
        self.user = user


class _FakeConnection:
    def __init__(self, user: str) -> None:
        self.info = _FakeLibpqInfo(user)


def make_offline_con(user: str, password: str) -> PostgresBackend:
    con = PostgresBackend()
    type(con).__init__(con, host="example.invalid", user=user, password=password)
    con.con = _FakeConnection(user)
    return con


@pytest.mark.parametrize(
    ("user", "password"),
    [
        pytest.param("u", "static", id="plain"),
        # Redshift IAM database users are named like this, colon included.
        pytest.param("IAMR:MyRole", "p@ss/w0rd", id="iam-user"),
        pytest.param("u", "a:b@c/d#e%f?g", id="every-reserved"),
        pytest.param("u@x", "100%", id="at-in-user"),
    ],
)
def test_get_uri_round_trips_userinfo_through_libpq(user: str, password: str) -> None:
    """libpq splits userinfo on the first ``:`` and the last ``@``, so raw
    interpolation shifts the components silently: ``IAMR:MyRole`` with
    ``p@ss/w0rd`` parsed as user ``IAMR``, and ``%`` is a parse error. The
    parse is libpq's own, through psycopg, because that is what the ADBC
    driver hands the URI to."""
    uri = PgADBC(make_offline_con(user, password)).get_uri()

    parsed = psycopg.conninfo.conninfo_to_dict(uri)

    assert (parsed["user"], parsed["password"]) == (user, password)
    assert (parsed["host"], parsed["port"], parsed["dbname"]) == (
        "example.invalid",
        "5432",
        "d",
    )
