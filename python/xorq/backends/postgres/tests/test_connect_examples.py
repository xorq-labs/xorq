import psycopg
import pytest

import xorq.api as xo
from xorq.backends.postgres import Backend
from xorq.common.utils.postgres_utils import postgres_config


FIELDS = ("host", "port", "user", "password", "database")


@pytest.fixture
def examples_env_to_local(monkeypatch):
    for field in FIELDS:
        monkeypatch.setenv(
            f"XORQ_EXAMPLES_POSTGRES_{field.upper()}",
            str(postgres_config[f"POSTGRES_{field.upper()}"]),
        )


def test_connect_examples_follows_env_overrides(examples_env_to_local):
    con = xo.postgres.connect_examples()
    with con.raw_sql("SELECT current_database()") as cursor:
        assert cursor.fetchone() == (postgres_config["POSTGRES_DATABASE"],)


def test_connect_examples_follows_port_override(examples_env_to_local, monkeypatch):
    monkeypatch.setenv("XORQ_EXAMPLES_POSTGRES_PORT", "1")
    with pytest.raises(psycopg.OperationalError, match="port 1"):
        xo.postgres.connect_examples()


def test_empty_override_keeps_the_public_value(monkeypatch):
    monkeypatch.setattr(Backend, "connect", lambda self, **kwargs: kwargs)
    public = xo.postgres.connect_examples()
    monkeypatch.setenv("XORQ_EXAMPLES_POSTGRES_HOST", "")
    monkeypatch.setenv("XORQ_EXAMPLES_POSTGRES_PASSWORD", "")
    assert xo.postgres.connect_examples() == public
