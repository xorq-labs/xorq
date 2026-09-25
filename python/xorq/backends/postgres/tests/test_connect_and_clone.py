"""Regression tests for the postgres module-level ``connect`` and for ``clone``.

Sited here rather than in a top-level test module so that
``backends/conftest.py``'s ``pytest_ignore_collect`` drops the file in jobs that
did not select the ``postgres`` marker. Importing ``xorq.backends.postgres``
pulls in ``adbc_driver_manager`` (``backends/postgres/__init__.py``) and
``psycopg`` (``vendor/ibis/backends/postgres/__init__.py``), neither of which is
a core dependency; path-gated collection is what lets this file import them at
module scope with no ``importorskip`` guard.

No live server: ``psycopg.connect`` is faked, so these exercise the wiring the
two defects broke rather than a round trip.
"""

from typing import Any

import psycopg
import pytest

import xorq.backends.postgres as postgres_module


class FakeConnectionInfo:
    def __init__(self, parameters: dict[str, str]) -> None:
        self.parameters = parameters

    def get_parameters(self) -> dict[str, str]:
        return dict(self.parameters)


class FakeConnection:
    def __init__(
        self, parameters: dict[str, str] | None = None, autocommit: bool = True
    ) -> None:
        self.info = FakeConnectionInfo(parameters or {})
        self.autocommit = autocommit


class SubclassBackend(postgres_module.Backend):
    """Stands in for any backend built on postgres, Redshift included."""


# what ``info.get_parameters`` can report: libpq conninfo keywords. psycopg-only
# settings such as ``autocommit`` are absent from it by construction.
LIBPQ_KEYWORDS = ("host", "port", "user", "dbname", "options", "sslmode", "passfile")


def fake_psycopg_connect(**kwargs: Any) -> FakeConnection:
    return FakeConnection(
        {
            key: str(value)
            for key, value in kwargs.items()
            if key in LIBPQ_KEYWORDS and value is not None
        },
        autocommit=kwargs.get("autocommit", True),
    )


@pytest.fixture
def faked_psycopg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(psycopg, "connect", fake_psycopg_connect)
    # touches ``self.con.cursor()`` and pandas adapters, neither of which the
    # fake connection has; the defects under test are upstream of it
    monkeypatch.setattr(postgres_module.Backend, "_post_connect", lambda self: None)


def test_module_level_connect_builds_a_connected_backend(faked_psycopg: None) -> None:
    """``Backend.connect(**kwargs)`` was an unbound call against
    ``BaseBackend.connect(self, *args, **kwargs)``: it raised ``TypeError``
    before the ``return con`` below it could run."""
    con = postgres_module.connect(host="example.invalid", user="u", database="d")

    assert isinstance(con, postgres_module.Backend)
    assert con.con is not None


def test_clone_returns_the_subclass_not_postgres(faked_psycopg: None) -> None:
    """``clone`` called the module-level ``connect``, which hard-codes
    ``Backend``. ``self.connect`` goes through ``BaseBackend.connect``, which
    instantiates ``self.__class__``, so a subclass clones into its own class."""
    con = SubclassBackend()
    con.con = FakeConnection(
        {
            "host": "example.invalid",
            "port": "5432",
            "user": "u",
            "dbname": "d",
            "options": "-c search_path=public",
        }
    )

    cloned = con.clone(password="pw")

    assert type(cloned) is SubclassBackend
    assert cloned is not con
    assert cloned.con is not None
    # ``dbname`` is renamed on the way through
    assert cloned._con_kwargs["database"] == "d"
    assert "dbname" not in cloned._con_kwargs


def test_clone_keeps_settings_the_dsn_cannot_report(faked_psycopg: None) -> None:
    """``clone`` rebuilt its kwargs from ``con.info.get_parameters()`` alone.
    That drops ``options`` (dissoc'd outright) and anything libpq never sees,
    so a clone silently connected with a different ``search_path`` and
    different transaction semantics than the connection it was cloned from."""
    con = postgres_module.connect(
        host="example.invalid",
        user="u",
        database="d",
        password="pw",
        options="-c search_path=analytics",
        autocommit=False,
        schema="analytics",
    )

    cloned = con.clone(password="pw")

    # a libpq runtime setting: reportable, but was dissoc'd away
    assert cloned._con_kwargs["options"] == "-c search_path=analytics"
    # a psycopg ``Connection`` setting and a ``_post_connect`` one: never libpq
    # conninfo keywords, so ``get_parameters`` cannot report them at all
    assert cloned._con_kwargs["autocommit"] is False
    assert cloned._con_kwargs["schema"] == "analytics"


def test_clone_takes_options_the_caller_never_passed(faked_psycopg: None) -> None:
    """``options`` reaching libpq from the environment (``PGOPTIONS``) is
    reported by the DSN and by nothing else, so the DSN is what must supply
    it."""
    con = SubclassBackend()
    con.con = FakeConnection({"dbname": "d", "options": "-c search_path=fromenv"})

    cloned = con.clone(password="pw")

    assert cloned._con_kwargs["options"] == "-c search_path=fromenv"


def test_clone_keeps_a_secret_as_the_env_reference_it_was_passed_as(
    faked_psycopg: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``options`` is a declared secret key. ``_con_kwargs`` and the DSN both
    hold it resolved, so taking either bakes the literal into the clone's
    profile -- which ``xo.build`` writes to disk with no secret check."""
    monkeypatch.setenv("PGOPTS", "-c search_path=analytics")
    monkeypatch.setenv("PGPW", "hunter2")
    con = postgres_module.connect(
        host="example.invalid",
        user="u",
        database="d",
        password="$PGPW",
        options="$PGOPTS",
    )
    # the source resolves it to connect, but its profile kept the reference
    assert con._con_kwargs["options"] == "-c search_path=analytics"
    assert con._profile.kwargs_dict["options"] == "$PGOPTS"
    con._profile.check_for_exposed_secrets()

    cloned = con.clone(password="$PGPW")

    assert cloned._profile.kwargs_dict["options"] == "$PGOPTS"
    # the check ``Profile.save`` runs and ``xo.build`` does not
    cloned._profile.check_for_exposed_secrets()


def test_clone_reads_autocommit_off_the_live_connection(faked_psycopg: None) -> None:
    """``from_connection`` leaves ``_con_kwargs`` empty, so a clone that learns
    ``autocommit`` only from there silently comes back autocommitting."""
    con = postgres_module.Backend.from_connection(
        FakeConnection({"dbname": "d"}, autocommit=False)
    )
    assert con._con_kwargs == {}

    cloned = con.clone(password="pw")

    assert cloned._con_kwargs["autocommit"] is False
