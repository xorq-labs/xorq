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

import psycopg
import pytest

import xorq.backends.postgres as postgres_module


class FakeConnectionInfo:
    def __init__(self, parameters):
        self.parameters = parameters

    def get_parameters(self):
        return dict(self.parameters)


class FakeConnection:
    def __init__(self, parameters=None):
        self.info = FakeConnectionInfo(parameters or {})


class SubclassBackend(postgres_module.Backend):
    """Stands in for any backend built on postgres, Redshift included."""


@pytest.fixture
def faked_psycopg(monkeypatch):
    monkeypatch.setattr(psycopg, "connect", lambda **kwargs: FakeConnection())
    # touches ``self.con.cursor()`` and pandas adapters, neither of which the
    # fake connection has; the defects under test are upstream of it
    monkeypatch.setattr(postgres_module.Backend, "_post_connect", lambda self: None)


def test_module_level_connect_builds_a_connected_backend(faked_psycopg):
    """``Backend.connect(**kwargs)`` was an unbound call against
    ``BaseBackend.connect(self, *args, **kwargs)``: it raised ``TypeError``
    before the ``return con`` below it could run."""
    con = postgres_module.connect(host="example.invalid", user="u", database="d")

    assert isinstance(con, postgres_module.Backend)
    assert con.con is not None


def test_clone_returns_the_subclass_not_postgres(faked_psycopg):
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
    # ``dbname`` is renamed and ``options`` dropped on the way through
    assert cloned._con_kwargs["database"] == "d"
    assert "dbname" not in cloned._con_kwargs
    assert "options" not in cloned._con_kwargs
