"""Tests for LazyBackend."""

import threading

import pytest


pytest.importorskip("duckdb")
pytest.importorskip("sqlite3")

import xorq.backends.duckdb as duckdb
import xorq.backends.sqlite as sqlite
from xorq.backends._lazy import LazyBackend
from xorq.vendor.ibis.backends import BaseBackend


def make_duckdb_lazy(**kwargs):
    return LazyBackend(duckdb.Backend().connect, database=":memory:", **kwargs)


def make_sqlite_lazy(**kwargs):
    return LazyBackend(sqlite.Backend().connect, **kwargs)


@pytest.mark.duckdb
def test_starts_unconnected():
    lazy = make_duckdb_lazy()
    assert lazy.is_connected is False


@pytest.mark.duckdb
def test_first_attr_access_triggers_connect():
    lazy = make_duckdb_lazy()
    _ = lazy.name
    assert lazy.is_connected is True


@pytest.mark.duckdb
def test_connect_called_only_once():
    raw = duckdb.Backend()
    call_count = 0
    original = raw.connect

    def counting_connect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    counting_connect.__self__ = raw

    lazy = LazyBackend(counting_connect, database=":memory:")
    _ = lazy.name
    _ = lazy.dialect
    _ = lazy.compiler
    assert call_count == 1


@pytest.mark.duckdb
def test_repr_before_and_after_connect():
    lazy = make_duckdb_lazy()
    assert "not connected" in repr(lazy)
    assert "Backend" in repr(lazy)

    _ = lazy.name
    assert "connected" in repr(lazy)
    assert "not connected" not in repr(lazy)


@pytest.mark.duckdb
def test_isinstance_base_backend():
    lazy = make_duckdb_lazy()
    assert isinstance(lazy, BaseBackend)


@pytest.mark.duckdb
def test_isinstance_concrete_backend_class():
    raw = duckdb.Backend()
    lazy = LazyBackend(raw.connect, database=":memory:")
    assert isinstance(lazy, type(raw))


@pytest.mark.duckdb
def test_name_attribute_delegated():
    lazy = make_duckdb_lazy()
    assert lazy.name == "duckdb"


@pytest.mark.sqlite
def test_sqlite_name_attribute_delegated():
    lazy = make_sqlite_lazy()
    assert lazy.name == "sqlite"


@pytest.mark.duckdb
def test_setattr_forwarded_to_backend():
    lazy = make_duckdb_lazy()
    _ = lazy.name
    connected_backend = object.__getattribute__(lazy, "_backend")
    lazy._test_sentinel = "hello"
    assert connected_backend._test_sentinel == "hello"


@pytest.mark.duckdb
def test_thread_safety_connect_once():
    raw = duckdb.Backend()
    call_count = 0
    original = raw.connect
    lock = threading.Lock()

    def counting_connect(*args, **kwargs):
        nonlocal call_count
        with lock:
            call_count += 1
        return original(*args, **kwargs)

    counting_connect.__self__ = raw

    lazy = LazyBackend(counting_connect, database=":memory:")
    errors = []

    def access():
        try:
            _ = lazy.name
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=access) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert call_count == 1


@pytest.mark.duckdb
def test_lazy_duckdb_full_roundtrip(tmp_path):
    db_file = tmp_path / "test.duckdb"

    seed = duckdb.Backend()
    seed.do_connect(database=str(db_file))
    seed.con.execute(
        "CREATE TABLE cities AS SELECT * FROM "
        "(VALUES ('London', 9), ('Paris', 11)) AS t(city, temp)"
    )
    seed.disconnect()

    raw = duckdb.Backend()
    lazy = LazyBackend(raw.connect, database=str(db_file))
    assert lazy.is_connected is False

    tbl = lazy.table("cities")
    assert lazy.is_connected is True

    result = tbl.execute()
    assert set(result["city"]) == {"London", "Paris"}
