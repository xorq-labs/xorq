"""Tests for LazyBackend."""

import threading
from unittest.mock import patch

import pytest


pytest.importorskip("duckdb")
pytest.importorskip("sqlite3")

import xorq.backends.duckdb as duckdb
import xorq.backends.sqlite as sqlite
from xorq.backends.lazy import LazyBackend
from xorq.vendor.ibis.backends import BaseBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_duckdb_lazy(**kwargs):
    """Return an unconnected LazyBackend wrapping a DuckDB backend."""
    return LazyBackend(duckdb.Backend(), database=":memory:", **kwargs)


def make_sqlite_lazy(**kwargs):
    """Return an unconnected LazyBackend wrapping a SQLite backend."""
    return LazyBackend(sqlite.Backend(), **kwargs)


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


def test_starts_unconnected():
    lazy = make_duckdb_lazy()
    assert lazy.is_connected is False


def test_first_attr_access_triggers_connect():
    lazy = make_duckdb_lazy()
    _ = lazy.name
    assert lazy.is_connected is True


def test_connect_called_only_once():
    raw = duckdb.Backend()
    call_count = 0
    original = raw.do_connect

    def counting_do_connect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    lazy = LazyBackend(raw, database=":memory:")
    with patch.object(raw, "do_connect", side_effect=counting_do_connect):
        _ = lazy.name
        _ = lazy.dialect
        _ = lazy.compiler
    assert call_count == 1


def test_repr_before_and_after_connect():
    lazy = make_duckdb_lazy()
    assert "not connected" in repr(lazy)
    assert "Backend" in repr(lazy)

    _ = lazy.name
    assert "connected" in repr(lazy)
    assert "not connected" not in repr(lazy)


# ---------------------------------------------------------------------------
# isinstance / __class__ proxy
# ---------------------------------------------------------------------------


def test_isinstance_base_backend():
    lazy = make_duckdb_lazy()
    assert isinstance(lazy, BaseBackend)


def test_isinstance_concrete_backend_class():
    raw = duckdb.Backend()
    lazy = LazyBackend(raw, database=":memory:")
    assert isinstance(lazy, type(raw))


# ---------------------------------------------------------------------------
# Attribute delegation
# ---------------------------------------------------------------------------


def test_name_attribute_delegated():
    lazy = make_duckdb_lazy()
    assert lazy.name == "duckdb"


def test_sqlite_name_attribute_delegated():
    lazy = make_sqlite_lazy()
    assert lazy.name == "sqlite"


def test_setattr_forwarded_to_backend():
    raw = duckdb.Backend()
    lazy = LazyBackend(raw, database=":memory:")
    # Trigger connection so backend is fully set up.
    _ = lazy.name
    # Setattr on the proxy should land on the wrapped backend.
    lazy._test_sentinel = "hello"
    assert raw._test_sentinel == "hello"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_thread_safety_connect_once():
    """Concurrent first accesses must trigger do_connect exactly once."""
    raw = duckdb.Backend()
    call_count = 0
    original = raw.do_connect
    lock = threading.Lock()

    def counting_do_connect(*args, **kwargs):
        nonlocal call_count
        with lock:
            call_count += 1
        return original(*args, **kwargs)

    lazy = LazyBackend(raw, database=":memory:")
    errors = []

    def access():
        try:
            _ = lazy.name
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=access) for _ in range(20)]
    with patch.object(raw, "do_connect", side_effect=counting_do_connect):
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors
    assert call_count == 1


# ---------------------------------------------------------------------------
# End-to-end: use as DatabaseTable source
# ---------------------------------------------------------------------------


def test_duckdb_lazy_as_table_source(tmp_path):
    """LazyBackend as source in ops.DatabaseTable — connection deferred until execute."""

    from xorq.vendor.ibis import Schema
    from xorq.vendor.ibis.expr import operations as ops

    db_file = tmp_path / "nums.duckdb"

    # Seed the file with a plain connected backend.
    seed = duckdb.Backend()
    seed.do_connect(database=str(db_file))
    seed.con.execute("CREATE TABLE nums AS SELECT 1 AS a, 'x' AS b")
    seed.disconnect()

    # Build a LazyBackend pointing at the file — not connected yet.
    lazy = LazyBackend(duckdb.Backend(), database=str(db_file))
    assert lazy.is_connected is False

    # Simulate what load_expr does: create a DatabaseTable with source=lazy.
    schema = Schema({"a": "int64", "b": "string"})
    table_op = ops.DatabaseTable(
        name="nums",
        schema=schema,
        source=lazy,
        namespace=ops.Namespace(),
    )
    expr = table_op.to_expr()

    # Connection is established only here, when the query actually runs.
    result = expr.execute()
    assert lazy.is_connected is True
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 1


def test_lazy_duckdb_full_roundtrip(tmp_path):
    """Full roundtrip: seed a file, then query it via a LazyBackend."""

    db_file = tmp_path / "test.duckdb"

    # Seed
    seed = duckdb.Backend()
    seed.do_connect(database=str(db_file))
    seed.con.execute(
        "CREATE TABLE cities AS SELECT * FROM "
        "(VALUES ('London', 9), ('Paris', 11)) AS t(city, temp)"
    )
    seed.disconnect()

    # Query via LazyBackend — not connected until first access
    raw = duckdb.Backend()
    lazy = LazyBackend(raw, database=str(db_file))
    assert lazy.is_connected is False

    tbl = lazy.table("cities")
    assert lazy.is_connected is True  # table() triggers get_schema → connection

    result = tbl.execute()
    assert set(result["city"]) == {"London", "Paris"}
