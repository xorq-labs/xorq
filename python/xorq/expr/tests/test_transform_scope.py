"""Lifecycle regression tests for ``RemoteTableScope``.

Every resource materialized while transforming an expr -- upstream readers,
StreamCaches, registered placeholder tables -- is owned by a RemoteTableScope
so cleanup survives planning failures, cleanup-chain aborts and abandoned
result readers.
"""

from __future__ import annotations

import gc
from collections.abc import Callable

import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.expr.remote_table_exec as remote_table_exec
import xorq.vendor.ibis.expr.types as ir
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.expr.api import get_plans, to_pyarrow_batches
from xorq.expr.remote_table_exec import (
    RemoteTableScope,
    bind_scope_to_reader,
    drop_placeholder,
    prepare_create_table_from_expr,
    register_and_transform_remote_tables,
)
from xorq.tests.util import assert_frame_equal
from xorq.vendor.ibis.backends import BaseBackend


pytest.importorskip("duckdb")


@pytest.fixture
def recording_caches(monkeypatch: pytest.MonkeyPatch) -> list:
    instances = []

    class RecordingStreamCache(remote_table_exec.StreamCache):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.close_count = 0
            instances.append(self)

        def close(self) -> None:
            self.close_count += 1
            super().close()

    monkeypatch.setattr(remote_table_exec, "StreamCache", RecordingStreamCache)
    return instances


def make_remote_expr(target: BaseBackend) -> ir.Table:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    rt = xo.memtable(df).into_backend(target, "rt_tbl")
    return rt.filter(rt.a > 1)


def assert_all_closed_once(caches: list) -> None:
    assert caches and all(cache.close_count == 1 for cache in caches)


class _FakeCloseable:
    def __init__(self, events: list[str], label: str) -> None:
        self._events = events
        self._label = label

    def close(self) -> None:
        self._events.append(self._label)


def test_scope_close_is_idempotent() -> None:
    events: list[str] = []
    scope = RemoteTableScope()
    scope.adopt_cache(_FakeCloseable(events, "one"))
    scope.close()
    scope.close()
    assert scope.closed
    assert events == ["one"]


def test_scope_close_is_lifo_within_category() -> None:
    events: list[str] = []
    scope = RemoteTableScope()
    scope.adopt_reader(_FakeCloseable(events, "r1"))
    scope.adopt_reader(_FakeCloseable(events, "r2"))
    scope.adopt_cache(_FakeCloseable(events, "c1"))
    scope.adopt_cache(_FakeCloseable(events, "c2"))
    scope.close()
    # caches reversed, then readers reversed (tables -> caches -> readers)
    assert events == ["c2", "c1", "r2", "r1"]


def test_scope_close_tables_before_caches_before_readers() -> None:
    target = xo.duckdb.connect()
    events: list[str] = []
    scope = RemoteTableScope()
    scope.adopt_reader(_FakeCloseable(events, "reader"))
    scope.adopt_cache(_FakeCloseable(events, "cache"))
    schema = pa.schema([("a", pa.int64())])
    reader = pa.RecordBatchReader.from_batches(
        schema, iter([pa.RecordBatch.from_pydict({"a": [1]})])
    )
    target.read_record_batches(reader, table_name="ph")
    scope.adopt_table(target, "ph")
    scope.close()
    assert events[0] == "cache"
    assert events[1] == "reader"
    assert "ph" not in target.list_tables()


def test_scope_failing_cleanup_does_not_skip_others() -> None:
    events: list[str] = []
    scope = RemoteTableScope()

    class BoomCache:
        def close(self) -> None:
            events.append("boom")
            raise RuntimeError("cleanup failed")

    scope.adopt_reader(_FakeCloseable(events, "reader"))
    scope.adopt_cache(BoomCache())
    scope.adopt_cache(_FakeCloseable(events, "good_cache"))
    scope.close()
    # good_cache (LIFO), boom, then reader
    assert events == ["good_cache", "boom", "reader"]


def test_scope_context_manager_closes() -> None:
    events: list[str] = []
    with RemoteTableScope() as scope:
        scope.adopt_cache(_FakeCloseable(events, "one"))
    assert scope.closed
    assert events == ["one"]


def test_scope_close_drops_all_entries() -> None:
    events: list[str] = []
    scope = RemoteTableScope()
    scope.adopt_cache(_FakeCloseable(events, "cache"))
    scope.adopt_reader(_FakeCloseable(events, "reader"))
    scope.close()
    assert scope.closed
    # caches before readers in teardown order
    assert events == ["cache", "reader"]


def test_drop_placeholder_handles_views_and_missing_names() -> None:
    con = xo.duckdb.connect()
    table = pa.table({"a": [1]})
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    # duckdb registers record batch readers as a VIEW
    con.read_record_batches(reader, table_name="ph")
    assert "ph" in con.list_tables()
    drop_placeholder(con, "ph")
    assert "ph" not in con.list_tables()
    # missing name: must not raise
    drop_placeholder(con, "ph")


def test_planning_failure_releases_resources(
    recording_caches: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)

    def boom(*args, **kwargs):
        raise RuntimeError("planning failed")

    monkeypatch.setattr(type(target), "to_pyarrow_batches", boom)
    with pytest.raises(RuntimeError, match="planning failed"):
        to_pyarrow_batches(expr)
    assert_all_closed_once(recording_caches)
    assert target.list_tables() == []


def test_natural_planning_failure_releases_resources(recording_caches: list) -> None:
    target = xo.duckdb.connect()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    rt = xo.memtable(df).into_backend(target, "rt_tbl")
    expr = rt.mutate(c=rt.b.cast("int64"))
    with pytest.raises(Exception, match="(?i)convert"):
        expr.execute()
    assert_all_closed_once(recording_caches)
    assert target.list_tables() == []


def test_deferred_read_failure_releases_resources(recording_caches: list) -> None:
    target = xo.duckdb.connect()
    df = pd.DataFrame({"a": [1, 2, 3]})
    rt = xo.memtable(df).into_backend(target, "rt_tbl")
    missing = deferred_read_parquet(
        "/nonexistent/missing.parquet",
        con=target,
        schema=xo.schema({"a": "int64"}),
    )
    expr = rt.union(missing)
    with pytest.raises(ValueError, match="At least one path is required"):
        expr.execute()
    assert_all_closed_once(recording_caches)
    assert target.list_tables() == []


def test_failing_drop_does_not_skip_cache_close(
    recording_caches: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)
    reader = to_pyarrow_batches(expr)

    def boom(self, name, *args, **kwargs):
        raise RuntimeError("drop failed")

    monkeypatch.setattr(type(target), "drop_table", boom)
    monkeypatch.setattr(type(target), "drop_view", boom)
    reader.read_all()
    assert_all_closed_once(recording_caches)


def test_get_plans_drops_placeholders(recording_caches: list) -> None:
    con = xo.connect()
    df = pd.DataFrame({"a": [1, 2, 3]})
    expr = xo.memtable(df).into_backend(con, "rt_tbl")
    plans = get_plans(expr)
    assert plans
    assert_all_closed_once(recording_caches)
    assert con.list_tables() == []


def test_table_sql_failure_does_not_leak(recording_caches: list) -> None:
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)
    with pytest.raises(Exception, match="rt_tbl does not exist"):
        expr.sql("SELECT * FROM rt_tbl")
    assert_all_closed_once(recording_caches)
    assert target.list_tables() == []


def test_prepare_create_table_returns_scope(recording_caches: list) -> None:
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)
    (table, scope) = prepare_create_table_from_expr(target, expr)
    assert table is not None
    assert not scope.closed
    assert target.list_tables() != []
    scope.close()
    assert_all_closed_once(recording_caches)
    assert target.list_tables() == []


def test_stream_full_drain_closes_once(recording_caches: list) -> None:
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)
    reader = to_pyarrow_batches(expr)
    table = reader.read_all()
    assert table.num_rows == 2
    (cache,) = recording_caches
    assert cache.close_count == 1
    assert target.list_tables() == []
    # releasing the drained reader is a no-op (idempotent close)
    del reader
    gc.collect()
    assert cache.close_count == 1


def test_stream_abandoned_reader_gc_closes(recording_caches: list) -> None:
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)
    reader = to_pyarrow_batches(expr)
    (cache,) = recording_caches
    assert cache.close_count == 0
    assert target.list_tables() != []
    # never read a single batch: the weakref.finalize backstop must fire
    del reader
    gc.collect()
    assert cache.close_count == 1
    assert target.list_tables() == []


def test_stream_explicit_close_then_release(recording_caches: list) -> None:
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)
    reader = to_pyarrow_batches(expr)
    # pyarrow does not finalize wrapped generators on close(); cleanup
    # fires when the last reference drops
    reader.close()
    del reader
    gc.collect()
    (cache,) = recording_caches
    assert cache.close_count == 1
    assert target.list_tables() == []


def test_stream_early_termination_then_close(recording_caches: list) -> None:
    target = xo.duckdb.connect()
    df = pd.DataFrame({"a": range(1000)})
    rt = xo.memtable(df).into_backend(target, "rt_tbl")
    reader = to_pyarrow_batches(rt.limit(1))
    batch = reader.read_next_batch()
    assert len(batch) == 1
    reader.close()
    del reader
    gc.collect()
    (cache,) = recording_caches
    assert cache.close_count == 1
    assert target.list_tables() == []


@pytest.mark.parametrize(
    "connect",
    [
        pytest.param(xo.duckdb.connect, id="duckdb"),
        pytest.param(xo.connect, id="xorq"),
    ],
)
def test_into_backend_executes_correctly(connect: Callable[[], BaseBackend]) -> None:
    target = connect()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    rt = xo.memtable(df).into_backend(target, "rt_tbl")
    expr = rt.filter(rt.a > 1).select("a", "b")
    result = expr.execute().reset_index(drop=True)
    expected = df[df.a > 1].reset_index(drop=True)
    assert_frame_equal(result, expected)
    assert target.list_tables() == []


@pytest.mark.parametrize(
    "connect",
    [
        pytest.param(xo.duckdb.connect, id="duckdb"),
        pytest.param(xo.connect, id="xorq"),
    ],
)
def test_into_backend_fanout_executes_correctly(
    connect: Callable[[], BaseBackend],
) -> None:
    target = connect()
    df = pd.DataFrame({"id": [1, 2], "v": [10, 20]})
    rt = xo.memtable(df).into_backend(target, "rt_tbl")
    expr = rt.join(rt, "id")
    result = expr.execute()
    assert len(result) == 2
    assert target.list_tables() == []


def test_bind_closes_at_drain_not_gc() -> None:
    closed: list[str] = []
    scope = RemoteTableScope()
    scope.adopt_cache(_FakeCloseable(closed, "drain"))
    schema = pa.schema([("a", pa.int64())])
    inner = pa.RecordBatchReader.from_batches(
        schema, iter([pa.RecordBatch.from_pydict({"a": [1, 2]})])
    )
    bound = bind_scope_to_reader(scope, inner)
    bound.read_all()
    assert closed == ["drain"], "bind must close at exhaustion, not at reader GC"
    del bound


def test_scope_close_closes_adopted_readers() -> None:
    class StubReader:
        closed = False

        def close(self) -> None:
            self.closed = True

    scope = RemoteTableScope()
    stub = scope.adopt_reader(StubReader())
    scope.close()
    assert stub.closed


def test_replacer_adopts_reader_cache_and_table() -> None:
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)
    _, scope = register_and_transform_remote_tables(expr.op().to_expr())
    try:
        assert scope.reader_count == 1
        assert scope.cache_count == 1
        assert scope.table_count == 1
        assert scope.table_names
    finally:
        scope.close()
    assert target.list_tables() == []


def test_partial_read_record_batches_failure_drops_placeholder(
    recording_caches: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    # register-then-raise inside read_record_batches: the pre-adopted
    # placeholder must still be dropped
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)

    def register_then_boom(self, source, table_name=None):
        self.con.register(table_name, source)
        raise RuntimeError("partial registration failure")

    monkeypatch.setattr(type(target), "read_record_batches", register_then_boom)
    with pytest.raises(RuntimeError, match="partial registration failure"):
        expr.execute()
    assert_all_closed_once(recording_caches)
    assert target.list_tables() == []
