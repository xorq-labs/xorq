"""Lifecycle regression tests for ``TransformScope``.

Every resource materialized while transforming an expr -- upstream readers,
StreamCaches, registered placeholder tables -- is owned by a TransformScope
so cleanup survives planning failures, cleanup-chain aborts and abandoned
result readers.
"""

import functools
import gc

import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo
import xorq.expr.relations as relations
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.expr.api import get_plans, to_pyarrow_batches
from xorq.expr.relations import (
    TransformScope,
    prepare_create_table_from_expr,
    register_and_transform_remote_tables,
)
from xorq.tests.util import assert_frame_equal


pytest.importorskip("duckdb")


@pytest.fixture
def recording_caches(monkeypatch):
    instances = []

    class RecordingStreamCache(relations.StreamCache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.close_count = 0
            instances.append(self)

        def close(self):
            self.close_count += 1
            super().close()

    monkeypatch.setattr(relations, "StreamCache", RecordingStreamCache)
    return instances


def make_remote_expr(target):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    rt = xo.memtable(df).into_backend(target, "rt_tbl")
    return rt.filter(rt.a > 1)


def assert_all_closed_once(caches):
    assert caches and all(cache.close_count == 1 for cache in caches)


def test_scope_close_is_idempotent():
    events = []
    scope = TransformScope()
    scope.adopt("one", functools.partial(events.append, "one"))
    scope.close()
    scope.close()
    assert scope.closed
    assert events == ["one"]


def test_scope_close_is_lifo():
    events = []
    scope = TransformScope()
    for label in ("first", "second", "third"):
        scope.adopt(label, functools.partial(events.append, label))
    scope.close()
    assert events == ["third", "second", "first"]


def test_scope_failing_cleanup_does_not_skip_others():
    events = []
    scope = TransformScope()

    def boom():
        events.append("boom")
        raise RuntimeError("cleanup failed")

    scope.adopt("inner", functools.partial(events.append, "inner"))
    scope.adopt("boom", boom)
    scope.adopt("outer", functools.partial(events.append, "outer"))
    scope.close()
    assert events == ["outer", "boom", "inner"]


def test_scope_context_manager_closes():
    events = []
    with TransformScope() as scope:
        scope.adopt("one", functools.partial(events.append, "one"))
    assert scope.closed
    assert events == ["one"]


def test_scope_close_can_leave_tables():
    events = []
    scope = TransformScope()
    scope.adopt("cache", functools.partial(events.append, "cache"))
    scope.adopt("tbl", functools.partial(events.append, "drop"), "table")
    scope.close(drop_tables=False)
    assert scope.closed
    assert events == ["cache"]


def test_drop_placeholder_handles_views_and_missing_names():
    con = xo.duckdb.connect()
    table = pa.table({"a": [1]})
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    # duckdb registers record batch readers as a VIEW
    con.read_record_batches(reader, table_name="ph")
    assert "ph" in con.list_tables()
    relations.drop_placeholder(con, "ph")
    assert "ph" not in con.list_tables()
    # missing name: must not raise
    relations.drop_placeholder(con, "ph")


def test_planning_failure_releases_resources(recording_caches, monkeypatch):
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)

    def boom(*args, **kwargs):
        raise RuntimeError("planning failed")

    monkeypatch.setattr(type(target), "to_pyarrow_batches", boom)
    with pytest.raises(RuntimeError, match="planning failed"):
        to_pyarrow_batches(expr)
    assert_all_closed_once(recording_caches)
    assert target.list_tables() == []


def test_natural_planning_failure_releases_resources(recording_caches):
    target = xo.duckdb.connect()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    rt = xo.memtable(df).into_backend(target, "rt_tbl")
    expr = rt.mutate(c=rt.b.cast("int64"))
    with pytest.raises(Exception, match="(?i)convert"):
        expr.execute()
    assert_all_closed_once(recording_caches)
    assert target.list_tables() == []


def test_deferred_read_failure_releases_resources(recording_caches):
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


def test_failing_drop_does_not_skip_cache_close(recording_caches, monkeypatch):
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)
    reader = to_pyarrow_batches(expr)

    def boom(self, name, *args, **kwargs):
        raise RuntimeError("drop failed")

    monkeypatch.setattr(type(target), "drop_table", boom)
    monkeypatch.setattr(type(target), "drop_view", boom)
    reader.read_all()
    assert_all_closed_once(recording_caches)


def test_get_plans_drops_placeholders(recording_caches):
    con = xo.connect()
    df = pd.DataFrame({"a": [1, 2, 3]})
    expr = xo.memtable(df).into_backend(con, "rt_tbl")
    plans = get_plans(expr)
    assert plans
    assert_all_closed_once(recording_caches)
    assert con.list_tables() == []


def test_table_sql_failure_does_not_leak(recording_caches):
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)
    with pytest.raises(Exception, match="rt_tbl does not exist"):
        expr.sql("SELECT * FROM rt_tbl")
    assert_all_closed_once(recording_caches)
    assert target.list_tables() == []


def test_prepare_create_table_keeps_placeholders(recording_caches):
    # the sole caller (snowflake create_table) runs CTAS against the
    # placeholder after prepare returns: tables must stay registered while
    # the caches are closed
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)
    table = prepare_create_table_from_expr(target, expr)
    assert table is not None
    assert_all_closed_once(recording_caches)
    assert target.list_tables() != []


def test_stream_full_drain_closes_once(recording_caches):
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


def test_stream_abandoned_reader_gc_closes(recording_caches):
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


def test_stream_explicit_close_then_release(recording_caches):
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


def test_stream_early_termination_then_close(recording_caches):
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
    [xo.duckdb.connect, xo.connect],
    ids=["duckdb", "xorq"],
)
def test_into_backend_executes_correctly(connect):
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
    [xo.duckdb.connect, xo.connect],
    ids=["duckdb", "xorq"],
)
def test_into_backend_fanout_executes_correctly(connect):
    target = connect()
    df = pd.DataFrame({"id": [1, 2], "v": [10, 20]})
    rt = xo.memtable(df).into_backend(target, "rt_tbl")
    expr = rt.join(rt, "id")
    result = expr.execute()
    assert len(result) == 2
    assert target.list_tables() == []


def test_bind_closes_at_drain_not_gc():
    # the generator finally in bind() must fire AT exhaustion, not at GC:
    # drain the bound reader directly while still holding a strong reference
    # to it (the public path can mask the distinction -- the otel wrapper
    # drops its reference at drain, letting the finalize backstop fire at the
    # same moment under refcounting)
    closed = []
    scope = TransformScope()
    scope.adopt("marker", functools.partial(closed.append, True))
    schema = pa.schema([("a", pa.int64())])
    inner = pa.RecordBatchReader.from_batches(
        schema, iter([pa.RecordBatch.from_pydict({"a": [1, 2]})])
    )
    bound = scope.bind(inner)
    bound.read_all()
    assert closed == [True], "bind must close at exhaustion, not at reader GC"
    del bound


def test_scope_close_closes_adopted_readers():
    class StubReader:
        closed = False

        def close(self):
            self.closed = True

    scope = TransformScope()
    stub = scope.adopt_reader(StubReader())
    scope.close()
    assert stub.closed


def test_replacer_adopts_reader_cache_and_table():
    # ownership contract: one (reader, cache, table) triple per RemoteTable,
    # adopted in acquisition order so close() runs table -> cache -> reader
    target = xo.duckdb.connect()
    expr = make_remote_expr(target)
    _, scope = register_and_transform_remote_tables(expr.op().to_expr())
    try:
        labels = [(kind, label) for kind, label, _ in scope._entries]
        assert labels == [
            ("resource", "reader"),
            ("resource", "cache"),
            ("table", next(iter(scope.created))),
        ]
    finally:
        scope.close()
    assert target.list_tables() == []


def test_partial_read_record_batches_failure_drops_placeholder(
    recording_caches, monkeypatch
):
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
