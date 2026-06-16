from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.caching.strategy import SnapshotStrategy
from xorq.common.utils.node_utils import compute_expr_hash
from xorq.sinking import BackendSink, ParquetSink, ThreadedBackendSink


if TYPE_CHECKING:
    from xorq.vendor.ibis.expr.types import Table


TABLE = {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]}


@pytest.fixture
def t() -> Table:
    con = xo.connect()
    return con.register(xo.memtable(TABLE), table_name="t0")


def _connect(name: str) -> Any:
    factory = {
        "datafusion": xo.connect,
        "duckdb": xo.duckdb.connect,
        "pandas": xo.pandas.connect,
    }[name]
    try:
        return factory()
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"backend {name} unavailable: {exc}")


# duckdb is excluded: its single connection is not re-entrant, so the streaming
# tee deadlocks when it pulls the parent reader while the same connection serves
# the outer query. The Phase 1 streaming tee targets engines that allow
# concurrent reader pulls (datafusion). See ADR-0014.
@pytest.fixture(params=["datafusion", "pandas"])
def backend_table(request: pytest.FixtureRequest) -> Table:
    con = _connect(request.param)
    return con.create_table("sink_src", pa.table(TABLE))


def test_sink_is_passthrough(t: Table, tmp_path: Path) -> None:
    expr = t.tee(ParquetSink(path=tmp_path / "out.parquet"))
    assert expr.schema() == t.schema()
    assert expr.execute().equals(t.execute())


def test_sink_writes_what_flows(t: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    t.tee(ParquetSink(path=target)).execute()
    written = pq.read_table(str(target))
    assert len(written) == len(t.execute())


def test_sink_hash_is_transparent(t: Table, tmp_path: Path) -> None:
    strategy = SnapshotStrategy()
    sinked = t.tee(ParquetSink(path=tmp_path / "out.parquet"))
    assert compute_expr_hash(sinked, strategy=strategy) == compute_expr_hash(
        t, strategy=strategy
    )


def test_cache_hit_does_not_write(t: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    cached = t.tee(ParquetSink(path=target)).cache()
    cached.execute()  # miss: writes
    mtime = target.stat().st_mtime_ns
    cached.execute()  # hit: must not write again
    assert target.stat().st_mtime_ns == mtime


def test_append_accumulates_rows(t: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    t.tee(ParquetSink(path=target, mode="append")).execute()
    assert len(pq.read_table(str(target))) == 4
    t.tee(ParquetSink(path=target, mode="append")).execute()
    assert len(pq.read_table(str(target))) == 8


def test_create_fails_if_target_exists(t: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    t.tee(ParquetSink(path=target, mode="create")).execute()
    assert target.exists()
    # raised mid-pull, the engine wraps FileExistsError (Arrow C-stream boundary)
    with pytest.raises(Exception, match="already exists"):
        t.tee(ParquetSink(path=target, mode="create")).execute()
    # the failed run published nothing: original file intact, no stray temp
    assert len(pq.read_table(str(target))) == 4
    assert list(tmp_path.glob("*.tmp")) == []


def test_create_sink_execute_raises_fileexists(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    sink = ParquetSink(path=target, mode="create")
    batches = [pa.record_batch({"a": [1]})]
    list(sink.execute(batches))
    assert target.exists()
    with pytest.raises(FileExistsError):
        list(ParquetSink(path=target, mode="create").execute(batches))


def test_concurrent_create_only_one_wins(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    barrier = threading.Barrier(2)
    results: list[Exception | None] = [None, None]

    # each worker writes a value-disjoint dataset (worker idx -> [idx*100 + i]),
    # so a corrupted interleave of the two staging streams is detectable: the
    # published file must contain exactly one worker's rows, never a mix.
    def make_batches(idx: int) -> list[pa.RecordBatch]:
        return [pa.record_batch({"a": [idx * 100 + i]}) for i in range(4)]

    def worker(idx: int) -> None:
        sink = ParquetSink(path=target, mode="create")
        try:
            gen = sink.execute(iter(make_batches(idx)))
            first = next(gen)
            barrier.wait(timeout=5)
            _ = [first, *gen]
        except Exception as exc:
            results[idx] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10)

    winners = [i for i, r in enumerate(results) if r is None]
    losers = [r for r in results if isinstance(r, Exception)]
    assert len(winners) == 1, f"expected exactly one winner, got results={results}"
    assert len(losers) == 1
    assert target.exists()

    # published content must be exactly the winner's dataset — no interleave,
    # no partial rows from the loser's stream.
    expected = pa.Table.from_batches(make_batches(winners[0]))
    written = pq.read_table(str(target))
    assert written.equals(expected), f"corrupted publish: {written.to_pydict()}"

    # no stray staging files left behind (unique per-invocation temp names)
    assert list(tmp_path.glob("*.tmp")) == []


def test_invalid_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ParquetSink(path=tmp_path, mode="merge")


def test_execute_publishes(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    sink = ParquetSink(path=target, mode="append")
    batches = [pa.record_batch({"a": [1, 2]})]
    list(sink.execute(batches))
    assert target.exists()


def test_execute_empty_publishes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    sink = ParquetSink(path=target, mode="append")
    list(sink.execute([]))
    assert not target.exists()


# ---- cross-backend ----------------------------------------------------------


def test_sink_across_backends(backend_table: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    out = backend_table.tee(ParquetSink(path=target, mode="append")).execute()
    assert len(out) == 4
    assert len(pq.read_table(str(target))) == 4
    # second append merges into the same file
    backend_table.tee(ParquetSink(path=target, mode="append")).execute()
    assert len(pq.read_table(str(target))) == 8


# ---- mixed ops --------------------------------------------------------------


def test_sink_after_deferred_read(tmp_path: Path) -> None:
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3, 4]}), str(src))
    target = tmp_path / "out.parquet"
    expr = xo.deferred_read_parquet(path=src, con=xo.connect(), table_name="dr").tee(
        ParquetSink(path=target)
    )
    assert len(expr.execute()) == 4
    assert target.exists()


def test_sink_after_into_backend(tmp_path: Path) -> None:
    con = xo.connect()
    other = xo.connect()  # second datafusion; duckdb would deadlock (see fixture)
    t = con.create_table("ib_src", pa.table({"a": [1, 2, 3, 4]}))
    target = tmp_path / "out.parquet"
    t.into_backend(other, "ib").tee(ParquetSink(path=target)).execute()
    assert target.exists()


@pytest.mark.skip(
    reason="duckdb single connection is not re-entrant: the streaming tee pulls "
    "the parent reader while the same connection serves the outer query, which "
    "deadlocks. Phase 1 targets engines with concurrent reader pulls (datafusion)."
)
def test_sink_duckdb_streaming_deadlocks(tmp_path: Path) -> None:
    con = xo.duckdb.connect()
    t = con.create_table("dd_src", pa.table({"a": [1, 2, 3, 4]}))
    t.tee(ParquetSink(path=tmp_path / "out.parquet")).execute()


def test_sink_with_cache_upstream(tmp_path: Path) -> None:
    con = xo.connect()
    t = con.create_table("c_src", pa.table({"a": [1, 2, 3, 4]}))
    target = tmp_path / "out.parquet"
    t.cache().tee(ParquetSink(path=target)).execute()
    assert target.exists()


def test_sink_in_middle_writes_full_parent(t: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    # a downstream filter reduces the result, but the tee wrote the full parent
    out = t.tee(ParquetSink(path=target)).filter(xo._.a > 2).execute()
    assert len(out) == 2
    assert len(pq.read_table(str(target))) == 4


def test_chained_sinks_fan_out(t: Table, tmp_path: Path) -> None:
    f1, f2 = tmp_path / "t1.parquet", tmp_path / "t2.parquet"
    t.tee(ParquetSink(path=f1)).tee(ParquetSink(path=f2)).execute()
    assert f1.exists()
    assert f2.exists()


# ---- BackendSink ------------------------------------------------------------


def test_backend_sink_creates_table(t: Table) -> None:
    target_con = xo.connect()
    t.tee(target_con, table_name="bs_tgt", mode="create").execute()
    result = target_con.table("bs_tgt").execute()
    assert len(result) == len(t.execute())


def test_backend_sink_via_explicit_sink_node(t: Table) -> None:
    target_con = xo.connect()
    sink = BackendSink(target_con, table_name="bs_explicit", mode="create")
    t.tee(sink).execute()
    result = target_con.table("bs_explicit").execute()
    assert len(result) == len(t.execute())


def test_backend_sink_append_mode(t: Table) -> None:
    # DataFusion re-registers the table on each call (no true append), so the
    # second run overwrites rather than accumulating.  Backends with mode
    # support (Postgres via ADBC) would accumulate rows.
    target_con = xo.connect()
    t.tee(target_con, table_name="bs_app", mode="create").execute()
    t.tee(target_con, table_name="bs_app", mode="append").execute()
    result = target_con.table("bs_app").execute()
    assert len(result) == len(t.execute())


def test_tee_rejects_invalid_target(t: Table) -> None:
    with pytest.raises(TypeError, match="SinkNode or a backend"):
        t.tee("not_a_backend")


def test_backend_sink_bulk_writes_nothing_on_error() -> None:
    # Bulk backends (no mode support, e.g. DataFusion) register the table after
    # full stream exhaustion.  A mid-stream error means nothing is written.
    target_con = xo.connect()

    def exploding_batches():
        yield pa.record_batch({"a": [1, 2]})
        raise RuntimeError("simulated mid-stream failure")

    sink = BackendSink(target_con, table_name="bs_err", mode="create")
    with pytest.raises(RuntimeError, match="simulated"):
        list(sink.execute(exploding_batches()))
    with pytest.raises(ValueError, match="Table not found"):
        target_con.table("bs_err")


def test_backend_sink_bulk_registers_all_batches() -> None:
    # Bulk backends register all batches in a single call after exhaustion,
    # so the resulting table contains every batch — not just the last one.
    target_con = xo.connect()
    batches = [pa.record_batch({"a": [1, 2]}), pa.record_batch({"a": [3, 4]})]
    sink = BackendSink(target_con, table_name="bs_bulk", mode="create")
    list(sink.execute(batches))
    result = target_con.table("bs_bulk").execute()
    assert len(result) == 4


# ---- generator lifecycle / cleanup -------------------------------------------


def test_parquet_execute_abandoned_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    sink = ParquetSink(path=target, mode="append")
    batches = [pa.record_batch({"a": [1, 2]}), pa.record_batch({"a": [3, 4]})]
    gen = sink.execute(iter(batches))
    next(gen)  # consume one batch, open the writer
    gen.close()  # abandon mid-stream
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


# ---- per-batch ingest path (BackendSink with mode support) -------------------


class _FakePerBatchBackend:
    """Minimal fake backend whose read_record_batches accepts a mode kwarg."""

    name = "fake_per_batch"

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def read_record_batches(
        self, source: Any, table_name: str | None = None, mode: str | None = None
    ) -> None:
        self.calls.append((table_name, mode, list(source)))


def test_per_batch_path_create_then_append() -> None:
    backend = _FakePerBatchBackend()
    sink = BackendSink(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    result = list(sink.execute(batches))
    assert len(result) == 2
    assert backend.calls[0][1] == "create"
    assert all(c[1] == "append" for c in backend.calls[1:])


def test_per_batch_path_append_mode() -> None:
    backend = _FakePerBatchBackend()
    sink = BackendSink(backend, table_name="t", mode="append")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    list(sink.execute(batches))
    assert all(c[1] == "append" for c in backend.calls)


def test_per_batch_partial_write_on_error() -> None:
    class _FailOnSecond(_FakePerBatchBackend):
        def read_record_batches(
            self, source: Any, table_name: str | None = None, mode: str | None = None
        ) -> None:
            batches = list(source)
            if len(self.calls) == 1:
                raise RuntimeError("simulated failure on second batch")
            self.calls.append((table_name, mode, batches))

    backend = _FailOnSecond()
    sink = BackendSink(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    with pytest.raises(RuntimeError, match="simulated"):
        list(sink.execute(batches))
    assert len(backend.calls) == 1


# ---- tee() argument validation -----------------------------------------------


def test_tee_rejects_extra_kwargs_with_sink_node(t: Table, tmp_path: Path) -> None:
    sink = ParquetSink(path=tmp_path / "out.parquet")
    with pytest.raises(TypeError, match="does not accept"):
        t.tee(sink, table_name="oops")


def test_tee_requires_table_name_for_backend(t: Table) -> None:
    target_con = xo.connect()
    with pytest.raises(TypeError, match="requires table_name"):
        t.tee(target_con)


def test_tee_rejects_backend_without_read_record_batches(t: Table) -> None:
    class _NoRRB:
        name = "fake"

    with pytest.raises(TypeError, match="SinkNode or a backend"):
        t.tee(_NoRRB(), table_name="tgt")


# ---- expression-tree transformation layer ------------------------------------


def test_to_sql_strips_tee_node(t: Table, tmp_path: Path) -> None:
    bare_sql = xo.to_sql(t)
    teed_sql = xo.to_sql(t.tee(ParquetSink(path=tmp_path / "out.parquet")))
    assert bare_sql == teed_sql


def test_to_sql_strips_nested_tee_nodes(t: Table, tmp_path: Path) -> None:
    bare_sql = xo.to_sql(t)
    double_tee = t.tee(ParquetSink(path=tmp_path / "t1.parquet")).tee(
        ParquetSink(path=tmp_path / "t2.parquet")
    )
    assert xo.to_sql(double_tee) == bare_sql


def test_nested_tee_hash_is_transparent(t: Table, tmp_path: Path) -> None:
    strategy = SnapshotStrategy()
    double_tee = t.tee(ParquetSink(path=tmp_path / "t1.parquet")).tee(
        ParquetSink(path=tmp_path / "t2.parquet")
    )
    assert compute_expr_hash(double_tee, strategy=strategy) == compute_expr_hash(
        t, strategy=strategy
    )


def test_tee_plus_tag_hash_is_transparent(t: Table, tmp_path: Path) -> None:
    strategy = SnapshotStrategy()
    tee_then_tag = t.tee(ParquetSink(path=tmp_path / "out.parquet")).tag("v1")
    tag_then_tee = t.tag("v1").tee(ParquetSink(path=tmp_path / "out2.parquet"))
    base_hash = compute_expr_hash(t, strategy=strategy)
    assert compute_expr_hash(tee_then_tag, strategy=strategy) == base_hash
    assert compute_expr_hash(tag_then_tee, strategy=strategy) == base_hash


def test_to_sql_with_tag_and_tee(t: Table, tmp_path: Path) -> None:
    bare_sql = xo.to_sql(t)
    assert (
        xo.to_sql(t.tee(ParquetSink(path=tmp_path / "out.parquet")).tag("v1"))
        == bare_sql
    )
    assert (
        xo.to_sql(t.tag("v1").tee(ParquetSink(path=tmp_path / "out2.parquet")))
        == bare_sql
    )


# ---- BackendSink kwargs passthrough ------------------------------------------


class _FakeKwargsBackend:
    """Fake backend that records all kwargs passed to read_record_batches."""

    name = "fake_kwargs"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def read_record_batches(self, source: Any, **kwargs: Any) -> None:
        kwargs["_batches"] = list(source)
        self.calls.append(kwargs)


def test_backend_sink_extra_kwargs_reach_backend() -> None:
    backend = _FakeKwargsBackend()
    sink = BackendSink(
        backend, table_name="t", mode="create", kwargs={"custom_opt": 42}
    )
    batches = [pa.record_batch({"a": [1]})]
    list(sink.execute(batches))
    assert len(backend.calls) == 1
    assert backend.calls[0]["custom_opt"] == 42
    assert backend.calls[0]["table_name"] == "t"


# ---- ParquetSink multi-batch -------------------------------------------------


def test_parquet_multi_batch_single_file(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    sink = ParquetSink(path=target, mode="append")
    batches = [
        pa.record_batch({"a": [1, 2]}),
        pa.record_batch({"a": [3, 4]}),
        pa.record_batch({"a": [5, 6]}),
    ]
    list(sink.execute(batches))
    assert len(pq.read_table(str(target))) == 6


# ---- ParquetSink publish failure cleanup --------------------------------------


def test_parquet_create_link_failure_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.parquet"
    sink = ParquetSink(path=target, mode="create")
    batches = [pa.record_batch({"a": [1, 2]})]

    def failing_link(src, dst):
        raise OSError("simulated link failure")

    monkeypatch.setattr(os, "link", failing_link)
    with pytest.raises(OSError, match="simulated link failure"):
        list(sink.execute(batches))
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_parquet_append_rename_failure_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.parquet"
    sink = ParquetSink(path=target, mode="append")
    batches = [pa.record_batch({"a": [1, 2]})]

    original_rename = Path.rename

    def failing_rename(self, tgt):
        if str(self).endswith(".tmp"):
            raise OSError("simulated rename failure")
        return original_rename(self, tgt)

    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(OSError, match="simulated rename failure"):
        list(sink.execute(batches))
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


# ---- per-batch partial write retains committed data --------------------------


def test_per_batch_error_retains_first_batch_data() -> None:
    class _FailOnSecondBatch(_FakePerBatchBackend):
        def read_record_batches(
            self,
            source: Any,
            table_name: str | None = None,
            mode: str | None = None,
        ) -> None:
            batches = list(source)
            self.calls.append((table_name, mode, batches))
            if len(self.calls) == 2:
                raise RuntimeError("simulated failure on second batch")

    backend = _FailOnSecondBatch()
    sink = BackendSink(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    with pytest.raises(RuntimeError, match="simulated"):
        list(sink.execute(batches))
    assert len(backend.calls) == 2
    assert backend.calls[0][1] == "create"
    assert len(backend.calls[0][2]) == 1


# ---- threaded ingest path (ThreadedBackendSink) ------------------------------


class _RecordingBackend:
    """Fake mode-capable backend that drains the reader as a stream.

    Records one entry per ``read_record_batches`` call (the threaded path makes
    exactly one) holding the table name, mode, kwargs, and the batches it pulled
    — in pull order. ``on_batch(idx, batch)`` fires as each batch is drained, so
    a test can observe consumption interleaving with production.
    """

    name = "recording"

    def __init__(self, on_batch=None) -> None:
        self.calls: list[dict] = []
        self.worker: threading.Thread | None = None
        self._on_batch = on_batch

    def read_record_batches(
        self,
        source: Any,
        table_name: str | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.worker = threading.current_thread()
        batches: list = []
        for batch in source:
            if self._on_batch is not None:
                self._on_batch(len(batches), batch)
            batches.append(batch)
        self.calls.append(
            {
                "table_name": table_name,
                "mode": mode,
                "kwargs": kwargs,
                "batches": batches,
            }
        )


def _first(batch) -> int:
    return batch.column("a")[0].as_py()


def test_threaded_single_call_all_batches() -> None:
    # mode-capable backend: threaded path makes ONE read_record_batches call
    # carrying every batch, not one call per batch.
    backend = _FakePerBatchBackend()
    sink = ThreadedBackendSink(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    result = list(sink.execute(batches))
    assert len(result) == 2  # passthrough: every batch yielded downstream
    assert len(backend.calls) == 1
    table_name, mode, ingested = backend.calls[0]
    assert table_name == "t"
    assert mode == "create"
    assert len(ingested) == 2


def test_threaded_no_mode_backend_omits_mode() -> None:
    # backend without a `mode` parameter: mode must NOT be forwarded (the bug
    # the _ingest gate fixes), and all batches still land in one call.
    backend = _FakeKwargsBackend()
    sink = ThreadedBackendSink(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    list(sink.execute(batches))
    assert len(backend.calls) == 1
    assert "mode" not in backend.calls[0]
    assert len(backend.calls[0]["_batches"]) == 2


def test_threaded_kwargs_reach_backend() -> None:
    backend = _FakeKwargsBackend()
    sink = ThreadedBackendSink(
        backend, table_name="t", mode="create", kwargs={"custom_opt": 42}
    )
    list(sink.execute([pa.record_batch({"a": [1]})]))
    assert backend.calls[0]["custom_opt"] == 42


def test_threaded_empty_stream_no_call() -> None:
    backend = _FakePerBatchBackend()
    sink = ThreadedBackendSink(backend, table_name="t", mode="create")
    assert list(sink.execute([])) == []
    assert backend.calls == []


def test_threaded_sink_thread_error_propagates() -> None:
    # an error raised inside read_record_batches is captured on the thread and
    # re-raised on the main thread after the join.
    class _Exploding(_FakePerBatchBackend):
        def read_record_batches(
            self, source: Any, table_name: str | None = None, mode: str | None = None
        ) -> None:
            list(source)  # drain the queue so the producer never blocks
            raise RuntimeError("simulated ingest failure")

    backend = _Exploding()
    sink = ThreadedBackendSink(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    with pytest.raises(RuntimeError, match="simulated ingest failure"):
        list(sink.execute(batches))


def test_threaded_upstream_error_propagates_and_joins() -> None:
    backend = _RecordingBackend()
    sink = ThreadedBackendSink(backend, table_name="t", mode="create")

    def exploding_batches():
        yield pa.record_batch({"a": [1, 2]})
        raise RuntimeError("simulated mid-stream failure")

    with pytest.raises(RuntimeError, match="simulated mid-stream failure"):
        list(sink.execute(exploding_batches()))
    # the worker saw a truncated stream and was joined before execute returned
    assert backend.worker is not None
    assert not backend.worker.is_alive()


def test_threaded_early_stop_ends_reader_and_joins() -> None:
    # abandoning the generator mid-stream signals the reader to end short; the
    # worker drains what it has and is joined — no hang, no leaked thread.
    backend = _RecordingBackend()
    sink = ThreadedBackendSink(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [i]}) for i in range(5)]
    gen = sink.execute(iter(batches))
    next(gen)
    gen.close()  # GeneratorExit -> reader ends short, worker joins
    assert backend.worker is not None
    assert not backend.worker.is_alive()


def test_threaded_preserves_order_and_content() -> None:
    # passthrough is identity in order, and the sink pulls in the same order.
    backend = _RecordingBackend()
    sink = ThreadedBackendSink(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [i]}) for i in range(10)]
    out = list(sink.execute(iter(batches)))
    assert [_first(b) for b in out] == list(range(10))
    assert [_first(b) for b in backend.calls[0]["batches"]] == list(range(10))


def test_threaded_forwards_append_mode() -> None:
    backend = _RecordingBackend()
    sink = ThreadedBackendSink(backend, table_name="t", mode="append")
    list(sink.execute([pa.record_batch({"a": [1]})]))
    assert backend.calls[0]["mode"] == "append"


def test_threaded_streams_without_buffering() -> None:
    # proves concurrency: the sink consumes batch 0 while the producer is still
    # blocked before yielding batch 1.  A buffer-everything-then-ingest path
    # would never set the event during production and time out.
    received_first = threading.Event()

    def on_batch(idx: int, batch) -> None:
        if idx == 0:
            received_first.set()

    backend = _RecordingBackend(on_batch=on_batch)
    sink = ThreadedBackendSink(backend, table_name="t", mode="create")

    def producer():
        yield pa.record_batch({"a": [0]})
        assert received_first.wait(timeout=5), "sink did not consume batch 0 early"
        yield pa.record_batch({"a": [1]})

    out = list(sink.execute(producer()))
    assert [_first(b) for b in out] == [0, 1]
    assert len(backend.calls[0]["batches"]) == 2


def test_threaded_joins_worker_before_return() -> None:
    backend = _RecordingBackend()
    sink = ThreadedBackendSink(backend, table_name="t", mode="create")
    list(sink.execute([pa.record_batch({"a": [i]}) for i in range(3)]))
    # the single ingest completed (one recorded call) and its worker is dead
    assert len(backend.calls) == 1
    assert backend.worker is not None
    assert not backend.worker.is_alive()


def test_threaded_sink_is_reusable() -> None:
    # frozen, no shared mutable state in execute(): the same sink runs twice.
    backend = _RecordingBackend()
    sink = ThreadedBackendSink(backend, table_name="t", mode="create")
    list(sink.execute([pa.record_batch({"a": [1]})]))
    list(sink.execute([pa.record_batch({"a": [2]})]))
    assert len(backend.calls) == 2
    assert _first(backend.calls[0]["batches"][0]) == 1
    assert _first(backend.calls[1]["batches"][0]) == 2


def test_threaded_unbounded_queue_no_deadlock_when_sink_lags() -> None:
    # the sink refuses to pull until released; the unbounded queue lets the
    # producer push and yield all 50 batches anyway (no backpressure, no hang).
    release = threading.Event()

    class _LaggyBackend(_RecordingBackend):
        def read_record_batches(self, source, table_name=None, mode=None, **kwargs):
            assert release.wait(timeout=5)
            super().read_record_batches(
                source, table_name=table_name, mode=mode, **kwargs
            )

    backend = _LaggyBackend()
    sink = ThreadedBackendSink(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [i]}) for i in range(50)]

    out = []
    for i, batch in enumerate(sink.execute(iter(batches))):
        out.append(_first(batch))
        if i == len(batches) - 1:
            release.set()  # unblock the sink before the join on the next pull
    assert out == list(range(50))
    assert [_first(b) for b in backend.calls[0]["batches"]] == list(range(50))


def test_threaded_stops_feeding_after_sink_error() -> None:
    # the sink dies after one batch; the `if error: break` guard stops the
    # producer well short of its full run instead of pushing into a dead queue.
    sink_failed = threading.Event()

    class _FailFast(_RecordingBackend):
        def read_record_batches(self, source, table_name=None, mode=None, **kwargs):
            self.worker = threading.current_thread()
            it = iter(source)
            next(it)
            sink_failed.set()
            raise RuntimeError("simulated ingest failure")

    backend = _FailFast()
    sink = ThreadedBackendSink(backend, table_name="t", mode="create")
    produced = []

    def producer():
        for i in range(100):
            yield pa.record_batch({"a": [i]})
            produced.append(i)
            if i == 0:
                assert sink_failed.wait(timeout=5)

    with pytest.raises(RuntimeError, match="simulated ingest failure"):
        list(sink.execute(producer()))
    assert len(produced) < 100
    assert backend.worker is not None
    assert not backend.worker.is_alive()


def test_threaded_creates_table_real_backend(t: Table) -> None:
    target_con = xo.connect()
    sink = ThreadedBackendSink(target_con, table_name="th_tgt", mode="create")
    out = t.tee(sink).execute()
    assert len(out) == len(t.execute())
    assert len(target_con.table("th_tgt").execute()) == len(t.execute())
