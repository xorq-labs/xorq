from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.caching.strategy import SnapshotStrategy
from xorq.common.utils.node_utils import compute_expr_hash
from xorq.sinking import BackendSink, ParquetSink


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
    assert not Path(str(target) + ".tmp").exists()


def test_create_sink_execute_raises_fileexists(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    sink = ParquetSink(path=target, mode="create")
    batches = [pa.record_batch({"a": [1]})]
    list(sink.execute(batches))
    assert target.exists()
    with pytest.raises(FileExistsError):
        list(ParquetSink(path=target, mode="create").execute(batches))


def test_concurrent_create_only_one_wins(tmp_path: Path) -> None:
    import threading  # noqa: PLC0415

    target = tmp_path / "out.parquet"
    batches = [pa.record_batch({"a": [i]}) for i in range(4)]
    barrier = threading.Barrier(2)
    results: list[Exception | None] = [None, None]

    def worker(idx: int) -> None:
        sink = ParquetSink(path=target, mode="create")
        try:
            gen = sink.execute(iter(batches))
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

    winners = [r for r in results if r is None]
    losers = [r for r in results if isinstance(r, (FileExistsError, Exception))]
    assert len(winners) == 1, f"expected exactly one winner, got results={results}"
    assert len(losers) == 1
    assert target.exists()
    assert not Path(str(target) + ".tmp").exists()


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
    assert not Path(str(target) + ".tmp").exists()


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
    import os as _os  # noqa: PLC0415

    target = tmp_path / "out.parquet"
    sink = ParquetSink(path=target, mode="create")
    batches = [pa.record_batch({"a": [1, 2]})]

    def failing_link(src, dst):
        raise OSError("simulated link failure")

    monkeypatch.setattr(_os, "link", failing_link)
    with pytest.raises(OSError, match="simulated link failure"):
        list(sink.execute(batches))
    assert not target.exists()
    assert not Path(str(target) + ".tmp").exists()


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
    assert not Path(str(target) + ".tmp").exists()


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
