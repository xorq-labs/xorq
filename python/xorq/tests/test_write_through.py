from __future__ import annotations

import importlib.metadata
import os
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.caching.strategy import SnapshotStrategy
from xorq.common.utils.graph_utils import walk_nodes
from xorq.common.utils.node_utils import compute_expr_hash
from xorq.common.utils.provenance_utils import get_expr_hash
from xorq.expr.api import (
    _close_and_join_drains,
    _drop_created_tables,
    _run_cleanup,
    get_plans,
)
from xorq.expr.relations import (
    HashingTag,
    TeeNode,
    register_and_transform_tee_nodes,
)
from xorq.writes import (
    BackendWriteThrough,
    DrainingIterator,
    ParquetWriteThrough,
    ThreadedBackendWriteThrough,
    WritePrimaryWriteThrough,
    WriteThrough,
)


if TYPE_CHECKING:
    from xorq.vendor.ibis.expr.types import Table


TABLE = {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]}


@pytest.fixture
def t() -> Table:
    con = xo.connect()
    return con.register(xo.memtable(TABLE), table_name="t0")


# Early-stop tests need a multi-batch source so the parent's un-pulled tail is
# what the drain/suppression assertions turn on; a single batch can't express it
# (the 0.2.9 EOF probe pulls it to exhaustion). Count is 2x the ~4-batch
# read-ahead for margin; widen if a future bump grows the window -- the positive
# control in test_tee_drain_false_does_not_drain fails loud if it doesn't. See
# ADR-0014 for the full read-ahead / probe rationale.
_DATAFUSION_READ_AHEAD = 4
_MULTI_BATCH_COUNT = 2 * _DATAFUSION_READ_AHEAD

# EOF probe lands in 0.2.9 (#1977); missing package -> pre-probe defaults.
try:
    _raw_version = importlib.metadata.version("xorq-datafusion")
except importlib.metadata.PackageNotFoundError:
    _DATAFUSION_VERSION = (0, 0, 0)
else:
    _DATAFUSION_VERSION = tuple(
        int(part) for part in re.findall(r"\d+", _raw_version)[:3]
    )
_DATAFUSION_HAS_EOF_PROBE = _DATAFUSION_VERSION >= (0, 2, 9)


def _multi_batch_source(con: Any, table_name: str) -> Table:
    batches = [
        pa.record_batch({"a": [i], "b": [str(i)]}) for i in range(_MULTI_BATCH_COUNT)
    ]
    reader = pa.RecordBatchReader.from_batches(batches[0].schema, iter(batches))
    return con.read_record_batches(reader, table_name=table_name)


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
    return con.create_table("write_src", pa.table(TABLE))


def test_write_is_passthrough(t: Table, tmp_path: Path) -> None:
    expr = t.tee(ParquetWriteThrough(path=tmp_path / "out.parquet"))
    assert expr.schema() == t.schema()
    assert expr.execute().equals(t.execute())


def test_write_writes_what_flows(t: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    t.tee(ParquetWriteThrough(path=target)).execute()
    written = pq.read_table(str(target))
    assert len(written) == len(t.execute())


def test_write_hash_is_transparent(t: Table, tmp_path: Path) -> None:
    strategy = SnapshotStrategy()
    teed = t.tee(ParquetWriteThrough(path=tmp_path / "out.parquet"))
    assert compute_expr_hash(teed, strategy=strategy) == compute_expr_hash(
        t, strategy=strategy
    )


def test_build_hash_distinguishes_writers(t: Table, tmp_path: Path) -> None:
    a = t.tee(ParquetWriteThrough(path=tmp_path / "a.parquet"))
    b = t.tee(ParquetWriteThrough(path=tmp_path / "b.parquet"))
    hash_a = get_expr_hash(a)
    hash_b = get_expr_hash(b)
    assert hash_a != hash_b, "different writers must produce different build hashes"


def test_cache_hit_does_not_write(t: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    cached = t.tee(ParquetWriteThrough(path=target)).cache()
    cached.execute()  # miss: writes
    mtime = target.stat().st_mtime_ns
    cached.execute()  # hit: must not write again
    assert target.stat().st_mtime_ns == mtime


def test_append_accumulates_rows(t: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    t.tee(ParquetWriteThrough(path=target, mode="append")).execute()
    assert len(pq.read_table(str(target))) == 4
    t.tee(ParquetWriteThrough(path=target, mode="append")).execute()
    assert len(pq.read_table(str(target))) == 8


def test_create_fails_if_target_exists(t: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    t.tee(ParquetWriteThrough(path=target, mode="create")).execute()
    assert target.exists()
    # raised mid-pull, the engine wraps FileExistsError (Arrow C-stream boundary)
    with pytest.raises(Exception, match="already exists"):
        t.tee(ParquetWriteThrough(path=target, mode="create")).execute()
    # the failed run published nothing: original file intact, no stray temp
    assert len(pq.read_table(str(target))) == 4
    assert not list(target.parent.glob("*.tmp"))


def test_create_write_raises_fileexists(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    writer = ParquetWriteThrough(path=target, mode="create")
    batches = [pa.record_batch({"a": [1]})]
    list(writer.write_through(batches))
    assert target.exists()
    with pytest.raises(FileExistsError):
        list(ParquetWriteThrough(path=target, mode="create").write_through(batches))


def test_concurrent_create_only_one_wins(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    batches = [pa.record_batch({"a": [i]}) for i in range(4)]
    barrier = threading.Barrier(2)
    results: list[Exception | None] = [None, None]

    def worker(idx: int) -> None:
        writer = ParquetWriteThrough(path=target, mode="create")
        try:
            gen = writer.write_through(iter(batches))
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
    assert not list(target.parent.glob("*.tmp"))


def test_concurrent_append_no_lost_rows(tmp_path: Path) -> None:
    # The permanent lock file must serialize concurrent appends so every
    # writer's rows survive. Unlinking the lock let appenders race on separate
    # inodes (last merge-then-rename wins, dropping the others' rows).
    target = tmp_path / "out.parquet"
    n = 3
    barrier = threading.Barrier(n)
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout=5)
            writer = ParquetWriteThrough(path=target, mode="append")
            list(writer.write_through([pa.record_batch({"a": [i]})]))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10)

    assert not errors, errors
    assert target.exists()
    result = pq.read_table(str(target)).column("a").to_pylist()
    assert sorted(result) == [0, 1, 2]


def test_invalid_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ParquetWriteThrough(path=tmp_path, mode="merge")


def test_write_publishes(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    writer = ParquetWriteThrough(path=target, mode="append")
    batches = [pa.record_batch({"a": [1, 2]})]
    list(writer.write_through(batches))
    assert target.exists()


def test_write_empty_publishes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    writer = ParquetWriteThrough(path=target, mode="append")
    list(writer.write_through([]))
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


# ---- cross-backend ----------------------------------------------------------


def test_write_across_backends(backend_table: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    out = backend_table.tee(ParquetWriteThrough(path=target, mode="append")).execute()
    assert len(out) == 4
    assert len(pq.read_table(str(target))) == 4
    # second append merges into the same file
    backend_table.tee(ParquetWriteThrough(path=target, mode="append")).execute()
    assert len(pq.read_table(str(target))) == 8


# ---- mixed ops --------------------------------------------------------------


def test_write_after_deferred_read(tmp_path: Path) -> None:
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3, 4]}), str(src))
    target = tmp_path / "out.parquet"
    expr = xo.deferred_read_parquet(path=src, con=xo.connect(), table_name="dr").tee(
        ParquetWriteThrough(path=target)
    )
    assert len(expr.execute()) == 4
    assert target.exists()


def test_write_after_into_backend(tmp_path: Path) -> None:
    con = xo.connect()
    other = xo.connect()  # second datafusion; duckdb would deadlock (see fixture)
    t = con.create_table("ib_src", pa.table({"a": [1, 2, 3, 4]}))
    target = tmp_path / "out.parquet"
    t.into_backend(other, "ib").tee(ParquetWriteThrough(path=target)).execute()
    assert target.exists()


@pytest.mark.skip(
    reason="duckdb single connection is not re-entrant: the streaming tee pulls "
    "the parent reader while the same connection serves the outer query, which "
    "deadlocks. Phase 1 targets engines with concurrent reader pulls (datafusion)."
)
def test_write_duckdb_streaming_deadlocks(tmp_path: Path) -> None:
    con = xo.duckdb.connect()
    t = con.create_table("dd_src", pa.table({"a": [1, 2, 3, 4]}))
    t.tee(ParquetWriteThrough(path=tmp_path / "out.parquet")).execute()


def test_tee_duckdb_warns_deadlock(tmp_path: Path) -> None:
    # The transform must warn before wiring a streaming tee on a non-reentrant
    # backend. duckdb registration is lazy, so the transform itself does not
    # deadlock — only a later execute would — which lets us assert the warning.
    con = _connect("duckdb")
    t = con.create_table("dd_warn", pa.table({"a": [1, 2, 3, 4]}))
    expr = t.tee(ParquetWriteThrough(path=tmp_path / "out.parquet"))
    with pytest.warns(UserWarning, match="deadlock"):
        register_and_transform_tee_nodes(expr)


def test_write_with_cache_upstream(tmp_path: Path) -> None:
    con = xo.connect()
    t = con.create_table("c_src", pa.table({"a": [1, 2, 3, 4]}))
    target = tmp_path / "out.parquet"
    t.cache().tee(ParquetWriteThrough(path=target)).execute()
    assert target.exists()


def test_write_in_middle_writes_full_parent(t: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    # a downstream filter reduces the result, but the tee wrote the full parent
    out = t.tee(ParquetWriteThrough(path=target)).filter(xo._.a > 2).execute()
    assert len(out) == 2
    assert len(pq.read_table(str(target))) == 4


def test_chained_writers_fan_out(t: Table, tmp_path: Path) -> None:
    f1, f2 = tmp_path / "t1.parquet", tmp_path / "t2.parquet"
    t.tee(ParquetWriteThrough(path=f1)).tee(ParquetWriteThrough(path=f2)).execute()
    assert f1.exists()
    assert f2.exists()


def test_chained_tees_call_each_write_once(t: Table) -> None:
    class _CountingWriteThrough(WriteThrough):
        def __init__(self):
            self.call_count = 0

        def write_through(self, batches):
            self.call_count += 1
            return (batch for batch in batches)

    s1, s2 = _CountingWriteThrough(), _CountingWriteThrough()
    t.tee(s1).tee(s2).execute()
    assert s1.call_count == 1, f"inner writer called {s1.call_count} times, expected 1"
    assert s2.call_count == 1, f"outer writer called {s2.call_count} times, expected 1"


# ---- BackendWriteThrough ------------------------------------------------------------


def test_backend_write_creates_table(t: Table) -> None:
    target_con = xo.connect()
    t.tee(target_con, table_name="bs_tgt", mode="create").execute()
    result = target_con.table("bs_tgt").execute()
    assert len(result) == len(t.execute())


def test_backend_write_via_explicit_write_node(t: Table) -> None:
    target_con = xo.connect()
    writer = BackendWriteThrough(target_con, table_name="bs_explicit", mode="create")
    t.tee(writer).execute()
    result = target_con.table("bs_explicit").execute()
    assert len(result) == len(t.execute())


def test_backend_write_append_mode(t: Table) -> None:
    # DataFusion re-registers the table on each call (no true append), so the
    # second run overwrites rather than accumulating.  Backends with mode
    # support (Postgres via ADBC) would accumulate rows.
    target_con = xo.connect()
    t.tee(target_con, table_name="bs_app", mode="create").execute()
    t.tee(target_con, table_name="bs_app", mode="append").execute()
    result = target_con.table("bs_app").execute()
    assert len(result) == len(t.execute())


def test_tee_drops_intermediate_table(tmp_path: Path) -> None:
    # The tee pass registers a pass-through table on the parent backend to feed
    # the writer; it must be dropped after execution rather than accumulating in
    # the backend catalog (mirrors the RemoteTable cleanup).
    con = xo.connect()
    t0 = con.register(xo.memtable(TABLE), table_name="t0")
    before = set(con.list_tables())
    t0.tee(tmp_path / "out.parquet").execute()
    assert set(con.list_tables()) == before


def test_tee_drops_intermediate_table_on_early_stop(tmp_path: Path) -> None:
    # Same cleanup guarantee as full consumption, but the downstream stops early
    # (.limit) and drain=True finishes the write. The intermediate pass-through
    # table must still be dropped once the drain joins.
    con = xo.connect()
    t0 = con.register(xo.memtable(TABLE), table_name="t0")
    target = tmp_path / "out.parquet"
    before = set(con.list_tables())
    out = t0.tee(ParquetWriteThrough(path=target), drain=True).limit(2).execute()
    assert len(out) == 2
    assert len(pq.read_table(str(target))) == 4  # drain wrote the full parent
    assert set(con.list_tables()) == before


def test_draining_iterator_serializes_concurrent_advance() -> None:
    # Deterministic guard for #2105: __next__ and _drain must never advance the
    # generator concurrently. The sleeps force the overlap that would otherwise
    # raise 'generator already executing'.
    started = threading.Event()

    def slow_gen():
        for i in range(50):
            started.set()
            time.sleep(0.001)
            yield pa.record_batch({"a": [i]})

    di = DrainingIterator(slow_gen())
    errors: list[BaseException] = []

    def foreground():
        try:
            for _ in di:
                time.sleep(0.0005)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=foreground)
    thread.start()
    started.wait()
    di.close()  # start the background drain while the foreground is mid-pull
    thread.join()
    di.join()

    assert not errors, errors
    assert di.exhausted


def test_tee_partial_consumption_leaks_intermediate_table(tmp_path: Path) -> None:
    # KNOWN HAZARD (see the to_pyarrow_batches docstring / ADR-0014): cleanup
    # runs only after the reader is fully consumed. Abandoning it after a partial
    # read leaks the intermediate pass-through table. Pinned to lock the contract
    # the docstring promises; tighten this test if partial cleanup is ever added.
    con = xo.connect()
    t0 = con.register(xo.memtable(TABLE), table_name="t0")
    target = tmp_path / "out.parquet"
    before = set(con.list_tables())
    reader = t0.tee(ParquetWriteThrough(path=target)).to_pyarrow_batches(chunk_size=1)
    reader.read_next_batch()  # consume one batch only, then abandon
    assert set(con.list_tables()) - before, (
        "expected the intermediate tee table to leak on partial consumption"
    )


def test_tee_rejects_invalid_target(t: Table) -> None:
    with pytest.raises(TypeError, match="must be a WriteThrough"):
        t.tee(42)


def test_backend_write_bulk_writes_nothing_on_error() -> None:
    # Bulk backends (no mode support, e.g. DataFusion) register the table after
    # full stream exhaustion.  A mid-stream error means nothing is written.
    target_con = xo.connect()

    def exploding_batches():
        yield pa.record_batch({"a": [1, 2]})
        raise RuntimeError("simulated mid-stream failure")

    writer = BackendWriteThrough(target_con, table_name="bs_err", mode="create")
    with pytest.raises(RuntimeError, match="simulated"):
        list(writer.write_through(exploding_batches()))
    with pytest.raises(ValueError, match="Table not found"):
        target_con.table("bs_err")


def test_backend_write_bulk_registers_all_batches() -> None:
    # Bulk backends register all batches in a single call after exhaustion,
    # so the resulting table contains every batch — not just the last one.
    target_con = xo.connect()
    batches = [pa.record_batch({"a": [1, 2]}), pa.record_batch({"a": [3, 4]})]
    writer = BackendWriteThrough(target_con, table_name="bs_bulk", mode="create")
    list(writer.write_through(batches))
    result = target_con.table("bs_bulk").execute()
    assert len(result) == 4


def test_bulk_path_warns_not_atomic() -> None:
    # Bulk backends (no mode support, e.g. DataFusion) deliver every batch
    # downstream before the single post-stream write, so the write is not
    # atomic. The bulk path must warn callers of that hazard. Drive the writer
    # directly: under .execute() datafusion pulls the reader on a worker thread
    # where pytest.warns cannot observe the warning.
    target_con = xo.connect()
    writer = BackendWriteThrough(target_con, table_name="bulk_warn", mode="create")
    with pytest.warns(UserWarning, match="bulk path"):
        list(writer.write_through([pa.record_batch({"a": [1, 2]})]))


# ---- generator lifecycle / cleanup -------------------------------------------


def test_parquet_write_abandoned_cleans_up(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    writer = ParquetWriteThrough(path=target, mode="append")
    batches = [pa.record_batch({"a": [1, 2]}), pa.record_batch({"a": [3, 4]})]
    gen = writer.write_through(iter(batches))
    next(gen)  # consume one batch, open the writer
    gen.close()  # abandon mid-stream
    assert not target.exists()
    assert not list(target.parent.glob("*.tmp"))


# ---- per-batch ingest path (BackendWriteThrough with mode support) -------------------


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
    writer = BackendWriteThrough(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    result = list(writer.write_through(batches))
    assert len(result) == 2
    assert backend.calls[0][1] == "create"
    assert all(c[1] == "append" for c in backend.calls[1:])


def test_per_batch_path_append_mode() -> None:
    backend = _FakePerBatchBackend()
    writer = BackendWriteThrough(backend, table_name="t", mode="append")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    list(writer.write_through(batches))
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
    writer = BackendWriteThrough(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    with pytest.raises(RuntimeError, match="simulated"):
        list(writer.write_through(batches))
    assert len(backend.calls) == 1


# ---- tee() argument validation -----------------------------------------------


def test_tee_rejects_extra_kwargs_with_write_node(t: Table, tmp_path: Path) -> None:
    writer = ParquetWriteThrough(path=tmp_path / "out.parquet")
    with pytest.raises(TypeError, match="does not accept"):
        t.tee(writer, table_name="oops")


def test_tee_path_target_builds_parquet_writer(t: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    t.tee(target).execute()
    assert len(pq.read_table(str(target))) == len(t.execute())


def test_tee_path_target_forwards_mode(t: Table, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    t.tee(str(target), mode="append").execute()
    t.tee(str(target), mode="append").execute()
    assert len(pq.read_table(str(target))) == 2 * len(t.execute())


def test_tee_path_target_rejects_table_name(t: Table, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="does not accept table_name"):
        t.tee(tmp_path / "out.parquet", table_name="oops")


def test_tee_requires_table_name_for_backend(t: Table) -> None:
    target_con = xo.connect()
    with pytest.raises(TypeError, match="requires table_name"):
        t.tee(target_con)


def test_tee_rejects_backend_without_read_record_batches(t: Table) -> None:
    class _NoRRB:
        name = "fake"

    with pytest.raises(TypeError, match="must be a WriteThrough"):
        t.tee(_NoRRB(), table_name="tgt")


# ---- expression-tree transformation layer ------------------------------------


def test_to_sql_strips_tee_node(t: Table, tmp_path: Path) -> None:
    bare_sql = xo.to_sql(t)
    teed_sql = xo.to_sql(t.tee(ParquetWriteThrough(path=tmp_path / "out.parquet")))
    assert bare_sql == teed_sql


def test_to_sql_strips_nested_tee_nodes(t: Table, tmp_path: Path) -> None:
    bare_sql = xo.to_sql(t)
    double_tee = t.tee(ParquetWriteThrough(path=tmp_path / "t1.parquet")).tee(
        ParquetWriteThrough(path=tmp_path / "t2.parquet")
    )
    assert xo.to_sql(double_tee) == bare_sql


def test_tee_transform_leaves_tee_inside_opaque_untouched(
    t: Table, tmp_path: Path
) -> None:
    """register_and_transform_tee_nodes uses op.replace, which deliberately does
    not descend into opaque sub-exprs (here CachedNode.parent). A TeeNode buried
    in an opaque node must survive the outer pass untouched and its write must not
    fire: it is handled lazily when the opaque sub-expr executes (e.g. on a cache
    miss). Guards against a regression to replace_nodes, which would descend and
    fire the side-effecting write at the wrong time."""
    target = tmp_path / "opaque.parquet"
    cached = t.tee(ParquetWriteThrough(path=target)).cache()

    # The TeeNode lives under CachedNode.parent: the descending traversal sees
    # it, the structural (non-descending) traversal op.replace uses does not.
    assert len(walk_nodes((TeeNode,), cached)) == 1
    assert len(cached.op().find(TeeNode)) == 0

    transformed, created, drains = register_and_transform_tee_nodes(cached)

    assert not target.exists(), "outer transform must not fire the buried write"
    assert created == {}, "no pass-through table should be registered"
    assert drains == []
    assert len(walk_nodes((TeeNode,), transformed)) == 1, (
        "TeeNode inside the opaque sub-expr must survive the outer transform"
    )


def _placeholder_tables(con: Any) -> set[str]:
    return {name for name in con.list_tables() if "placeholder" in name}


def test_get_plans_does_not_write_or_leak(tmp_path: Path) -> None:
    """get_plans is a non-executing path: it must neither fire the side-effect
    write nor leak the pass-through table it would otherwise register."""
    con = xo.connect()
    tt = con.register(xo.memtable(TABLE), table_name="t0")
    target = tmp_path / "plans.parquet"
    expr = tt.tee(ParquetWriteThrough(path=target))

    before = _placeholder_tables(con)
    get_plans(expr)

    assert not target.exists(), "EXPLAIN must not fire the deferred write"
    assert _placeholder_tables(con) == before, "tee pass-through table leaked"


def test_sql_view_does_not_write_or_leak(tmp_path: Path) -> None:
    """Defining a SQL view is a non-executing path: it must neither fire the
    side-effect write nor leak the pass-through table."""
    con = xo.connect()
    tt = con.register(xo.memtable(TABLE), table_name="t0")
    target = tmp_path / "view.parquet"
    expr = tt.tee(ParquetWriteThrough(path=target))

    before = _placeholder_tables(con)
    view = expr.sql("SELECT a FROM t0")

    assert not target.exists(), "view definition must not fire the deferred write"
    assert _placeholder_tables(con) == before, "tee pass-through table leaked"
    # The view is still usable and the tee is transparent to it.
    assert view.execute()["a"].tolist() == TABLE["a"]


def test_drop_created_tables_drops_all_and_raises(tmp_path: Path) -> None:
    """_drop_created_tables must attempt every table even when one fails its
    drop entirely, and surface the failure rather than swallowing it."""

    class _FakeCon:
        def __init__(self, *, fail_both: bool = False) -> None:
            self.fail_both = fail_both
            self.dropped: list[str] = []

        def drop_table(self, name: str, force: bool = False) -> None:
            if self.fail_both:
                raise ValueError(f"drop_table {name}")
            self.dropped.append(name)

        def drop_view(self, name: str) -> None:
            if self.fail_both:
                raise ValueError(f"drop_view {name}")
            self.dropped.append(name)

    good_a, bad, good_b = _FakeCon(), _FakeCon(fail_both=True), _FakeCon()
    created = {"a": good_a, "bad": bad, "b": good_b}

    with pytest.raises(ValueError, match="drop_view bad"):
        _drop_created_tables(created)

    # the failing table in the middle did not strand the others
    assert good_a.dropped == ["a"]
    assert good_b.dropped == ["b"]


def test_run_cleanup_joins_drains_before_dropping_tables() -> None:
    """_run_cleanup must join every drain before dropping any table, so the
    backend reader the drain consumes stays valid until the write completes."""
    log: list[tuple[str, str]] = []

    class _RecordingDrain:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            log.append(("close", self.name))

        def join(self) -> None:
            log.append(("join", self.name))

    class _RecordingCon:
        def drop_table(self, name: str, force: bool = False) -> None:
            log.append(("drop", name))

    _run_cleanup(
        [_RecordingDrain("d0"), _RecordingDrain("d1")], {"t0": _RecordingCon()}
    )

    join_idx = [i for i, (action, _) in enumerate(log) if action == "join"]
    drop_idx = [i for i, (action, _) in enumerate(log) if action == "drop"]
    assert join_idx and drop_idx
    assert max(join_idx) < min(drop_idx), f"drop ran before a join: {log}"


def test_run_cleanup_drops_tables_even_if_drain_fails() -> None:
    """A drain failure must not strand the created tables, and the drain error
    must still surface rather than being swallowed by the drop step."""
    dropped: list[str] = []

    class _FailingDrain:
        def close(self) -> None:
            pass

        def join(self) -> None:
            raise ValueError("drain boom")

    class _RecordingCon:
        def drop_table(self, name: str, force: bool = False) -> None:
            dropped.append(name)

    with pytest.raises(ValueError, match="drain boom"):
        _run_cleanup([_FailingDrain()], {"t0": _RecordingCon()})

    assert dropped == ["t0"], "table was not dropped after the drain failed"


def test_close_and_join_drains_closes_all_when_one_close_fails() -> None:
    # A close() failure on one drain (e.g. thread start fails under resource
    # exhaustion) must not skip closing the rest, nor skip the join loop that
    # reaps already-started drain threads. The close error is collected, not
    # raised mid-loop.
    log: list[tuple[str, str]] = []

    class _RecordingDrain:
        def __init__(self, name: str, fail_close: bool = False) -> None:
            self.name = name
            self.fail_close = fail_close

        def close(self) -> None:
            log.append(("close", self.name))
            if self.fail_close:
                raise RuntimeError(f"close boom {self.name}")

        def join(self, timeout: float | None = None) -> None:
            log.append(("join", self.name))

    drains = [
        _RecordingDrain("d0", fail_close=True),
        _RecordingDrain("d1"),
    ]

    with pytest.raises(BaseException) as excinfo:  # noqa: PT011
        _close_and_join_drains(drains)

    assert ("close", "d0") in log and ("close", "d1") in log, (
        f"a close() failure skipped a later close(): {log}"
    )
    assert ("join", "d1") in log, f"join loop did not run after a close failure: {log}"
    messages = [str(e) for e in _flatten_exceptions(excinfo.value)]
    assert any("close boom d0" in m for m in messages)


def test_to_pyarrow_batches_full_consumption_cleans_up(tmp_path: Path) -> None:
    """The documented contract: drains are joined and the pass-through table is
    dropped only once the reader is fully consumed (which also publishes the
    write)."""
    con = xo.connect()
    tt = con.register(xo.memtable(TABLE), table_name="t0")
    target = tmp_path / "out.parquet"

    before = _placeholder_tables(con)
    reader = tt.tee(ParquetWriteThrough(path=target)).to_pyarrow_batches()
    # the pass-through table is registered eagerly at transform time
    assert _placeholder_tables(con) - before, "tee pass-through table not registered"

    reader.read_all()

    assert target.exists(), "full consumption must publish the write"
    assert _placeholder_tables(con) == before, "pass-through table not cleaned up"


def test_to_pyarrow_batches_partial_consumption_holds_resources(
    tmp_path: Path,
) -> None:
    """Counterpart to the contract above: a partial read (an early break) does
    not run cleanup, so nothing is published and the table is held. This locks
    in the leak the to_pyarrow_batches docstring warns about. Uses a multi-batch
    source so datafusion's read-ahead leaves the tail un-pulled and the writer
    un-exhausted (see _multi_batch_source)."""
    con = xo.connect()
    tt = _multi_batch_source(con, "partial_src")
    target = tmp_path / "out.parquet"

    before = _placeholder_tables(con)
    reader = tt.tee(ParquetWriteThrough(path=target)).to_pyarrow_batches()
    reader.read_next_batch()  # read one batch, do not exhaust

    assert not target.exists(), "nothing should be published before exhaustion"
    assert _placeholder_tables(con) - before, "table released before full consume"


def test_cross_backend_tee_cleans_up_all_placeholders(tmp_path: Path) -> None:
    """A tee over an into_backend parent registers two placeholders (the remote
    table and the tee pass-through). Full execution must drop both, exercising
    the multi-entry created-table cleanup."""
    con = xo.connect()
    other = xo.connect()
    t = con.create_table("ib_src", pa.table({"a": [1, 2, 3, 4]}))
    target = tmp_path / "out.parquet"

    before_con = _placeholder_tables(con)
    before_other = _placeholder_tables(other)

    t.into_backend(other, "ib").tee(ParquetWriteThrough(path=target)).execute()

    assert target.exists()
    assert _placeholder_tables(con) == before_con, "source backend leaked a table"
    assert _placeholder_tables(other) == before_other, "target backend leaked a table"


def test_hashing_tag_tokenize_ignores_parent(t: Table) -> None:
    """A HashingTag tokenizes by (schema, metadata) only -- not its parent. Two
    tags with identical metadata over same-schema but different parents collapse
    to the same token; graph position is captured by the SQL instead."""
    ht1 = t.filter(t.a > 1).hashing_tag("k").op()
    ht2 = t.filter(t.a > 2).hashing_tag("k").op()

    assert isinstance(ht1, HashingTag) and isinstance(ht2, HashingTag)
    assert ht1.parent != ht2.parent
    assert ht1.schema == ht2.schema
    assert ht1.__dasher_tokenize__() == ht2.__dasher_tokenize__()


def test_hashing_tag_build_hash_uses_sql_for_position(t: Table) -> None:
    """Because the tag token drops the parent, position must still come through
    the SQL: structurally different exprs carrying the same tag must not collapse
    to one build hash, while the tag's metadata still contributes."""
    h1 = get_expr_hash(t.filter(t.a > 1).hashing_tag("k"))
    h2 = get_expr_hash(t.filter(t.a > 2).hashing_tag("k"))
    assert h1 != h2, "differing SQL must yield differing build hashes"

    assert get_expr_hash(t.hashing_tag("a")) != get_expr_hash(t.hashing_tag("b")), (
        "tag metadata must contribute to the build hash"
    )
    assert get_expr_hash(t.hashing_tag("a")) != get_expr_hash(t), (
        "attaching a HashingTag must change the build hash"
    )


def test_nested_tee_hash_is_transparent(t: Table, tmp_path: Path) -> None:
    strategy = SnapshotStrategy()
    double_tee = t.tee(ParquetWriteThrough(path=tmp_path / "t1.parquet")).tee(
        ParquetWriteThrough(path=tmp_path / "t2.parquet")
    )
    assert compute_expr_hash(double_tee, strategy=strategy) == compute_expr_hash(
        t, strategy=strategy
    )


def test_tee_affects_build_hash(t: Table, tmp_path: Path) -> None:
    teed = t.tee(ParquetWriteThrough(path=tmp_path / "out.parquet"))
    assert get_expr_hash(teed) != get_expr_hash(t)


def test_different_writers_produce_different_build_hashes(
    t: Table, tmp_path: Path
) -> None:
    teed_a = t.tee(ParquetWriteThrough(path=tmp_path / "a.parquet"))
    teed_b = t.tee(ParquetWriteThrough(path=tmp_path / "b.parquet"))
    assert get_expr_hash(teed_a) != get_expr_hash(teed_b)


def test_nested_tee_build_hash_includes_both_writers(t: Table, tmp_path: Path) -> None:
    single = t.tee(ParquetWriteThrough(path=tmp_path / "t1.parquet"))
    double = t.tee(ParquetWriteThrough(path=tmp_path / "t1.parquet")).tee(
        ParquetWriteThrough(path=tmp_path / "t2.parquet")
    )
    assert get_expr_hash(single) != get_expr_hash(double)


def test_backend_write_affects_build_hash(t: Table) -> None:
    target_con = xo.connect()
    teed = t.tee(BackendWriteThrough(target_con, table_name="bs_hash", mode="create"))
    assert get_expr_hash(teed) != get_expr_hash(t)


def test_unknown_write_type_raises_on_build_hash(t: Table) -> None:
    class CustomWriteThrough(WriteThrough):
        def write_through(self, batches, **_kw):
            yield from batches

    teed = t.tee(CustomWriteThrough())
    with pytest.raises(ValueError, match="No normalizer registered"):
        get_expr_hash(teed)


def test_tee_plus_tag_hash_is_transparent(t: Table, tmp_path: Path) -> None:
    strategy = SnapshotStrategy()
    tee_then_tag = t.tee(ParquetWriteThrough(path=tmp_path / "out.parquet")).tag("v1")
    tag_then_tee = t.tag("v1").tee(ParquetWriteThrough(path=tmp_path / "out2.parquet"))
    base_hash = compute_expr_hash(t, strategy=strategy)
    assert compute_expr_hash(tee_then_tag, strategy=strategy) == base_hash
    assert compute_expr_hash(tag_then_tee, strategy=strategy) == base_hash


def test_to_sql_with_tag_and_tee(t: Table, tmp_path: Path) -> None:
    bare_sql = xo.to_sql(t)
    assert (
        xo.to_sql(t.tee(ParquetWriteThrough(path=tmp_path / "out.parquet")).tag("v1"))
        == bare_sql
    )
    assert (
        xo.to_sql(t.tag("v1").tee(ParquetWriteThrough(path=tmp_path / "out2.parquet")))
        == bare_sql
    )


# ---- BackendWriteThrough kwargs passthrough ------------------------------------------


class _FakeKwargsBackend:
    """Fake backend that records all kwargs passed to read_record_batches."""

    name = "fake_kwargs"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def read_record_batches(self, source: Any, **kwargs: Any) -> None:
        kwargs["_batches"] = list(source)
        self.calls.append(kwargs)


def test_backend_write_extra_kwargs_reach_backend() -> None:
    backend = _FakeKwargsBackend()
    writer = BackendWriteThrough(
        backend, table_name="t", mode="create", kwargs={"custom_opt": 42}
    )
    batches = [pa.record_batch({"a": [1]})]
    list(writer.write_through(batches))
    assert len(backend.calls) == 1
    assert backend.calls[0]["custom_opt"] == 42
    assert backend.calls[0]["table_name"] == "t"


# ---- ParquetWriteThrough multi-batch -------------------------------------------------


def test_parquet_multi_batch_single_file(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    writer = ParquetWriteThrough(path=target, mode="append")
    batches = [
        pa.record_batch({"a": [1, 2]}),
        pa.record_batch({"a": [3, 4]}),
        pa.record_batch({"a": [5, 6]}),
    ]
    list(writer.write_through(batches))
    assert len(pq.read_table(str(target))) == 6


# ---- ParquetWriteThrough publish failure cleanup --------------------------------------


def test_parquet_create_link_failure_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.parquet"
    writer = ParquetWriteThrough(path=target, mode="create")
    batches = [pa.record_batch({"a": [1, 2]})]

    def failing_link(src, dst):
        raise OSError("simulated link failure")

    monkeypatch.setattr(os, "link", failing_link)
    with pytest.raises(OSError, match="simulated link failure"):
        list(writer.write_through(batches))
    assert not target.exists()
    assert not list(target.parent.glob("*.tmp"))


def test_parquet_append_rename_failure_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.parquet"
    writer = ParquetWriteThrough(path=target, mode="append")
    batches = [pa.record_batch({"a": [1, 2]})]

    original_rename = Path.rename

    def failing_rename(self, tgt):
        if str(self).endswith(".tmp"):
            raise OSError("simulated rename failure")
        return original_rename(self, tgt)

    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(OSError, match="simulated rename failure"):
        list(writer.write_through(batches))
    assert not target.exists()
    assert not list(target.parent.glob("*.tmp"))


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
    writer = BackendWriteThrough(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    with pytest.raises(RuntimeError, match="simulated"):
        list(writer.write_through(batches))
    assert len(backend.calls) == 2
    assert backend.calls[0][1] == "create"
    assert len(backend.calls[0][2]) == 1


# ---- threaded ingest path (ThreadedBackendWriteThrough) ------------------------------


class _RecordingBackend:
    """Fake mode-capable backend that drains the reader as a stream.

    Records one entry per ``read_record_batches`` call (the threaded path makes
    exactly one) holding the table name, mode, kwargs, and the batches it pulled
    — in pull order. ``on_batch(idx, batch)`` fires as each batch is drained, so
    a test can observe consumption interleaving with production.
    """

    name = "recording"

    def __init__(self, on_batch: Any = None) -> None:
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


def _first(batch: pa.RecordBatch) -> int:
    return batch.column("a")[0].as_py()


def test_threaded_single_call_all_batches() -> None:
    # mode-capable backend: threaded path makes ONE read_record_batches call
    # carrying every batch, not one call per batch.
    backend = _FakePerBatchBackend()
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    result = list(writer.write_through(batches))
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
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    list(writer.write_through(batches))
    assert len(backend.calls) == 1
    assert "mode" not in backend.calls[0]
    assert len(backend.calls[0]["_batches"]) == 2


def test_threaded_kwargs_reach_backend() -> None:
    backend = _FakeKwargsBackend()
    writer = ThreadedBackendWriteThrough(
        backend, table_name="t", mode="create", kwargs={"custom_opt": 42}
    )
    list(writer.write_through([pa.record_batch({"a": [1]})]))
    assert backend.calls[0]["custom_opt"] == 42


def test_threaded_empty_stream_no_call() -> None:
    backend = _FakePerBatchBackend()
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="create")
    assert list(writer.write_through([])) == []
    assert backend.calls == []


def test_threaded_write_thread_error_propagates() -> None:
    # an error raised inside read_record_batches is captured on the thread and
    # re-raised on the main thread after the join.
    class _Exploding(_FakePerBatchBackend):
        def read_record_batches(
            self, source: Any, table_name: str | None = None, mode: str | None = None
        ) -> None:
            list(source)  # drain the queue so the producer never blocks
            raise RuntimeError("simulated ingest failure")

    backend = _Exploding()
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})]
    with pytest.raises(RuntimeError, match="simulated ingest failure"):
        list(writer.write_through(batches))


def test_threaded_upstream_error_propagates_and_joins() -> None:
    backend = _RecordingBackend()
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="create")

    def exploding_batches():
        yield pa.record_batch({"a": [1, 2]})
        raise RuntimeError("simulated mid-stream failure")

    with pytest.raises(RuntimeError, match="simulated mid-stream failure"):
        list(writer.write_through(exploding_batches()))
    # the worker saw a truncated stream and was joined before writer returned
    assert backend.worker is not None
    assert not backend.worker.is_alive()


def test_threaded_early_stop_ends_reader_and_joins() -> None:
    # abandoning the generator mid-stream signals the reader to end short; the
    # worker drains what it has and is joined — no hang, no leaked thread.
    backend = _RecordingBackend()
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [i]}) for i in range(5)]
    gen = writer.write_through(iter(batches))
    next(gen)
    gen.close()  # GeneratorExit -> reader ends short, worker joins
    assert backend.worker is not None
    assert not backend.worker.is_alive()


def test_threaded_preserves_order_and_content() -> None:
    # passthrough is identity in order, and the writer pulls in the same order.
    backend = _RecordingBackend()
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [i]}) for i in range(10)]
    out = list(writer.write_through(iter(batches)))
    assert [_first(b) for b in out] == list(range(10))
    assert [_first(b) for b in backend.calls[0]["batches"]] == list(range(10))


def test_threaded_forwards_append_mode() -> None:
    backend = _RecordingBackend()
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="append")
    list(writer.write_through([pa.record_batch({"a": [1]})]))
    assert backend.calls[0]["mode"] == "append"


def test_threaded_streams_without_buffering() -> None:
    # proves concurrency: the writer consumes batch 0 while the producer is still
    # blocked before yielding batch 1.  A buffer-everything-then-ingest path
    # would never set the event during production and time out.
    received_first = threading.Event()

    def on_batch(idx: int, batch) -> None:
        if idx == 0:
            received_first.set()

    backend = _RecordingBackend(on_batch=on_batch)
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="create")

    def producer():
        yield pa.record_batch({"a": [0]})
        assert received_first.wait(timeout=5), "writer did not consume batch 0 early"
        yield pa.record_batch({"a": [1]})

    out = list(writer.write_through(producer()))
    assert [_first(b) for b in out] == [0, 1]
    assert len(backend.calls[0]["batches"]) == 2


def test_threaded_joins_worker_before_return() -> None:
    backend = _RecordingBackend()
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="create")
    list(writer.write_through([pa.record_batch({"a": [i]}) for i in range(3)]))
    # the single ingest completed (one recorded call) and its worker is dead
    assert len(backend.calls) == 1
    assert backend.worker is not None
    assert not backend.worker.is_alive()


def test_threaded_write_is_reusable() -> None:
    # frozen, no shared mutable state in writer(): the same instance runs twice.
    backend = _RecordingBackend()
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="create")
    list(writer.write_through([pa.record_batch({"a": [1]})]))
    list(writer.write_through([pa.record_batch({"a": [2]})]))
    assert len(backend.calls) == 2
    assert _first(backend.calls[0]["batches"][0]) == 1
    assert _first(backend.calls[1]["batches"][0]) == 2


def test_threaded_unbounded_queue_no_deadlock_when_write_lags() -> None:
    # the writer refuses to pull until released; the unbounded queue lets the
    # producer push and yield all 50 batches anyway (no backpressure, no hang).
    release = threading.Event()

    class _LaggyBackend(_RecordingBackend):
        def read_record_batches(self, source, table_name=None, mode=None, **kwargs):
            assert release.wait(timeout=5)
            super().read_record_batches(
                source, table_name=table_name, mode=mode, **kwargs
            )

    backend = _LaggyBackend()
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="create")
    batches = [pa.record_batch({"a": [i]}) for i in range(50)]

    out = []
    for i, batch in enumerate(writer.write_through(iter(batches))):
        out.append(_first(batch))
        if i == len(batches) - 1:
            release.set()  # unblock the writer before the join on the next pull
    assert out == list(range(50))
    assert [_first(b) for b in backend.calls[0]["batches"]] == list(range(50))


def test_threaded_stops_feeding_after_write_error() -> None:
    # the writer dies after one batch; the `if error: break` guard stops the
    # producer well short of its full run instead of pushing into a dead queue.
    write_failed = threading.Event()

    class _FailFast(_RecordingBackend):
        def read_record_batches(self, source, table_name=None, mode=None, **kwargs):
            self.worker = threading.current_thread()
            it = iter(source)
            next(it)
            write_failed.set()
            raise RuntimeError("simulated ingest failure")

    backend = _FailFast()
    writer = ThreadedBackendWriteThrough(backend, table_name="t", mode="create")
    produced = []

    def producer():
        for i in range(100):
            yield pa.record_batch({"a": [i]})
            produced.append(i)
            if i == 0:
                assert write_failed.wait(timeout=5)

    with pytest.raises(RuntimeError, match="simulated ingest failure"):
        list(writer.write_through(producer()))
    assert len(produced) < 100
    assert backend.worker is not None
    assert not backend.worker.is_alive()


def test_threaded_bounded_writes_all_batches() -> None:
    # a bounded queue still streams every batch through, in order, to a single
    # ingest call — backpressure caps memory, it does not drop data.
    backend = _RecordingBackend()
    writer = ThreadedBackendWriteThrough(
        backend, table_name="t", mode="create", maxsize=2
    )
    batches = [pa.record_batch({"a": [i]}) for i in range(20)]
    out = list(writer.write_through(iter(batches)))
    assert [_first(b) for b in out] == list(range(20))
    assert [_first(b) for b in backend.calls[0]["batches"]] == list(range(20))


def test_threaded_bounded_no_deadlock_when_write_dies() -> None:
    # the discard-on-death invariant: with a full bounded queue the producer
    # blocks on put(); when the write thread dies it keeps draining (discarding)
    # so the producer unblocks, the error surfaces, and downstream is cut off at
    # the failure point instead of receiving every batch.
    class _FailAfterFirst(_RecordingBackend):
        def read_record_batches(self, source, table_name=None, mode=None, **kwargs):
            self.worker = threading.current_thread()
            it = iter(source)
            next(it)  # pull exactly one batch, then stop draining
            raise RuntimeError("simulated ingest failure")

    backend = _FailAfterFirst()
    writer = ThreadedBackendWriteThrough(
        backend, table_name="t", mode="create", maxsize=1
    )
    yielded: list[int] = []
    result: list[BaseException | None] = []

    def run() -> None:
        try:
            for batch in writer.write_through(
                iter([pa.record_batch({"a": [i]}) for i in range(100)])
            ):
                yielded.append(_first(batch))
            result.append(None)
        except BaseException as exc:  # noqa: BLE001
            result.append(exc)

    th = threading.Thread(target=run)
    th.start()
    th.join(timeout=10)
    assert not th.is_alive(), "bounded write_through deadlocked on write death"
    assert isinstance(result[0], RuntimeError)
    assert len(yielded) < 100  # downstream cut off at the failure point


def test_threaded_bounded_drain_on_close_writes_all() -> None:
    # drain semantics over a *bounded* threaded write: stopping downstream early
    # and draining (DrainingIterator.close) must finish the full write without
    # wedging on the bounded queue — the drain thread keeps consuming yields so
    # the producer never blocks, and every batch reaches the single ingest.
    backend = _RecordingBackend()
    writer = ThreadedBackendWriteThrough(
        backend, table_name="t", mode="create", maxsize=2
    )
    batches = [pa.record_batch({"a": [i]}) for i in range(10)]
    gen = writer.write_through(iter(batches))
    it = DrainingIterator(gen)
    next(it)
    next(it)
    it.close()  # downstream stops early; drain finishes the write
    it.join(timeout=5)
    assert it.exhausted
    assert [_first(b) for b in backend.calls[0]["batches"]] == list(range(10))


def test_threaded_maxsize_is_identity_neutral() -> None:
    con = xo.connect()
    a = ThreadedBackendWriteThrough(con, table_name="x", mode="create")
    b = ThreadedBackendWriteThrough(con, table_name="x", mode="create", maxsize=4)
    assert a.__dasher_tokenize__() == b.__dasher_tokenize__()
    assert a == b
    assert hash(a) == hash(b)


def test_threaded_creates_table_real_backend(t: Table) -> None:
    target_con = xo.connect()
    writer = ThreadedBackendWriteThrough(target_con, table_name="th_tgt", mode="create")
    out = t.tee(writer).execute()
    assert len(out) == len(t.execute())
    assert len(target_con.table("th_tgt").execute()) == len(t.execute())


def test_threaded_bounded_terminal_write_limit_zero(t: Table) -> None:
    # terminal-write pattern: a bounded threaded writer behind .limit(0) returns
    # nothing downstream yet completes the full write via the post-execution
    # drain (drain=True default).  The bounded queue only paces producer<->write
    # thread here, so it must not wedge with no real downstream consuming.
    target_con = xo.connect()
    writer = ThreadedBackendWriteThrough(
        target_con, table_name="term_tgt", mode="create", maxsize=2
    )
    out = t.tee(writer).limit(0).execute()
    assert len(out) == 0
    assert len(target_con.table("term_tgt").execute()) == len(t.execute())


def test_threaded_bounded_terminal_write_no_drain_skips_write(t: Table) -> None:
    # drain=False + .limit(0): the generator is never driven, so the write never
    # fires — bounded mode does not change the early-stop-aborts semantics.
    target_con = xo.connect()
    writer = ThreadedBackendWriteThrough(
        target_con, table_name="term_no_drain", mode="create", maxsize=2
    )
    t.tee(writer, drain=False).limit(0).execute()
    with pytest.raises((ValueError, Exception), match="not found|term_no_drain"):
        target_con.table("term_no_drain")


# ---- DrainingIterator --------------------------------------------------------


def test_draining_iterator_passthrough_on_full_consumption(tmp_path: Path) -> None:
    target = tmp_path / "pass.parquet"
    batches = [pa.record_batch({"a": [i]}) for i in range(5)]
    writer = ParquetWriteThrough(path=target, mode="append")
    gen = writer.write_through(iter(batches))
    it = DrainingIterator(gen)
    result = list(it)
    assert len(result) == 5
    assert it.exhausted


def test_draining_iterator_drains_on_close(tmp_path: Path) -> None:
    target = tmp_path / "drain.parquet"
    writer = ParquetWriteThrough(path=target, mode="append")
    batches = [pa.record_batch({"a": [i]}) for i in range(10)]
    gen = writer.write_through(iter(batches))
    it = DrainingIterator(gen)
    next(it)
    next(it)
    it.close()
    it.join(timeout=5)
    assert target.exists()
    assert len(pq.read_table(str(target))) == 10


def test_draining_iterator_close_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "drain_idem.parquet"
    writer = ParquetWriteThrough(path=target, mode="append")
    batches = [pa.record_batch({"a": [i]}) for i in range(3)]
    gen = writer.write_through(iter(batches))
    it = DrainingIterator(gen)
    next(it)
    it.close()
    it.close()
    it.join(timeout=5)
    assert len(pq.read_table(str(target))) == 3


def test_draining_iterator_noop_when_exhausted() -> None:
    batches = [pa.record_batch({"a": [1]})]

    class PassthroughWriteThrough(WriteThrough):
        def write_through(self, batches, **_kw):
            yield from batches

    gen = PassthroughWriteThrough().write_through(iter(batches))
    it = DrainingIterator(gen)
    list(it)
    assert it.exhausted
    it.close()
    it.join(timeout=5)


def test_draining_iterator_join_surfaces_error() -> None:
    def exploding_gen():
        yield pa.record_batch({"a": [1]})
        raise RuntimeError("simulated writer failure")

    it = DrainingIterator(exploding_gen())
    next(it)
    it.close()
    with pytest.raises(RuntimeError, match="simulated writer failure"):
        it.join(timeout=5)


def _flatten_exceptions(exc: BaseException) -> list[BaseException]:
    out = [exc]
    for sub in getattr(exc, "exceptions", ()):  # BaseExceptionGroup (3.11+)
        out.extend(_flatten_exceptions(sub))
    if exc.__cause__ is not None:  # chained fallback (3.10)
        out.extend(_flatten_exceptions(exc.__cause__))
    return out


def test_run_cleanup_aggregates_drain_and_drop_failures() -> None:
    # Both cleanup steps run even when the first raises: a drain-join failure and
    # a table-drop failure must surface together as a group, neither hiding the
    # other (this is what the execute / to_pyarrow_batches paths rely on).
    class _FailingDrain:
        def close(self) -> None: ...

        def join(self, timeout: float | None = None) -> None:
            raise RuntimeError("drain join boom")

    class _FailingCon:
        def drop_table(self, name: str, force: bool = False) -> None:
            raise RuntimeError("drop_table boom")

        def drop_view(self, name: str) -> None:
            raise RuntimeError("drop_view boom")

    raised: BaseException | None = None
    try:
        _run_cleanup([_FailingDrain()], {"leaked_tbl": _FailingCon()})
    except BaseException as exc:  # noqa: BLE001
        raised = exc

    assert raised is not None, "cleanup must surface the failures"
    messages = [str(e) for e in _flatten_exceptions(raised)]
    assert any("drain join boom" in m for m in messages)
    assert any("drop_view boom" in m for m in messages)


def test_draining_iterator_join_before_close_raises() -> None:
    it = DrainingIterator(iter([pa.record_batch({"a": [1]})]))
    next(it)
    with pytest.raises(RuntimeError, match="join.*before close"):
        it.join()


# ---- drain=True via .tee() --------------------------------------------------


@pytest.mark.skipif(
    _DATAFUSION_HAS_EOF_PROBE,
    reason="#2105: on 0.2.9 the terminal EOF probe exhausts the single batch "
    "regardless of whether drain fired, so the assertion passes for the wrong "
    "reason -- skip rather than report a vacuous pass. Meaningful only on 0.2.7.",
)
def test_tee_drain_writes_full_on_early_stop(t: Table, tmp_path: Path) -> None:
    # TODO(#2105): still single-batch -- a multi-batch drain=True races
    # datafusion's in-flight pull on 0.2.9, so it can't be hardened like the
    # early-stop tests above. The skipif guards the 0.2.9 vacuous-pass hazard.
    target = tmp_path / "drain_tee.parquet"
    out = t.tee(ParquetWriteThrough(path=target), drain=True).limit(2).execute()
    assert len(out) == 2
    assert target.exists()
    written = pq.read_table(str(target))
    assert len(written) == 4


def test_tee_drain_writes_full_on_early_stop_multi_batch(tmp_path: Path) -> None:
    # Issue #2105 integration repro: multi-batch drain=True + early stop must
    # still write the full parent. Only opens the race on datafusion >=0.2.9
    # (bounded read-ahead); on the pinned 0.2.7 it passes even unfixed, so a green
    # run here is not proof -- test_draining_iterator_serializes_concurrent_advance
    # is the deterministic guard. See ADR-0014.
    con = xo.connect()
    tt = _multi_batch_source(con, "drain_multi_src")
    target = tmp_path / "drain_tee_multi.parquet"
    out = tt.tee(ParquetWriteThrough(path=target), drain=True).limit(2).execute()
    assert len(out) == 2
    assert target.exists()
    written = pq.read_table(str(target))
    assert len(written) == _MULTI_BATCH_COUNT


def test_tee_drain_false_does_not_drain(tmp_path: Path) -> None:
    # drain=False: an early downstream stop must NOT finish the side write. A
    # multi-batch source is required -- datafusion's read-ahead stops short of
    # exhausting the parent under a LIMIT, so the writer never publishes. See
    # _multi_batch_source for why a single-batch source cannot express this.
    con = xo.connect()
    tt = _multi_batch_source(con, "no_drain_src")
    target = tmp_path / "no_drain.parquet"
    tt.tee(ParquetWriteThrough(path=target), drain=False).limit(2).execute()
    assert not target.exists()

    # Positive control: same source fully consumed must publish, else the assertion above passes for the wrong reason.
    full_src = _multi_batch_source(con, "no_drain_full_src")
    full_target = tmp_path / "full.parquet"
    full_src.tee(ParquetWriteThrough(path=full_target), drain=False).execute()
    assert full_target.exists(), "full consumption must publish despite drain=False"
    assert len(pq.read_table(str(full_target))) == _MULTI_BATCH_COUNT


def test_drain_build_hash_same(t: Table, tmp_path: Path) -> None:
    writer = ParquetWriteThrough(path=tmp_path / "out.parquet")
    no_drain = t.tee(writer, drain=False)
    with_drain = t.tee(writer, drain=True)
    assert get_expr_hash(no_drain) == get_expr_hash(with_drain)


# ---- write-primary transport (WritePrimaryWriteThrough) ----------------------


class _RecordingWriteThrough(WriteThrough):
    """Inner write-through that records each batch it writes, in pull order."""

    def __init__(self) -> None:
        self.written: list[pa.RecordBatch] = []

    def __dasher_tokenize__(self) -> tuple:
        return ("_RecordingWriteThrough",)

    def write_through(self, batches: Any) -> Any:
        for batch in batches:
            self.written.append(batch)
            yield batch


class _ExplodingWriteThrough(WriteThrough):
    def __dasher_tokenize__(self) -> tuple:
        return ("_ExplodingWriteThrough",)

    def write_through(self, batches: Any) -> Any:
        for _ in batches:
            raise RuntimeError("boom")
            yield _  # pragma: no cover - unreachable, makes this a generator


def test_write_primary_full_consumption() -> None:
    inner = _RecordingWriteThrough()
    writer = WritePrimaryWriteThrough(inner)
    batches = [pa.record_batch({"a": [i]}) for i in range(3)]
    out = list(writer.write_through(batches))
    assert len(out) == 3  # passthrough preserved
    assert len(inner.written) == 3


def test_write_primary_completes_on_early_stop() -> None:
    # The write owns the pull loop, so closing the generator after two batches
    # still drives the inner write to completion — drain-always by construction.
    inner = _RecordingWriteThrough()
    writer = WritePrimaryWriteThrough(inner)
    batches = [pa.record_batch({"a": [i]}) for i in range(5)]
    gen = writer.write_through(batches)
    got = [next(gen), next(gen)]
    gen.close()
    assert len(got) == 2
    assert len(inner.written) == 5


def test_write_primary_bounded_completes_on_early_stop() -> None:
    # bounded queue must not deadlock when downstream stops: the close path
    # keeps draining so the blocked write thread can finish.
    inner = _RecordingWriteThrough()
    writer = WritePrimaryWriteThrough(inner, maxsize=1)
    batches = [pa.record_batch({"a": [i]}) for i in range(5)]
    gen = writer.write_through(batches)
    next(gen)
    gen.close()
    assert len(inner.written) == 5


def test_write_primary_error_propagates() -> None:
    writer = WritePrimaryWriteThrough(_ExplodingWriteThrough())
    with pytest.raises(RuntimeError, match="boom"):
        list(writer.write_through([pa.record_batch({"a": [1]})]))


def test_write_primary_identity_delegates_to_inner(tmp_path: Path) -> None:
    inner = ParquetWriteThrough(path=tmp_path / "x.parquet")
    writer = WritePrimaryWriteThrough(inner)
    assert writer.__dasher_tokenize__() == inner.__dasher_tokenize__()


def test_write_primary_build_hash_same(t: Table, tmp_path: Path) -> None:
    inner = ParquetWriteThrough(path=tmp_path / "out.parquet")
    plain = t.tee(inner)
    wrapped = t.tee(WritePrimaryWriteThrough(inner))
    assert get_expr_hash(plain) == get_expr_hash(wrapped)


def test_write_primary_via_tee_parquet(t: Table, tmp_path: Path) -> None:
    # end-to-end: a write-primary Parquet write (no dedicated threaded-parquet
    # class previously existed) completes through .tee().execute().
    target = tmp_path / "wp.parquet"
    out = t.tee(WritePrimaryWriteThrough(ParquetWriteThrough(path=target))).execute()
    assert target.exists()
    written = pq.read_table(str(target))
    assert len(written) == len(out)
