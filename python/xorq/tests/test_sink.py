import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.caching.strategy import SnapshotStrategy
from xorq.common.utils.node_utils import compute_expr_hash
from xorq.sinking import ParquetSink


TABLE = {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]}


@pytest.fixture
def t():
    con = xo.connect()
    return con.register(xo.memtable(TABLE), table_name="t0")


def _connect(name):
    try:
        return {
            "datafusion": xo.connect,
            "duckdb": xo.duckdb.connect,
            "pandas": xo.pandas.connect,
        }[name]()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"backend {name} unavailable: {exc}")


# duckdb is excluded: its single connection is not re-entrant, so the streaming
# tee deadlocks when it pulls the parent reader while the same connection serves
# the outer query. The Phase 1 streaming tee targets engines that allow
# concurrent reader pulls (datafusion). See ADR-0014.
@pytest.fixture(params=["datafusion", "pandas"])
def backend_table(request):
    con = _connect(request.param)
    return con.create_table("sink_src", pa.table(TABLE))


def _files(path):
    return sorted(path.glob("*.parquet"))


def test_sink_is_passthrough(t, tmp_path):
    expr = t.sink(ParquetSink(path=tmp_path / "tgt"))
    assert expr.schema() == t.schema()
    assert expr.execute().equals(t.execute())


def test_sink_writes_what_flows(t, tmp_path):
    target = tmp_path / "tgt"
    t.sink(ParquetSink(path=target)).execute()
    written = xo.connect().read_parquet(str(target / "*.parquet")).execute()
    assert len(written) == len(t.execute())


def test_sink_hash_is_transparent(t, tmp_path):
    strategy = SnapshotStrategy()
    sinked = t.sink(ParquetSink(path=tmp_path / "tgt"))
    assert compute_expr_hash(sinked, strategy=strategy) == compute_expr_hash(
        t, strategy=strategy
    )


def test_cache_hit_does_not_write(t, tmp_path):
    target = tmp_path / "tgt"
    cached = t.sink(ParquetSink(path=target)).cache()
    cached.execute()  # miss: writes
    assert len(_files(target)) == 1
    cached.execute()  # hit: must not write again
    assert len(_files(target)) == 1


def test_append_adds_a_file_per_run(t, tmp_path):
    target = tmp_path / "tgt"
    t.sink(ParquetSink(path=target, mode="append")).execute()
    t.sink(ParquetSink(path=target, mode="append")).execute()
    assert len(_files(target)) == 2


def test_create_fails_if_target_exists(t, tmp_path):
    target = tmp_path / "tgt"
    t.sink(ParquetSink(path=target, mode="create")).execute()
    assert len(_files(target)) == 1
    # raised mid-pull, the engine wraps FileExistsError (Arrow C-stream boundary)
    with pytest.raises(Exception, match="already has parquet files"):
        t.sink(ParquetSink(path=target, mode="create")).execute()
    # the failed run published nothing: still exactly one file, no stray temp
    assert len(_files(target)) == 1
    assert list(target.glob("*.tmp")) == []


def test_create_consumer_raises_fileexists_directly(tmp_path):
    # used directly (no engine), the plain FileExistsError type surfaces
    target = tmp_path / "tgt"
    s1 = ParquetSink(path=target, mode="create")
    s1.read(pa.record_batch({"a": [1]}))
    s1.commit()
    assert len(_files(target)) == 1
    with pytest.raises(FileExistsError):
        ParquetSink(path=target, mode="create").read(pa.record_batch({"a": [1]}))


def test_invalid_mode_raises(tmp_path):
    with pytest.raises(ValueError):
        ParquetSink(path=tmp_path, mode="merge")


def test_consumer_commit_publishes(tmp_path):
    target = tmp_path / "tgt"
    sink = ParquetSink(path=target, mode="append")
    sink.read(pa.record_batch({"a": [1, 2]}))
    sink.commit()
    assert len(list(target.glob("*.parquet"))) == 1


def test_consumer_abort_publishes_nothing(tmp_path):
    target = tmp_path / "tgt"
    sink = ParquetSink(path=target, mode="append")
    sink.read(pa.record_batch({"a": [1]}))
    sink.abort()
    # nothing published, no stray temp file
    assert list(target.glob("*")) == []


def test_consumer_commit_without_read_is_noop(tmp_path):
    target = tmp_path / "tgt"
    sink = ParquetSink(path=target, mode="append")
    sink.commit()  # nothing read
    assert not target.exists() or list(target.glob("*")) == []


# ---- cross-backend ----------------------------------------------------------


def test_sink_across_backends(backend_table, tmp_path):
    target = tmp_path / "tgt"
    out = backend_table.sink(ParquetSink(path=target, mode="append")).execute()
    assert len(out) == 4
    assert len(_files(target)) == 1
    # second append run adds a second file
    backend_table.sink(ParquetSink(path=target, mode="append")).execute()
    assert len(_files(target)) == 2


# ---- mixed ops --------------------------------------------------------------


def test_sink_after_deferred_read(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3, 4]}), str(src))
    target = tmp_path / "tgt"
    expr = xo.deferred_read_parquet(path=src, con=xo.connect(), table_name="dr").sink(
        ParquetSink(path=target)
    )
    assert len(expr.execute()) == 4
    assert len(_files(target)) == 1


def test_sink_after_into_backend(tmp_path):
    con = xo.connect()
    other = xo.connect()  # second datafusion; duckdb would deadlock (see fixture)
    t = con.create_table("ib_src", pa.table({"a": [1, 2, 3, 4]}))
    target = tmp_path / "tgt"
    t.into_backend(other, "ib").sink(ParquetSink(path=target)).execute()
    assert len(_files(target)) == 1


@pytest.mark.skip(
    reason="duckdb single connection is not re-entrant: the streaming tee pulls "
    "the parent reader while the same connection serves the outer query, which "
    "deadlocks. Phase 1 targets engines with concurrent reader pulls (datafusion)."
)
def test_sink_duckdb_streaming_deadlocks(tmp_path):
    con = xo.duckdb.connect()
    t = con.create_table("dd_src", pa.table({"a": [1, 2, 3, 4]}))
    t.sink(ParquetSink(path=tmp_path / "tgt")).execute()


def test_sink_with_cache_upstream(tmp_path):
    con = xo.connect()
    t = con.create_table("c_src", pa.table({"a": [1, 2, 3, 4]}))
    target = tmp_path / "tgt"
    t.cache().sink(ParquetSink(path=target)).execute()
    assert len(_files(target)) == 1


def test_sink_in_middle_writes_full_parent(t, tmp_path):
    target = tmp_path / "tgt"
    # a downstream filter reduces the result, but the tee wrote the full parent
    out = t.sink(ParquetSink(path=target)).filter(xo._.a > 2).execute()
    assert len(out) == 2
    assert len(pq.read_table(str(next(target.glob("*.parquet"))))) == 4


def test_chained_sinks_fan_out(t, tmp_path):
    d1, d2 = tmp_path / "t1", tmp_path / "t2"
    t.sink(ParquetSink(path=d1)).sink(ParquetSink(path=d2)).execute()
    assert len(_files(d1)) == 1
    assert len(_files(d2)) == 1
