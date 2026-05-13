import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.caching.strategy import ModificationTimeStrategy, SnapshotStrategy
from xorq.expr.relations import RemoteTable


def test_put_get_drop(tmp_path, parquet_dir):
    astronauts_path = parquet_dir.joinpath("astronauts.parquet")

    con = xo.datafusion.connect()
    t = con.read_parquet(astronauts_path, table_name="astronauts")

    cache = ParquetCache.from_kwargs(relative_path=tmp_path, source=con)
    put_node = cache.put(t, t.op())
    assert put_node is not None

    get_node = cache.get(t)
    assert get_node is not None

    cache.drop(t)
    with pytest.raises(KeyError):
        cache.get(t)


def test_default_connection(tmp_path, parquet_dir):
    batting_path = parquet_dir.joinpath("astronauts.parquet")

    con = xo.connect()
    t = con.read_parquet(batting_path, table_name="astronauts")

    # if we do cross source caching, then we get a random name and cache.calc_key result isn't stable
    cache = ParquetCache.from_kwargs(relative_path=tmp_path)
    cache.put(t, t.op())

    get_node = cache.get(t)
    assert get_node is not None
    assert get_node.source.name == con.name
    assert get_node.to_expr().execute is not None


def test_snapshot_strategy_key_is_path_identity(tmp_path):
    # Regression test: SnapshotStrategy must key on path identity only, not on
    # mtime/size/inode. Without this, snapshot keys flip whenever the underlying
    # file is rewritten, defeating the purpose of the strategy.
    #
    # SnapshotStrategy.cached_normalize_read is @functools.cache'd on op identity,
    # which would mask the bug in a single process. Clear it between runs to
    # simulate a fresh process — that is where the bug actually bit users.
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3]}), path)

    con = xo.connect()
    snapshot = SnapshotStrategy()
    mtime = ModificationTimeStrategy()

    expr = xo.deferred_read_parquet(path, con=con, table_name="t")
    snapshot_before = snapshot.calc_key(expr)
    mtime_before = mtime.calc_key(expr)

    pq.write_table(pa.table({"a": list(range(50))}), path)
    expr = xo.deferred_read_parquet(path, con=con, table_name="t")

    assert snapshot.calc_key(expr) == snapshot_before
    # Sanity: ModificationTimeStrategy *should* notice the change. If it doesn't,
    # the test setup isn't actually exercising stat sensitivity and the snapshot
    # invariant above is vacuous.
    assert mtime.calc_key(expr) != mtime_before


def test_snapshot_strategy_calc_key_with_hashing_tag_over_remote_table():
    t = xo.memtable({"a": [1, 2, 3]})
    con = t._find_backend()
    rt = RemoteTable.from_expr(con, t).to_expr()
    tagged = rt.hashing_tag("my-source", entry_name="test-source", kind="source")

    strategy = SnapshotStrategy()
    key = strategy.calc_key(tagged)
    assert key.startswith(f"{strategy.key_prefix}snapshot-")


# test_rename_remote_table_outside_normalization_context_raises removed: the
# guarded helper (``_rename_remote_table`` as a module-level function with a
# context-var assertion) was specific to the dask-monkeypatching design.
# SnapshotStrategy now uses a per-call local hasher built via ``snapshot_hasher``
# (no shared mutable state, no global cache to poison), so the failure mode the
# guard protected against can no longer occur.
