import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.caching.strategy import SnapshotStrategy
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


def test_snapshot_strategy_calc_key_with_hashing_tag_over_remote_table():
    t = xo.memtable({"a": [1, 2, 3]})
    con = t._find_backend()
    rt = RemoteTable.from_expr(con, t).to_expr()
    tagged = rt.hashing_tag("my-source", entry_name="test-source", kind="source")

    strategy = SnapshotStrategy()
    key = strategy.calc_key(tagged)
    assert key.startswith(f"{strategy.key_prefix}snapshot-")


def test_snapshot_strategy_calc_key_invariant_under_mtime(tmp_path):
    """SnapshotStrategy keys must depend on path identity, not file modification stats."""
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3]}), path)

    con = xo.datafusion.connect()
    t = con.read_parquet(path, table_name="t")

    strategy = SnapshotStrategy()
    key1 = strategy.calc_key(t)

    # bump the file's mtime/atime; content and path are unchanged
    stat = path.stat()
    os.utime(path, (stat.st_atime + 10_000, stat.st_mtime + 10_000))
    SnapshotStrategy.cached_normalize_read.cache_clear()

    key2 = strategy.calc_key(t)
    assert key1 == key2
