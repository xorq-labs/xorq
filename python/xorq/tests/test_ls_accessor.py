from pathlib import Path

import pytest

import xorq.api as xo
import xorq.vendor.ibis.expr.operations.relations as rel
from xorq.backends.let import Backend
from xorq.caching import (
    ParquetStorage,
)


@pytest.fixture
def cached_two(con, pg, tmp_path):
    parquet_storage = ParquetStorage(source=con, relative_path=tmp_path)
    return (
        pg.table("batting")[lambda t: t.yearID > 2014]
        .cache()[lambda t: t.stint == 1]
        .cache(storage=parquet_storage)
    )


@pytest.fixture
def duck_batting_raw(batting_df):
    return xo.duckdb.connect().create_table(
        "batting_df",
        batting_df,
    )


@pytest.fixture
def duck_batting(duck_batting_raw):
    return duck_batting_raw


@pytest.fixture
def cached_two_joined(cached_two, duck_batting):
    return cached_two.join(
        duck_batting,
        duck_batting.columns,
    )


def test_ls_exists(batting):
    attr = getattr(batting, "ls", None)
    assert attr is not None


def test_ls_cache_nodes(cached_two):
    assert len(cached_two.ls.cached_nodes) == 2


def test_storage(cached_two):
    assert cached_two.ls.is_cached
    assert cached_two.ls.storage


def test_storages(cached_two):
    assert cached_two.ls.is_cached
    assert len(cached_two.ls.storages) == len(cached_two.ls.cached_nodes)


def test_backends(duck_batting_raw, cached_two, cached_two_joined):
    assert len(duck_batting_raw.ls.backends) == 1
    assert len(cached_two.ls.backends) == 2
    assert len(cached_two_joined.ls.backends) == 3


def test_is_multiengine(duck_batting_raw, cached_two, cached_two_joined):
    assert not duck_batting_raw.ls.is_multiengine
    assert cached_two.ls.is_multiengine
    assert cached_two_joined.ls.is_multiengine


def test_dts(cached_two, cached_two_joined):
    dts = cached_two.ls.dts
    assert len(dts) == 1
    assert not any(dt.source.name == Backend.name for dt in dts)

    dts = cached_two_joined.ls.dts
    assert len(dts) == 2
    assert not any(dt.source.name == Backend.name for dt in dts)


def test_is_cached(cached_two, cached_two_joined):
    assert cached_two.ls.is_cached
    assert not cached_two_joined.ls.is_cached


def test_has_cached(cached_two, cached_two_joined):
    els = cached_two.ls
    assert els.is_cached and els.has_cached
    els = cached_two_joined.ls
    assert not els.is_cached and els.has_cached


def test_uncached(cached_two):
    assert cached_two.ls.has_cached and not cached_two.ls.uncached.ls.has_cached


def test_uncached_one(cached_two):
    assert (
        cached_two.ls.is_cached
        and not cached_two.ls.uncached_one.ls.is_cached
        and cached_two.ls.uncached_one.ls.has_cached
    )


def test_exists(cached_two):
    storage = cached_two.ls.storage

    assert not cached_two.ls.exists()
    assert not tuple(storage.path.iterdir())

    xo.execute(cached_two)
    assert cached_two.ls.exists()
    assert len(tuple(storage.path.iterdir())) == 1

    (path,) = storage.path.iterdir()
    path.unlink()
    assert not cached_two.ls.exists()


def test_cache_properties(parquet_dir, tmp_path):
    con = xo.connect()
    storage = ParquetStorage(
        source=con, relative_path="./tmp-cache", base_path=tmp_path
    )
    t = xo.deferred_read_parquet(
        parquet_dir.joinpath("batting.parquet"),
        con=con,
        table_name="t",
    )
    expr = (
        t.cache(storage)
        .mutate(a=1)
        .cache(storage)
        .mutate(b=2)
        .cache(storage)
        .mutate(b=3)
        .cache()
    )
    assert not any(p.exists() for p in expr.ls.get_cache_paths())
    expr.execute()
    assert all(p.exists() for p in expr.ls.get_cache_paths())
    (ssn, psn0, psn1, psn2) = expr.ls.cached_nodes
    (ss, ps, *rest) = expr.ls.storages
    assert (ps,) == tuple(set(rest))
    p = storage.cache.storage.path

    # downstream is still cached even if upstream caches don't exist
    (*to_remove, last) = sorted(
        storage.cache.storage.path.iterdir(), key=lambda p: p.stat().st_ctime
    )
    for p in to_remove:
        p.unlink()
    assert all(
        (
            psn0.to_expr().ls.exists(),
            not psn1.to_expr().ls.exists(),
            not psn2.to_expr().ls.exists(),
        )
    )

    # cached_dt differs based on type of Storage
    assert isinstance(psn0.to_expr().ls.cached_dt, xo.expr.relations.Read)
    assert isinstance(ssn.to_expr().ls.cached_dt, rel.DatabaseTable)
    assert all(
        isinstance(psn.to_expr().ls.cache_path, Path) for psn in (psn0, psn1, psn2)
    )
    assert psn0.to_expr().ls.cache_path.exists()
    assert not psn1.to_expr().ls.cache_path.exists()
    assert not psn2.to_expr().ls.cache_path.exists()
    assert ssn.to_expr().ls.cache_path is None
