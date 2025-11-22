import time

import pytest

import xorq.api as xo
from xorq.caching import (
    ParquetSnapshotStorage,
    SourceSnapshotStorage,
    SourceStorage,
)
from xorq.common.utils.postgres_utils import (
    do_analyze,
    get_postgres_n_scans,
)


@pytest.mark.parametrize(
    "name",
    (
        "diamonds",
        "astronauts",
    ),
)
def test_source_caching(name, pg, parquet_dir):
    con = xo.connect()
    example = xo.deferred_read_parquet(parquet_dir / f"{name}.parquet", con)
    expr = example.cache(storage=SourceStorage(pg))
    assert not expr.ls.exists()
    actual = expr.execute()
    expected = example.execute()
    cached = pg.table(expr.ls.get_key()).execute()
    assert actual.equals(expected)
    assert actual.equals(cached)
    assert expr.ls.exists()
    pg.drop_table(expr.ls.get_key())
    assert not expr.ls.exists()


def test_postgres_cache_invalidation(pg, con):
    def modify_postgres_table(dt):
        (con, name) = (dt.source, dt.name)
        statement = f"""
        INSERT INTO "{name}"
        DEFAULT VALUES
        """
        con.raw_sql(statement)

    def assert_n_scans_changes(dt, n_scans_before):
        do_analyze(dt.source, dt.name)
        for _ in range(10):  # noqa: F402
            # give postgres some time to update its tables
            time.sleep(0.1)
            n_scans_after = get_postgres_n_scans(dt)
            if n_scans_before != n_scans_after:
                return n_scans_after
        else:
            raise

    (from_name, to_name) = ("batting", "batting_to_modify")
    if to_name in pg.tables:
        pg.drop_table(to_name)
    pg_t = pg.create_table(to_name, obj=pg.table(from_name).limit(1000))
    expr_cached = (
        pg_t.group_by("playerID")
        .size()
        .order_by("playerID")
        .cache(storage=SourceStorage(source=con))
    )
    dt = pg_t.op()
    (storage, uncached) = (expr_cached.ls.storage, expr_cached.ls.uncached_one)

    # assert initial state
    assert not storage.exists(uncached)
    n_scans_before = get_postgres_n_scans(dt)
    assert n_scans_before == 0

    # assert first execution state
    expr_cached.execute()
    n_scans_after = assert_n_scans_changes(dt, n_scans_before)
    # should we test that SourceStorage.get is called?
    assert n_scans_after == 1
    assert storage.exists(uncached)

    # assert no change after re-execution of cached expr
    expr_cached.execute()
    assert n_scans_after == get_postgres_n_scans(dt)

    # assert cache invalidation happens
    modify_postgres_table(dt)
    expr_cached.execute()
    assert_n_scans_changes(dt, n_scans_after)


def test_postgres_snapshot(pg, con):
    def modify_postgres_table(dt):
        (con, name) = (dt.source, dt.name)
        statement = f"""
        INSERT INTO "{name}"
        DEFAULT VALUES
        """
        con.raw_sql(statement)

    def assert_n_scans_changes(dt, n_scans_before):
        do_analyze(dt.source, dt.name)
        for _ in range(10):  # noqa: F402
            # give postgres some time to update its tables
            time.sleep(0.1)
            n_scans_after = get_postgres_n_scans(dt)
            if n_scans_before != n_scans_after:
                return n_scans_after
        else:
            raise

    (from_name, to_name) = ("batting", "batting_to_modify")
    if to_name in pg.tables:
        pg.drop_table(to_name)
    pg_t = pg.create_table(to_name, obj=pg.table(from_name).limit(1000))
    storage = SourceSnapshotStorage(source=con)
    expr_cached = (
        pg_t.group_by("playerID").size().order_by("playerID").cache(storage=storage)
    )
    dt = pg_t.op()
    (storage, uncached) = (expr_cached.ls.storage, expr_cached.ls.uncached_one)

    # assert initial state
    assert not storage.exists(uncached)
    n_scans_before = get_postgres_n_scans(dt)
    assert n_scans_before == 0

    # assert first execution state
    executed0 = expr_cached.execute()
    n_scans_after = assert_n_scans_changes(dt, n_scans_before)
    # should we test that SourceStorage.get is called?
    assert n_scans_after == 1
    assert storage.exists(uncached)

    # assert no change after re-execution of cached expr
    executed1 = expr_cached.execute()
    assert n_scans_after == get_postgres_n_scans(dt)
    assert executed0.equals(executed1)

    # assert NO cache invalidation
    modify_postgres_table(dt)
    executed2 = expr_cached.execute()
    assert executed0.equals(executed2)
    with pytest.raises(Exception):
        assert_n_scans_changes(dt, n_scans_after)

    executed3 = expr_cached.ls.uncached.execute()
    assert not executed0.equals(executed3)


def test_postgres_parquet_snapshot(pg, tmp_path):
    def modify_postgres_table(dt):
        (con, name) = (dt.source, dt.name)
        statement = f"""
        INSERT INTO "{name}"
        DEFAULT VALUES
        """
        con.raw_sql(statement)

    def assert_n_scans_changes(dt, n_scans_before):
        do_analyze(dt.source, dt.name)
        for i in range(10):  # noqa: F402
            # give postgres some time to update its tables
            time.sleep(0.1)
            n_scans_after = get_postgres_n_scans(dt)
            if n_scans_before != n_scans_after:
                return n_scans_after
        else:
            raise

    (from_name, to_name) = ("batting", "batting_to_modify")
    if to_name in pg.tables:
        pg.drop_table(to_name)
    pg_t = pg.create_table(to_name, obj=pg.table(from_name).limit(1000))
    storage = ParquetSnapshotStorage(
        relative_path=tmp_path.joinpath("parquet-snapshot-storage")
    )
    expr = pg_t.group_by("playerID").size().order_by("playerID")
    expr_cached = expr.cache(storage=storage)
    dt = pg_t.op()
    (storage, uncached) = (expr_cached.ls.storage, expr_cached.ls.uncached_one)

    # assert initial state
    assert not storage.exists(uncached)
    n_scans_before = get_postgres_n_scans(dt)
    assert n_scans_before == 0

    # assert first execution state
    executed0 = expr_cached.execute()
    n_scans_after = assert_n_scans_changes(dt, n_scans_before)
    # should we test that SourceStorage.get is called?
    assert n_scans_after == 1
    assert storage.exists(uncached)
    assert storage.exists(expr)

    # assert no change after re-execution of cached expr
    executed1 = expr_cached.execute()
    assert n_scans_after == get_postgres_n_scans(dt)
    assert executed0.equals(executed1)

    # assert NO cache invalidation
    modify_postgres_table(dt)
    executed2 = expr_cached.execute()
    assert executed0.equals(executed2)
    with pytest.raises(Exception):
        assert_n_scans_changes(dt, n_scans_after)

    executed3 = expr_cached.ls.uncached.execute()
    assert not executed0.equals(executed3)
