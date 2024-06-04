from __future__ import annotations

import pathlib
import time

import ibis
import pytest
from ibis import _

import letsql
from letsql.backends.let import (
    Backend,
)
from letsql.backends.conftest import (
    get_storage_uncached,
)
from letsql.common.caching import (
    KEY_PREFIX,
    SourceStorage,
    ParquetCacheStorage,
)
from letsql.common.utils.postgres_utils import (
    do_analyze,
    get_postgres_n_scans,
)
from letsql.tests.util import (
    assert_frame_equal,
)


@pytest.fixture(scope="function")
def pg_alltypes(pg):
    return pg.table("functional_alltypes")


def test_cache_simple(con, alltypes, alltypes_df):
    initial_tables = con.list_tables()

    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    cached = expr.cache()
    tables_after_caching = con.list_tables()

    expected = alltypes_df[
        (alltypes_df["float_col"] > 0)
        & (alltypes_df["smallint_col"] == 9)
        & (alltypes_df["int_col"] < alltypes_df["float_col"] * 2)
    ][["smallint_col", "int_col", "float_col"]]

    cached = cached.execute()
    tables_after_executing = con.list_tables()

    assert_frame_equal(cached, expected)
    assert not any(
        table_name.startswith(KEY_PREFIX)
        for table_name in set(tables_after_caching).difference(initial_tables)
    )
    assert any(
        table_name.startswith(KEY_PREFIX)
        for table_name in set(tables_after_executing).difference(initial_tables)
    )


def test_cache_multiple_times(con, alltypes, alltypes_df):
    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    cached = expr.cache()

    # reassign the expression
    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )

    re_cached = expr.cache()

    first = cached.execute()
    tables_after_first_caching = con.list_tables()

    second = re_cached.execute()
    tables_after_second_caching = con.list_tables()

    expected = alltypes_df[
        (alltypes_df["float_col"] > 0)
        & (alltypes_df["smallint_col"] == 9)
        & (alltypes_df["int_col"] < alltypes_df["float_col"] * 2)
    ][["smallint_col", "int_col", "float_col"]]

    assert_frame_equal(first, expected)
    assert_frame_equal(second, expected)

    first_tables = [t for t in tables_after_first_caching if t.startswith(KEY_PREFIX)]
    second_tables = [t for t in tables_after_second_caching if t.startswith(KEY_PREFIX)]

    assert sorted(first_tables) == sorted(second_tables)


def test_cache_to_sql(con, alltypes):
    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    cached = expr.cache()

    assert ibis.to_sql(cached) == ibis.to_sql(expr)


def test_op_after_cache(alltypes):
    expr = alltypes.select(alltypes.smallint_col, alltypes.int_col, alltypes.float_col)
    cached = expr.cache()
    cached = cached.filter(
        [
            _.float_col > 0,
            _.smallint_col == 9,
            _.int_col < _.float_col * 2,
        ]
    )

    full_expr = expr.filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )

    actual = cached.execute()
    expected = full_expr.execute()

    assert_frame_equal(actual, expected)

    assert ibis.to_sql(cached) == ibis.to_sql(full_expr)


def setup_backend(table, table_name):
    other = Backend()
    other.do_connect()
    other.register(table, table_name=table_name)
    return other


def test_cache_recreate(con, alltypes, pg_alltypes):
    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    expr.cache().execute()  # execute creation of tables

    other = setup_backend(pg_alltypes, "functional_alltypes")
    other_alltypes = other.table("functional_alltypes")
    other_expr = other_alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    other_expr.cache().execute()

    con_cached_tables = set(
        table_name
        for table_name in con.list_tables()
        if table_name.startswith(KEY_PREFIX)
    )
    other_cached_tables = set(
        table_name
        for table_name in other.list_tables()
        if table_name.startswith(KEY_PREFIX)
    )

    assert con_cached_tables
    assert con_cached_tables == other_cached_tables
    for table_name in other_cached_tables:
        assert_frame_equal(
            con.table(table_name).to_pandas(), other.table(table_name).to_pandas()
        )


def test_cache_execution(alltypes):
    cached = (
        alltypes.select(alltypes.smallint_col, alltypes.int_col, alltypes.float_col)
        .cache()
        .filter(
            [
                _.float_col > 0,
                _.smallint_col == 9,
                _.int_col < _.float_col * 2,
            ]
        )
        .select(_.int_col * 4)
        .cache()
    )

    actual = cached.execute()

    expected = (
        alltypes.select(alltypes.smallint_col, alltypes.int_col, alltypes.float_col)
        .filter(
            [
                alltypes.float_col > 0,
                alltypes.smallint_col == 9,
                alltypes.int_col < alltypes.float_col * 2,
            ]
        )
        .select(alltypes.int_col * 4)
        .execute()
    )

    assert_frame_equal(actual, expected)


def test_parquet_cache_storage(tmp_path, alltypes_df):
    tmp_path = pathlib.Path(tmp_path)
    path = tmp_path.joinpath("to-delete.parquet")

    con = letsql.connect()
    alltypes_df.to_parquet(path)
    t = con.register(path, "t")
    cols = ["id", "bool_col", "float_col", "string_col"]
    expr = t[cols]
    expected = alltypes_df[cols]
    source = expr._find_backend()
    storage = letsql.common.caching.ParquetCacheStorage(
        tmp_path.joinpath("parquet-cache-storage"), source=source
    )
    cached = expr.cache(storage=storage)
    actual = cached.execute()
    assert_frame_equal(actual, expected)

    # the file must exist and have the same schema
    alltypes_df.head(1).to_parquet(path)
    actual = cached.execute()
    assert_frame_equal(actual, expected)

    path.unlink()
    pattern = "".join(
        (
            "Object Store error: Object at location",
            ".*",
            "not found: No such file or directory",
        )
    )
    with pytest.raises(Exception, match=pattern):
        # if the file doesn't exist, we get a failure, even for cached
        cached.execute()


def test_parquet_remote_to_local(con, alltypes, tmp_path):
    tmp_path = pathlib.Path(tmp_path)

    expr = alltypes.select(
        alltypes.smallint_col, alltypes.int_col, alltypes.float_col
    ).filter(
        [
            alltypes.float_col > 0,
            alltypes.smallint_col == 9,
            alltypes.int_col < alltypes.float_col * 2,
        ]
    )
    storage = letsql.common.caching.ParquetCacheStorage(
        tmp_path.joinpath("parquet-cache-storage"), source=con
    )
    cached = expr.cache(storage=storage)
    expected = expr.execute()
    actual = cached.execute()
    assert_frame_equal(actual, expected)


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
    pg_t = pg.create_table(name=to_name, obj=pg.table(from_name))
    expr_cached = (
        con.register(pg_t, table_name=to_name)
        .group_by("playerID")
        .size()
        .order_by("playerID")
        .cache()
    )
    dt = pg_t.op()
    (storage, uncached) = get_storage_uncached(con, expr_cached)

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


@pytest.mark.xfail(reason="no strategy to find the parquet file source")
def test_duckdb_cache_parquet(con, pg, tmp_path):
    name = "batting"
    parquet_path = tmp_path.joinpath(name).with_suffix(".parquet")
    pg.table(name).to_parquet(parquet_path)
    expr = (
        ibis.duckdb.connect()
        .read_parquet(parquet_path)
        .pipe(con.register, f"duckdb-{name}")[lambda t: t.yearID > 2000]
        .cache(storage=ParquetCacheStorage(path=tmp_path, source=con))
    )
    expr.execute()


@pytest.mark.xfail
def test_duckdb_cache_arrow(con, pg, tmp_path):
    name = "batting"
    expr = (
        ibis.duckdb.connect()
        .register(pg.table(name).to_pyarrow(), name)
        .pipe(con.register, f"duckdb-{name}")[lambda t: t.yearID > 2000]
        .cache(storage=ParquetCacheStorage(path=tmp_path, source=con))
    )
    expr.execute()


def test_cross_source_storage(con, pg):
    name = "batting"
    expr = (
        ibis.duckdb.connect()
        .register(pg.table(name).to_pyarrow(), name)
        .pipe(con.register, f"duckdb-{name}")[lambda t: t.yearID > 2000]
        .cache(storage=SourceStorage(source=pg))
    )
    expr.execute()
