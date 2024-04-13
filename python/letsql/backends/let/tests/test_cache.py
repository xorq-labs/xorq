from __future__ import annotations


import ibis

from letsql.backends.let.tests.conftest import assert_frame_equal
from letsql.backends.let import (
    Backend,
    KEY_PREFIX,
)

from ibis import _


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


def test_op_after_cache(con, alltypes):
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


def test_cache_recreate(con, alltypes):
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
    other = Backend()
    other.do_connect()

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


def test_cache_execution(con, alltypes):
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
