from typing import Any

import ibis
import pandas as pd
import pandas.testing as tm
import pytest

import letsql as ls
from letsql.backends.let import KEY_PREFIX

expected_tables = (
    "array_types",
    "astronauts",
    "awards_players",
    "awards_players_special_types",
    "batting",
    "diamonds",
    "functional_alltypes",
    "geo",
    "geography_columns",
    "geometry_columns",
    "json_t",
    "map",
    "spatial_ref_sys",
    "topk",
    "tzone",
)


@pytest.fixture(scope="session")
def pg():
    conn = ibis.postgres.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        database="ibis_testing",
    )
    yield conn
    for table in conn.list_tables():
        if table.startswith(KEY_PREFIX):
            conn.drop_table(table)


@pytest.fixture(scope="session")
def dirty(pg):
    con = ls.connect()

    for table_name in expected_tables:
        con.register(pg.table(table_name), table_name=table_name)

    return con


def remove_cached_tables(dirty):
    for table in dirty.list_tables():
        # FIXME: determine if we should drop all or only key-prefixed
        if table.startswith(KEY_PREFIX):
            dirty.drop_table(table)
    if sorted(dirty.list_tables()) != sorted(expected_tables):
        raise ValueError


@pytest.fixture(scope="function")
def con(dirty):
    remove_cached_tables(dirty)
    yield dirty
    # cleanup
    remove_cached_tables(dirty)


@pytest.fixture(scope="session")
def alltypes(dirty):
    return dirty.table("functional_alltypes")


@pytest.fixture(scope="session")
def batting(dirty):
    return dirty.table("batting")


@pytest.fixture(scope="session")
def awards_players(dirty):
    return dirty.table("awards_players")


@pytest.fixture(scope="session")
def alltypes_df(alltypes):
    return alltypes.execute()


def assert_frame_equal(
    left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
) -> None:
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    kwargs.setdefault("check_dtype", True)
    tm.assert_frame_equal(left, right, *args, **kwargs)
