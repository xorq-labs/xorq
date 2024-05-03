from typing import Any

import pytest
import ibis
import pandas as pd
import pandas.testing as tm

import letsql as ls
from letsql.backends.let import KEY_PREFIX


expected_tables = (
    "array_types",
    "astronauts",
    "awards_players",
    "awards_players_special_types",
    "batting",
    "diamonds",
    "films",
    "functional_alltypes",
    "geo",
    "geography_columns",
    "geometry_columns",
    "intervals",
    "json_t",
    "map",
    "not_supported_intervals",
    "spatial_ref_sys",
    "topk",
    "tzone",
    "win",
)


@pytest.fixture(scope="session")
def dirty():
    con = ls.connect()
    con.add_connection(
        ibis.postgres.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="postgres",
            database="ibis_testing",
        )
    )
    return con


def remove_cached_tables(dirty):
    for con in dirty.list_connections():
        for table in con.list_tables():
            # FIXME: determine if we should drop all or only key-prefixed
            if table.startswith(KEY_PREFIX):
                con.drop_table(table)
    if sorted(dirty.list_tables()) != sorted(expected_tables):
        raise ValueError


@pytest.fixture(scope="function")
def con(dirty):
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
