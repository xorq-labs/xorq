from typing import Any

import pytest
import ibis
import pandas as pd
import pandas.testing as tm

import letsql as ls


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


@pytest.fixture(scope="function")
def con(dirty):
    # cleanup
    for table in dirty.list_tables():
        if table.startswith("ibis_cache"):
            dirty.drop_table(table)
    dirty.cache_storage.clear()
    return dirty


@pytest.fixture(scope="session")
def alltypes(dirty):
    return dirty.table("functional_alltypes")


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
