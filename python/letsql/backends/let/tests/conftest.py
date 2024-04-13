from typing import Any

import pytest
import ibis
import pandas as pd
import pandas.testing as tm

import letsql as ls
from letsql.backends.let import KEY_PREFIX


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
    for con in dirty.connections.values():
        for table in con.list_tables():
            # FIXME: determine if we should drop all or only key-prefixed
            if table.startswith(KEY_PREFIX):
                con.drop_table(table)
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
