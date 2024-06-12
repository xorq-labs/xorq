import ibis
import pytest

import letsql as ls


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
    remove_unexpected_tables(conn)


@pytest.fixture(scope="session")
def dirty(pg):
    con = ls.connect()

    for table_name in expected_tables:
        con.register(pg.table(table_name), table_name=table_name)

    return con


def remove_unexpected_tables(dirty):
    for table in dirty.list_tables():
        if table not in expected_tables:
            dirty.drop_table(table)
    if sorted(dirty.list_tables()) != sorted(expected_tables):
        raise ValueError


@pytest.fixture(scope="function")
def con(dirty):
    remove_unexpected_tables(dirty)
    yield dirty
    # cleanup
    remove_unexpected_tables(dirty)


@pytest.fixture(scope="session")
def dirty_ls_con():
    con = ls.connect()
    return con


@pytest.fixture(scope="function")
def ls_con(dirty_ls_con):
    yield dirty_ls_con
    for table_name in dirty_ls_con.list_tables():
        dirty_ls_con.drop_table(table_name)


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


@pytest.fixture(scope="session")
def batting_df(batting):
    return batting.execute()
