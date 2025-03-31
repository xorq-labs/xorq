import pyarrow as pa
import pytest

import xorq as xo


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


def remove_unexpected_tables(dirty):
    # drop tables
    for table in dirty.list_tables():
        if table not in expected_tables:
            dirty.drop_table(table, force=True)

    # drop view
    for table in dirty.list_tables():
        if table not in expected_tables:
            dirty.drop_view(table, force=True)

    if sorted(dirty.list_tables()) != sorted(expected_tables):
        raise ValueError


@pytest.fixture(scope="function")
def con(pg):
    remove_unexpected_tables(pg)
    yield pg
    # cleanup
    remove_unexpected_tables(pg)


@pytest.fixture(scope="session")
def dirty_ls_con():
    con = xo.connect()
    return con


@pytest.fixture(scope="function")
def ls_con(dirty_ls_con):
    # since we don't register, maybe just create a fresh con
    yield dirty_ls_con
    # drop tables
    for table_name in dirty_ls_con.list_tables():
        dirty_ls_con.drop_table(table_name, force=True)
    # drop view
    for table_name in dirty_ls_con.list_tables():
        dirty_ls_con.drop_view(table_name, force=True)


@pytest.fixture(scope="session")
def alltypes(pg):
    return pg.table("functional_alltypes")


@pytest.fixture(scope="session")
def batting(pg):
    return pg.table("batting")


@pytest.fixture(scope="session")
def awards_players(pg):
    return pg.table("awards_players")


@pytest.fixture(scope="session")
def alltypes_df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="session")
def batting_df(batting):
    return batting.execute()


@pytest.fixture
def df():
    # create a RecordBatch and a new DataFrame from it
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([0, 1, 2, 3, 4, 5, 6]),
            pa.array([7, 4, 3, 8, 9, 1, 6]),
            pa.array(["A", "A", "A", "A", "B", "B", "B"]),
        ],
        names=["a", "b", "c"],
    )

    return batch.to_pandas()
