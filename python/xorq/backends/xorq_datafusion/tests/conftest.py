import pyarrow as pa
import pytest

import xorq.api as xo
from xorq.conftest import make_xo_con


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
        raise ValueError(
            f"unexpected tables after cleanup: {set(dirty.list_tables()) - set(expected_tables)}"
        )


@pytest.fixture(scope="session")
def xo_con(parquet_dir, ddl_file):
    # this directory's own connection: its tests create and drop tables
    return make_xo_con(parquet_dir, ddl_file)


@pytest.fixture(scope="session")
def con(xo_con):
    return xo_con


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
