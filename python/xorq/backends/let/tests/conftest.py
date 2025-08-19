from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import xorq.api as xo


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


array_types_df = pd.DataFrame(
    [
        (
            [np.int64(1), 2, 3],
            ["a", "b", "c"],
            [1.0, 2.0, 3.0],
            "a",
            1.0,
            [[], [np.int64(1), 2, 3], None],
        ),
        (
            [4, 5],
            ["d", "e"],
            [4.0, 5.0],
            "a",
            2.0,
            [],
        ),
        (
            [6, None],
            ["f", None],
            [6.0, np.nan],
            "a",
            3.0,
            [None, [], None],
        ),
        (
            [None, 1, None],
            [None, "a", None],
            [],
            "b",
            4.0,
            [[1], [2], [], [3, 4, 5]],
        ),
        (
            [2, None, 3],
            ["b", None, "c"],
            np.nan,
            "b",
            5.0,
            None,
        ),
        (
            [4, None, None, 5],
            ["d", None, None, "e"],
            [4.0, np.nan, np.nan, 5.0],
            "c",
            6.0,
            [[1, 2, 3]],
        ),
    ],
    columns=[
        "x",
        "y",
        "z",
        "grouper",
        "scalar_column",
        "multi_dim",
    ],
)


@pytest.fixture(scope="session")
def ddl_file(root_dir):
    ddl_dir = root_dir / "db" / "datafusion.sql"
    return ddl_dir


def statements(ddl_file: Path):
    return (
        statement
        for statement in map(str.strip, ddl_file.read_text().split(";"))
        if statement
    )


@pytest.fixture(scope="session")
def con(data_dir, ddl_file):
    conn = xo.connect()
    parquet_dir = data_dir / "parquet"
    conn.read_parquet(
        parquet_dir / "functional_alltypes.parquet", "functional_alltypes"
    )
    conn.read_parquet(parquet_dir / "batting.parquet", "batting")
    conn.read_parquet(parquet_dir / "diamonds.parquet", "diamonds")
    conn.read_parquet(parquet_dir / "astronauts.parquet", "astronauts")
    conn.read_parquet(parquet_dir / "awards_players.parquet", "awards_players")

    conn.create_table("array_types", array_types_df)

    if ddl_file.is_file() and ddl_file.name.endswith(".sql"):
        for statement in statements(ddl_file):
            with conn._safe_raw_sql(statement):  # noqa
                pass

    return conn


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
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def batting(con):
    return con.table("batting")


@pytest.fixture(scope="session")
def awards_players(con):
    return con.table("awards_players")


@pytest.fixture(scope="session")
def diamonds(con):
    return con.table("diamonds")


@pytest.fixture(scope="session")
def alltypes_df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="session")
def batting_df(batting):
    return batting.execute()


@pytest.fixture(scope="session")
def sorted_df(alltypes_df):
    return alltypes_df.sort_values("id").reset_index(drop=True)


@pytest.fixture(scope="session")
def array_types(con):
    return con.table("array_types")


@pytest.fixture(scope="session")
def struct(con):
    return con.table("structs")


@pytest.fixture(scope="session")
def struct_df(struct):
    return struct.execute()


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
