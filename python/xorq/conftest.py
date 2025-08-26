from pathlib import Path

import dask
import numpy as np
import pandas as pd
import pytest

import xorq.api as xo


# ensure registration of numpy and pandas objects for tokenization purposes
dask.base.normalize_token.dispatch(np.dtype)
dask.base.normalize_token.dispatch(pd.DataFrame)


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
def pg():
    conn = xo.postgres.connect_env()
    remove_unexpected_tables(conn)
    yield conn
    remove_unexpected_tables(conn)


@pytest.fixture(scope="session")
def root_dir():
    return Path(__file__).absolute().parents[2]


@pytest.fixture(scope="session")
def parquet_dir(root_dir):
    data_dir = root_dir / "ci" / "ibis-testing-data" / "parquet"
    return data_dir


@pytest.fixture(scope="session")
def fixture_dir(root_dir):
    return root_dir.joinpath("python", "xorq", "tests", "fixtures")


@pytest.fixture(scope="session")
def data_dir(root_dir):
    data_dir = root_dir / "ci" / "ibis-testing-data"
    return data_dir


@pytest.fixture(scope="session")
def csv_dir(data_dir):
    csv_dir = data_dir / "csv"
    return csv_dir


@pytest.fixture(scope="session")
def examples_dir(root_dir):
    examples_dir = root_dir / "examples"
    return examples_dir
