import pandas as pd
import pytest

import xorq.api as xo


@pytest.fixture(scope="function")
def con():
    return xo.connect()


@pytest.fixture(scope="session")
def iris_path() -> str:
    path = xo.options.pins.get_path("iris")
    return path


@pytest.fixture(scope="session")
def astronauts_parquet_path(parquet_dir):
    path = parquet_dir / "astronauts.parquet"
    return path


@pytest.fixture(scope="session")
def astronauts_df(astronauts_parquet_path) -> pd.DataFrame:
    return pd.read_parquet(astronauts_parquet_path)
