import pandas as pd
import pytest

import xorq.api as xo


@pytest.fixture(scope="session")
def sqlite_con():
    return xo.sqlite.connect()


@pytest.fixture(scope="session")
def astronauts_parquet_path(parquet_dir):
    return parquet_dir / "astronauts.parquet"


@pytest.fixture(scope="session")
def astronauts_csv_path(csv_dir):
    return csv_dir / "astronauts.csv"


@pytest.fixture(scope="session")
def persistent_sqlite_con(tmp_path_factory):
    path = tmp_path_factory.mktemp("database").joinpath("lite.db").resolve()
    return xo.sqlite.connect(str(path))


@pytest.fixture(scope="session")
def df():
    return pd.DataFrame(
        list(
            zip(
                *[
                    [0, 1.5, 2.3, 3, 4, 5, 6],
                    [7, 4, 3, 8, 9, 1, 6],
                    ["A", "A", "A", "A", "B", "B", "B"],
                ]
            )
        ),
        columns=["a", "b", "c"],
    )
