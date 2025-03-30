from pathlib import Path

import pytest


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
