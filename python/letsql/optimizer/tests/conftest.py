from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir():
    root = Path(__file__).absolute().parents[4]
    data_dir = root / "ci" / "ibis-testing-data"
    return data_dir


@pytest.fixture
def tmp_model_dir(tmpdir):
    # Create a temporary directory for the model
    model_dir = tmpdir.mkdir("models")
    return model_dir
