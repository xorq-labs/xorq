import pandas as pd
import pytest


@pytest.fixture(scope="session")
def utf8_data():
    return pd.DataFrame(
        data=[("A", 1), ("B", 2), ("A", 2), ("A", 4), ("C", 1), ("A", 1)],
        columns=["name", "age"],
    )
