from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest


# test modules import sklearn (and test_inline orbital) at module level
collect_ignore_glob = [
    *(() if find_spec("sklearn") else ("test_*.py",)),
    *(() if find_spec("orbital") else ("test_inline.py",)),
]


@pytest.fixture(scope="module")
def taxi_df():
    rng = np.random.default_rng(0)
    n = 2_000
    df = pd.DataFrame(
        {
            "trip_distance": rng.gamma(2, 2, n),
            "passenger_count": rng.integers(1, 6, n).astype("float64"),
            "hour": rng.integers(0, 24, n).astype("float64"),
            "dow": rng.integers(0, 7, n).astype("float64"),
            "payment_type": rng.choice(["1", "2", "3", "4"], n),
            "RatecodeID": rng.choice(["1", "2", "5"], n),
            "tip_amount": rng.random(n),
        }
    )
    return df.assign(
        fare_amount=3
        + 2.5 * df.trip_distance
        + 1.5 * (df.payment_type == "2")
        + rng.normal(0, 1, n)
    )
