import numpy as np
import pandas as pd
import pytest

import xorq as xo


QUOTES_TABLE_NAME = "quotes"


@pytest.fixture(scope="session")
def quotes_df():
    # Quote data (more frequent updates)
    quote_dates = pd.date_range(
        start="2024-01-01", end="2024-01-31", freq="30min"
    ).astype("datetime64[us]")

    quotes = pd.DataFrame(
        {
            "timestamp": quote_dates,
            "symbol": np.random.choice(["AAPL", "GOOGL", "MSFT"], len(quote_dates)),
            "bid": np.random.uniform(99, 198, len(quote_dates)),
            "ask": np.random.uniform(101, 202, len(quote_dates)),
        }
    )

    return quotes


@pytest.fixture
def trades_df():
    # Create sample trading data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="h").astype(
        "datetime64[us]"
    )

    # Trades data
    trades = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": np.random.choice(["AAPL", "GOOGL", "MSFT"], len(dates)),
            "price": np.random.uniform(100, 200, len(dates)),
            "volume": np.random.randint(100, 1000, len(dates)),
        }
    )

    return trades


@pytest.fixture(scope="session")
def iceberg_con(tmp_path_factory, quotes_df):
    warehouse_path = str(tmp_path_factory.mktemp("warehouse"))
    con = xo.pyiceberg.connect(warehouse_path=warehouse_path)
    con.create_table(QUOTES_TABLE_NAME, quotes_df, overwrite=True)
    return con


@pytest.fixture(scope="session")
def quotes_table(iceberg_con):
    return iceberg_con.table(QUOTES_TABLE_NAME)
