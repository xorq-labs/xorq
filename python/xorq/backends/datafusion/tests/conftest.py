import numpy as np
import pandas as pd
import pytest

import xorq.api as xo


@pytest.fixture
def quotes_df():
    # Quote data (more frequent updates)
    quote_dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="30min")
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
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="H")

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


@pytest.fixture
def ddb_con(quotes_df):
    """Create DuckDB connection with Ibis"""
    duckdb_connection = xo.duckdb.connect()
    duckdb_connection.create_table(
        "quotes",
        quotes_df,
    )
    return duckdb_connection


@pytest.fixture
def con(trades_df):
    """Create DataFusion connection with xorq"""
    con = xo.datafusion.connect()
    con.create_table("trades", trades_df, temp=False)
    return con


@pytest.fixture(scope="session")
def ls_con(parquet_dir):
    conn = xo.connect()

    for name in ("astronauts", "functional_alltypes", "awards_players", "batting"):
        conn.read_parquet(parquet_dir / f"{name}.parquet", name)

    return conn


@pytest.fixture(scope="session")
def alltypes(ls_con):
    return ls_con.table("functional_alltypes")


@pytest.fixture(scope="session")
def alltypes_df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="session")
def awards_players(ls_con):
    return ls_con.table("awards_players")


@pytest.fixture(scope="session")
def batting(ls_con):
    return ls_con.table("batting")
