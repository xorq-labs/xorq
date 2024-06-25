from pathlib import Path

import pytest


import letsql as ls


@pytest.fixture(scope="session")
def csv_dir():
    root = Path(__file__).absolute().parents[5]
    data_dir = root / "ci" / "ibis-testing-data" / "csv"
    return data_dir


@pytest.fixture(scope="session")
def parquet_dir():
    root = Path(__file__).absolute().parents[5]
    data_dir = root / "ci" / "ibis-testing-data" / "parquet"
    return data_dir


def test_register_read_csv(con, csv_dir):
    api_batting = con.register(
        ls.read_csv(csv_dir / "batting.csv"), table_name="api_batting"
    )
    result = api_batting.execute()

    assert result is not None


def test_register_read_parquet(con, parquet_dir):
    api_batting = con.register(
        ls.read_parquet(parquet_dir / "batting.parquet"), table_name="api_batting"
    )
    result = api_batting.execute()

    assert result is not None


def test_executed_on_original_backend(ls_con, parquet_dir, csv_dir, mocker):
    con = ls.config._backend_init()
    spy = mocker.spy(con, "execute")

    table_name = "batting"
    parquet_table = ls.read_parquet(parquet_dir / "batting.parquet")[
        lambda t: t.yearID == 2015
    ].pipe(ls_con.register, f"parquet-{table_name}")

    csv_table = ls.read_csv(csv_dir / "batting.csv")[lambda t: t.yearID == 2014].pipe(
        ls_con.register, f"csv-{table_name}"
    )

    expr = parquet_table.join(
        csv_table,
        "playerID",
    )

    assert expr.execute() is not None
    assert spy.call_count == 1
