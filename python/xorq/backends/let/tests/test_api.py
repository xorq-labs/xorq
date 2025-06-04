import pytest

import xorq as xo


def test_register_read_csv(csv_dir):
    # this will use ls.options.backend: do we want to clear it out between function invocations?
    api_batting = xo.read_csv(csv_dir / "batting.csv", table_name="api_batting")
    result = xo.execute(api_batting)

    assert result is not None


def test_register_read_parquet(parquet_dir):
    # this will use ls.options.backend: do we want to clear it out between function invocations?
    api_batting = xo.read_parquet(
        parquet_dir / "batting.parquet", table_name="api_batting"
    )
    result = xo.execute(api_batting)

    assert result is not None


@pytest.mark.xfail(reason="No purpose with no registration api")
def test_executed_on_original_backend(parquet_dir, csv_dir, mocker):
    con = xo.config._backend_init()
    spy = mocker.spy(con, "execute")

    parquet_table = xo.read_parquet(parquet_dir / "batting.parquet")[
        lambda t: t.yearID == 2015
    ]

    csv_table = xo.read_csv(csv_dir / "batting.csv")[lambda t: t.yearID == 2014]

    expr = parquet_table.join(
        csv_table,
        "playerID",
    )

    assert xo.execute(expr) is not None
    assert spy.call_count == 1


def test_read_postgres():
    import os

    uri = (
        f"postgres://{os.environ['POSTGRES_USER']}:"
        f"{os.environ['POSTGRES_PASSWORD']}@"
        f"{os.environ['POSTGRES_HOST']}:"
        f"{os.environ['POSTGRES_PORT']}/"
        f"{os.environ['POSTGRES_DATABASE']}"
    )
    t = xo.read_postgres(uri, table_name="batting")
    res = xo.execute(t)

    assert res is not None and len(res)


@pytest.mark.parametrize(
    ("with_repartition_file_scans", "keep_partition_by_columns"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_with_config(
    with_repartition_file_scans, keep_partition_by_columns, parquet_dir
):
    import pandas as pd

    from xorq import SessionConfig

    session_config = (
        SessionConfig()
        .with_repartition_file_scans(with_repartition_file_scans)
        .set(
            "datafusion.execution.keep_partition_by_columns",
            str(keep_partition_by_columns).lower(),
        )
    )

    con = xo.connect(session_config=session_config)

    expr = con.read_parquet(parquet_dir / "batting.parquet").limit(10)
    result = expr.execute()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10
