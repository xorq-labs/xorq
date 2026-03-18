from __future__ import annotations

import concurrent.futures
import getpass
import sys
from os import environ as env

import pytest

import xorq.api as xo


DATABRICKS_CATALOG = env.get("IBIS_TESTING_DATABRICKS_CATALOG") or "workspace"


def put_into(con, query):
    with con.cursor() as cur:
        cur.execute(query)


@pytest.fixture(scope="session")
def con(data_dir):
    import databricks.sql  # noqa: PLC0415

    files = list(data_dir.joinpath("parquet").glob("*.parquet"))

    user = getpass.getuser()
    python_version = "".join(map(str, sys.version_info[:3]))
    volume = f"{user}_{python_version}"
    volume_prefix = f"/Volumes/{DATABRICKS_CATALOG}/default/{volume}"

    with databricks.sql.connect(
        server_hostname=env["DATABRICKS_SERVER_HOSTNAME"],
        http_path=env["DATABRICKS_HTTP_PATH"],
        access_token=env["DATABRICKS_TOKEN"],
        staging_allowed_local_path=str(data_dir),
    ) as raw_con:
        with raw_con.cursor() as cur:
            cur.execute(
                f"CREATE VOLUME IF NOT EXISTS {DATABRICKS_CATALOG}.default.{volume} COMMENT 'xorq test data storage'"
            )
        with concurrent.futures.ThreadPoolExecutor() as exe:
            for fut in concurrent.futures.as_completed(
                exe.submit(
                    put_into,
                    raw_con,
                    f"PUT '{file}' INTO '{volume_prefix}/{file.name}' OVERWRITE",
                )
                for file in files
            ):
                fut.result()

    connection = xo.databricks.connect(
        server_hostname=env["DATABRICKS_SERVER_HOSTNAME"],
        http_path=env["DATABRICKS_HTTP_PATH"],
        access_token=env["DATABRICKS_TOKEN"],
        catalog=DATABRICKS_CATALOG,
        schema="default",
    )
    connection._test_volume_prefix = volume_prefix
    yield connection
    connection.con.close()


@pytest.fixture(scope="session")
def functional_alltypes(con):
    path = f"{con._test_volume_prefix}/functional_alltypes.parquet"
    return con.sql(f"SELECT * FROM read_files('{path}', format => 'parquet')")


@pytest.fixture(scope="session")
def batting(con):
    path = f"{con._test_volume_prefix}/batting.parquet"
    return con.sql(f"SELECT * FROM read_files('{path}', format => 'parquet')")


@pytest.fixture(scope="session")
def diamonds(con):
    path = f"{con._test_volume_prefix}/diamonds.parquet"
    return con.sql(f"SELECT * FROM read_files('{path}', format => 'parquet')")
