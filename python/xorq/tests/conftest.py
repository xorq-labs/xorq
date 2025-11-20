import os
from pathlib import Path

import pytest

import xorq.api as xo
from xorq.common.utils.aws_utils import (
    connection_is_set,
    make_s3_credentials_defaults,
)
from xorq.conftest import array_types_df
from xorq.vendor import ibis


FIXTURES_DIR = Path(__file__).parent / "fixtures"

TEST_TABLES = {
    "functional_alltypes": ibis.schema(
        {
            "id": "int32",
            "bool_col": "boolean",
            "tinyint_col": "int8",
            "smallint_col": "int16",
            "int_col": "int32",
            "bigint_col": "int64",
            "float_col": "float32",
            "double_col": "float64",
            "date_string_col": "string",
            "string_col": "string",
            "timestamp_col": "timestamp",
            "year": "int32",
            "month": "int32",
        }
    ),
    "diamonds": ibis.schema(
        {
            "carat": "float64",
            "cut": "string",
            "color": "string",
            "clarity": "string",
            "depth": "float64",
            "table": "float64",
            "price": "int64",
            "x": "float64",
            "y": "float64",
            "z": "float64",
        }
    ),
    "batting": ibis.schema(
        {
            "playerID": "string",
            "yearID": "int64",
            "stint": "int64",
            "teamID": "string",
            "lgID": "string",
            "G": "int64",
            "AB": "int64",
            "R": "int64",
            "H": "int64",
            "X2B": "int64",
            "X3B": "int64",
            "HR": "int64",
            "RBI": "int64",
            "SB": "int64",
            "CS": "int64",
            "BB": "int64",
            "SO": "int64",
            "IBB": "int64",
            "HBP": "int64",
            "SH": "int64",
            "SF": "int64",
            "GIDP": "int64",
        }
    ),
    "awards_players": ibis.schema(
        {
            "playerID": "string",
            "awardID": "string",
            "yearID": "int64",
            "lgID": "string",
            "tie": "string",
            "notes": "string",
        }
    ),
    "astronauts": ibis.schema(
        {
            "id": "int64",
            "number": "int64",
            "nationwide_number": "int64",
            "name": "string",
            "original_name": "string",
            "sex": "string",
            "year_of_birth": "int64",
            "nationality": "string",
            "military_civilian": "string",
            "selection": "string",
            "year_of_selection": "int64",
            "mission_number": "int64",
            "total_number_of_missions": "int64",
            "occupation": "string",
            "year_of_mission": "int64",
            "mission_title": "string",
            "ascend_shuttle": "string",
            "in_orbit": "string",
            "descend_shuttle": "string",
            "hours_mission": "float64",
            "total_hrs_sum": "float64",
            "field21": "int64",
            "eva_hrs_mission": "float64",
            "total_eva_hrs": "float64",
        }
    ),
}

have_s3_credentials = connection_is_set(make_s3_credentials_defaults())


def pytest_runtest_setup(item):
    if any(mark.name == "s3" for mark in item.iter_markers()):
        if not have_s3_credentials:
            pytest.skip("cannot run s3 tests without s3 credentials")
    if any(mark.name == "gcs" for mark in item.iter_markers()):
        if not os.environ.get("GCS_ENABLED", False):
            pytest.skip("GCS are not enabled by default")


@pytest.fixture(scope="session")
def ddl_file(root_dir):
    ddl_dir = root_dir / "db" / "datafusion.sql"
    return ddl_dir


def statements(ddl_file: Path):
    return (
        statement
        for statement in map(str.strip, ddl_file.read_text().split(";"))
        if statement
    )


@pytest.fixture(scope="session")
def con(data_dir, ddl_file):
    conn = xo.connect()
    parquet_dir = data_dir / "parquet"
    conn.read_parquet(
        parquet_dir / "functional_alltypes.parquet", "functional_alltypes"
    )
    conn.read_parquet(parquet_dir / "batting.parquet", "batting")
    conn.read_parquet(parquet_dir / "diamonds.parquet", "diamonds")
    conn.read_parquet(parquet_dir / "astronauts.parquet", "astronauts")
    conn.read_parquet(parquet_dir / "awards_players.parquet", "awards_players")

    conn.create_table("array_types", array_types_df)

    if ddl_file.is_file() and ddl_file.name.endswith(".sql"):
        for statement in statements(ddl_file):
            with conn._safe_raw_sql(statement):  # noqa
                pass

    return conn


@pytest.fixture(scope="session")
def functional_alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope="session")
def batting(con):
    return con.table("batting")


@pytest.fixture(scope="session")
def batting_df(batting):
    return batting.execute()


@pytest.fixture(scope="session")
def awards_players(con):
    return con.table("awards_players")


@pytest.fixture(scope="session")
def awards_players_df(awards_players):
    return awards_players.execute(limit=None)


@pytest.fixture(scope="session")
def sorted_df(df):
    return df.sort_values("id").reset_index(drop=True)


@pytest.fixture(scope="session")
def diamonds(con):
    return con.table("diamonds")


@pytest.fixture(scope="session")
def array_types(con):
    return con.table("array_types")


@pytest.fixture(scope="session")
def struct(con):
    return con.table("structs")


@pytest.fixture(scope="session")
def struct_df(struct):
    return struct.execute()
