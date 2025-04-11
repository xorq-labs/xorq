import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import xorq as xo
from xorq.common.utils.aws_utils import (
    connection_is_set,
    make_s3_credentials_defaults,
)
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

array_types_df = pd.DataFrame(
    [
        (
            [np.int64(1), 2, 3],
            ["a", "b", "c"],
            [1.0, 2.0, 3.0],
            "a",
            1.0,
            [[], [np.int64(1), 2, 3], None],
        ),
        (
            [4, 5],
            ["d", "e"],
            [4.0, 5.0],
            "a",
            2.0,
            [],
        ),
        (
            [6, None],
            ["f", None],
            [6.0, np.nan],
            "a",
            3.0,
            [None, [], None],
        ),
        (
            [None, 1, None],
            [None, "a", None],
            [],
            "b",
            4.0,
            [[1], [2], [], [3, 4, 5]],
        ),
        (
            [2, None, 3],
            ["b", None, "c"],
            np.nan,
            "b",
            5.0,
            None,
        ),
        (
            [4, None, None, 5],
            ["d", None, None, "e"],
            [4.0, np.nan, np.nan, 5.0],
            "c",
            6.0,
            [[1, 2, 3]],
        ),
    ],
    columns=[
        "x",
        "y",
        "z",
        "grouper",
        "scalar_column",
        "multi_dim",
    ],
)

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


@pytest.fixture(scope="session")
def float_model_path():
    """Diamonds model with all float features"""
    return FIXTURES_DIR / "pretrained_model.json"


@pytest.fixture(scope="session")
def mixed_model_path():
    return FIXTURES_DIR / "pretrained_model_mixed.json"


@pytest.fixture(scope="session")
def hyphen_model_path(tmp_path_factory):
    model_path = FIXTURES_DIR / "pretrained_model.json"
    fn = tmp_path_factory.mktemp("data") / "diamonds-model.json"
    shutil.copy(model_path, fn)
    return fn


@pytest.fixture(scope="session")
def mixed_feature_table():
    con = xo.connect()
    df = pd.DataFrame(
        {
            "carat": [0.23, 0.21, 0.23],
            "depth": [61.5, None, 56.9],
            "table": [55.0, 61.0, 65.0],
            "x": [3.95, 3.89, 4.05],
            "y": [3.98, 3.84, 4.07],
            "z": [2.43, 2.31, 2.31],
            "cut_good": [False, False, True],
            "cut_ideal": [True, False, False],
            "cut_premium": [False, True, False],
            "cut_very_good": [False, False, False],
            "color_e": [True, True, True],
            "color_f": [False, False, False],
            "color_g": [False, False, False],
            "color_h": [False, False, False],
            "color_i": [False, False, False],
            "color_j": [False, False, False],
            "clarity_if": [False, False, False],
            "clarity_si1": [False, True, False],
            "clarity_si2": [True, False, False],
            "clarity_vs1": [False, False, True],
            "clarity_vs2": [False, False, False],
            "clarity_vvs1": [False, False, False],
            "clarity_vvs2": [False, False, False],
            "target": [326, 326, 327],
            "expected_pred": [472.00235, 580.98920, 480.31976],
        }
    )
    return con.create_table("mixed_table", df)


@pytest.fixture(scope="session")
def feature_table():
    con = xo.connect()
    df = pd.DataFrame(
        {
            "carat": [0.23, 0.21, 0.23],
            "depth": [61.5, None, 56.9],
            "table": [55.0, 61.0, 65.0],
            "x": [3.95, 3.89, 4.05],
            "y": [3.98, 3.84, 4.07],
            "z": [2.43, 2.31, 2.31],
            "cut_good": [0.0, 0.0, 1.0],
            "cut_ideal": [1.0, 0.0, 0.0],
            "cut_premium": [0.0, 1.0, 0.0],
            "cut_very_good": [0.0, 0.0, 0.0],
            "color_e": [1.0, 1.0, 1.0],
            "color_f": [0.0, 0.0, 0.0],
            "color_g": [0.0, 0.0, 0.0],
            "color_h": [0.0, 0.0, 0.0],
            "color_i": [0.0, 0.0, 0.0],
            "color_j": [0.0, 0.0, 0.0],
            "clarity_if": [0.0, 0.0, 0.0],
            "clarity_si1": [0.0, 1.0, 0.0],
            "clarity_si2": [1.0, 0.0, 0.0],
            "clarity_vs1": [0.0, 0.0, 1.0],
            "clarity_vs2": [0.0, 0.0, 0.0],
            "clarity_vvs1": [0.0, 0.0, 0.0],
            "clarity_vvs2": [0.0, 0.0, 0.0],
            "target": [326, 326, 327],
            "expected_pred": [472.00235, 580.98920, 480.31976],
        }
    )

    return con.create_table("xgb_table", df)


@pytest.fixture
def prediction_expr(feature_table, float_model_path):
    predict_fn = xo.expr.ml.make_quickgrove_udf(float_model_path)
    return feature_table.mutate(pred=predict_fn.on_expr)


@pytest.fixture
def mixed_prediction_expr(mixed_feature_table, mixed_model_path):
    predict_fn = xo.expr.ml.make_quickgrove_udf(mixed_model_path)
    return mixed_feature_table.mutate(pred=predict_fn.on_expr)


@pytest.fixture
def tmp_model_dir(tmpdir):
    # Create a temporary directory for the model
    model_dir = tmpdir.mkdir("models")
    return model_dir
