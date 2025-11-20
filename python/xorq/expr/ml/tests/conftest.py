import shutil
from pathlib import Path

import pandas as pd
import pytest

import xorq.api as xo


FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
