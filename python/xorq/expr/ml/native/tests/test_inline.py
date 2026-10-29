import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

import xorq.api as xo
from xorq.expr.ml.native.inline import inline_pipeline
from xorq.expr.ml.native.tests.taxi import FEATURES, TARGET, make_pipeline


@pytest.mark.parametrize(
    "model,rtol",
    [
        pytest.param(Ridge(alpha=1.0), 1e-9, id="ridge"),
        # ONNX tree ensembles store leaf values as float32
        pytest.param(
            DecisionTreeRegressor(max_depth=8, random_state=0), 1e-6, id="tree"
        ),
    ],
)
def test_inline_pipeline_matches_sklearn(taxi_df, model, rtol):
    pipeline = make_pipeline(model).fit(taxi_df[list(FEATURES)], taxi_df[TARGET])
    table = xo.duckdb.connect().create_table("taxi", taxi_df)

    actual = inline_pipeline(pipeline, table, FEATURES).execute()["variable"]

    np.testing.assert_allclose(
        actual, pipeline.predict(taxi_df[list(FEATURES)]), rtol=rtol
    )
