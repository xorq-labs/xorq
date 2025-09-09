import pytest

import xorq.api as xo


sklearn = pytest.importorskip("sklearn")


TARGET = "target"


def test_infer_features():
    features = (feature0, feature1) = ("feature_0", "feature_1")
    t = xo.memtable(
        {
            feature0: [1, 2],
            feature1: [3, 4],
            TARGET: [0, 1],
        }
    )
    sklearn_pipeline = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        sklearn.linear_model.LinearRegression(),
    )
    xorq_pipeline = xo.Pipeline.from_instance(sklearn_pipeline)
    fitted_xorq_pipeline = xorq_pipeline.fit(t, target=TARGET)
    assert all(
        features == step.features for step in fitted_xorq_pipeline.transform_steps
    )
