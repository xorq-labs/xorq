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


def expected_tags(keys=None):
    ds = (
        {
            "tag": "predict",
            "predict": (
                ("copy_X", True),
                ("fit_intercept", True),
                ("n_jobs", None),
                ("positive", False),
                ("tol", 1e-06),
            ),
        },
        {
            "tag": "transform",
            "transform": ("standardscaler",),
        },
    )
    return tuple(d for d in ds if d.keys() & keys)


@pytest.mark.parametrize(
    "keys",
    (
        ("transform",),
        (
            "transform",
            "predict",
        ),
        ("predict",),
    ),
)
def test_tagging_pipeline(keys):
    (feature0, feature1) = ("feature_0", "feature_1")
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

    def filter_by_keys(d):
        if not keys:
            return True
        else:
            return d.keys() & tuple(keys)

    assert tuple(
        value
        for _, value in fitted_xorq_pipeline.predict(t).ls.tags(
            predicate=filter_by_keys,
            with_metadata=True,
        )
    ) == expected_tags(keys)
