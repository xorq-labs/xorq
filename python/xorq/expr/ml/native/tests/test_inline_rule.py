from importlib.util import find_spec

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

import xorq.api as xo
from xorq.expr.ml.native.api import explain, fit, predict
from xorq.expr.ml.native.tests.taxi import FEATURES, TARGET, make_pipeline


pytestmark = pytest.mark.skipif(
    find_spec("orbital") is None, reason="inlining needs orbital"
)


@pytest.fixture
def predict_calls(monkeypatch):
    """Rows each estimator class scores in Python."""
    calls = []
    for cls in (Ridge, DecisionTreeRegressor):

        def counting(self, X, *args, _predict=cls.predict, **kwargs):
            calls.append(len(X))
            return _predict(self, X, *args, **kwargs)

        monkeypatch.setattr(cls, "predict", counting)
    return calls


MODELS = (
    pytest.param(Ridge(alpha=1.0), 1e-9, id="ridge"),
    # ONNX tree ensembles store leaf values as float32
    pytest.param(DecisionTreeRegressor(max_depth=8, random_state=0), 1e-6, id="tree"),
)
BACKENDS = (
    pytest.param(xo.connect, id="datafusion"),
    pytest.param(lambda: xo.duckdb.connect(), id="duckdb"),
)


def scored(taxi, model):
    fitted = fit(taxi, FEATURES, TARGET, make_pipeline(model))
    return taxi.mutate(predicted=predict(fitted, taxi))


@pytest.mark.parametrize("connect", BACKENDS)
@pytest.mark.parametrize("model,rtol", MODELS)
def test_inlined_scoring_skips_python(taxi_df, predict_calls, connect, model, rtol):
    taxi = connect().create_table("taxi", taxi_df)
    expected = make_pipeline(model).fit(taxi_df[list(FEATURES)], taxi_df[TARGET])
    expected = expected.predict(taxi_df[list(FEATURES)])
    predict_calls.clear()

    actual = scored(taxi, model).execute()["predicted"]

    np.testing.assert_allclose(actual, expected, rtol=rtol)
    assert predict_calls == []


def test_inline_rule_off_scores_in_python(monkeypatch, predict_calls, taxi_df):
    monkeypatch.setattr(xo.options.ml, "rules", frozenset())
    taxi = xo.connect().create_table("taxi", taxi_df)
    scored(taxi, Ridge()).execute()

    assert predict_calls == [len(taxi_df)]


def test_explain_shows_inlined_model(taxi_df):
    taxi = xo.duckdb.connect().create_table("taxi", taxi_df)

    sql = explain(scored(taxi, Ridge()))

    assert "CASE WHEN" in sql and "payment_type" in sql


# these fall back to the pandas UDF, which runs on datafusion only
def test_constant_model_scores_every_row(taxi_df):
    taxi = xo.connect().create_table("taxi", taxi_df.assign(fare_amount=7.5))
    fitted = fit(taxi, FEATURES, TARGET, make_pipeline(DecisionTreeRegressor()))

    actual = predict(fitted, taxi).execute()

    np.testing.assert_array_equal(actual, np.full(len(taxi_df), 7.5))


def test_predict_proba_returns_class_probabilities(taxi_df):
    labelled = taxi_df.assign(expensive=(taxi_df.fare_amount > 10).astype(str))
    taxi = xo.connect().create_table("taxi", labelled)
    model = make_pipeline(LogisticRegression())
    fitted = fit(taxi, FEATURES, "expensive", model)

    actual = taxi.mutate(p=predict(fitted, taxi, method="predict_proba")).execute()

    expected = model.fit(labelled[list(FEATURES)], labelled.expensive)
    np.testing.assert_allclose(
        np.stack(actual.p), expected.predict_proba(labelled[list(FEATURES)])
    )


def test_mixed_numeric_feature_types_still_score(taxi_df):
    mixed = taxi_df.assign(passenger_count=taxi_df.passenger_count.astype("int64"))
    taxi = xo.connect().create_table("taxi", mixed)
    numeric = ["trip_distance", "passenger_count"]
    fitted = fit(taxi, numeric, TARGET, Ridge())

    actual = predict(fitted, taxi).execute()

    expected = Ridge().fit(mixed[numeric], mixed[TARGET]).predict(mixed[numeric])
    np.testing.assert_allclose(actual, expected)
