import cloudpickle
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

import xorq.api as xo
from xorq.expr.ml.native.api import fit, predict
from xorq.expr.ml.native.tests.taxi import FEATURES, TARGET, make_pipeline


MODELS = (
    pytest.param(Ridge(alpha=1.0), id="ridge"),
    pytest.param(DecisionTreeRegressor(max_depth=8, random_state=0), id="tree"),
)


@pytest.fixture
def taxi(taxi_df):
    return xo.connect().create_table("taxi", taxi_df)


@pytest.mark.parametrize("model", MODELS)
def test_predict_matches_sklearn(taxi_df, taxi, model):
    expected = make_pipeline(model).fit(taxi_df[list(FEATURES)], taxi_df[TARGET])
    fitted = fit(taxi, FEATURES, TARGET, make_pipeline(model))

    actual = taxi.mutate(predicted=predict(fitted, taxi)).execute()["predicted"]

    np.testing.assert_array_equal(actual, expected.predict(taxi_df[list(FEATURES)]))


def test_fit_executes_to_fitted_estimator(taxi_df, taxi):
    fitted = fit(taxi, FEATURES, TARGET, make_pipeline(Ridge()))

    model = cloudpickle.loads(fitted.execute())

    np.testing.assert_array_equal(
        model.predict(taxi_df[list(FEATURES)]),
        make_pipeline(Ridge())
        .fit(taxi_df[list(FEATURES)], taxi_df[TARGET])
        .predict(taxi_df[list(FEATURES)]),
    )


def test_predict_on_filtered_rows(taxi_df, taxi):
    fitted = fit(taxi, FEATURES, TARGET, make_pipeline(Ridge()))
    subset = taxi_df[taxi_df.payment_type == "2"]

    filtered = taxi.filter(taxi.payment_type == "2")

    actual = filtered.mutate(predicted=predict(fitted, filtered)).execute()["predicted"]

    np.testing.assert_allclose(
        actual,
        make_pipeline(Ridge())
        .fit(taxi_df[list(FEATURES)], taxi_df[TARGET])
        .predict(subset[list(FEATURES)]),
    )


def test_regressor_on_integer_target_scores_floats(taxi_df):
    taxi = xo.connect().create_table(
        "taxi", taxi_df.assign(fare_amount=taxi_df.fare_amount.round().astype("int64"))
    )
    fitted = fit(taxi, FEATURES, TARGET, make_pipeline(Ridge()))

    actual = taxi.mutate(predicted=predict(fitted, taxi)).execute()["predicted"]

    assert actual.dtype == "float64" and (actual % 1 != 0).any()
