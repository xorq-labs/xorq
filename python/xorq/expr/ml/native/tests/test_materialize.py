import numpy as np
import pytest
from sklearn.linear_model import Ridge

import xorq.api as xo
from xorq.expr.ml.native.api import explain, fit, predict
from xorq.expr.ml.native.tests.taxi import FEATURES, TARGET, make_pipeline


FIT_CALLS = []


class CountingRidge(Ridge):
    def fit(self, X, y, sample_weight=None):
        FIT_CALLS.append(len(X))
        return super().fit(X, y, sample_weight)


@pytest.fixture
def taxi(taxi_df):
    return xo.connect().create_table("taxi", taxi_df)


@pytest.fixture
def expected(taxi_df):
    return (
        make_pipeline(Ridge())
        .fit(taxi_df[list(FEATURES)], taxi_df[TARGET])
        .predict(taxi_df[list(FEATURES)])
    )


def scored(taxi, model):
    fitted = fit(taxi, FEATURES, TARGET, make_pipeline(model))
    return taxi.mutate(predicted=predict(fitted, taxi))


def test_fit_runs_once_across_executions(taxi):
    FIT_CALLS.clear()
    expr = scored(taxi, CountingRidge(alpha=0.5))

    first, second = (expr.execute()["predicted"] for _ in range(2))

    np.testing.assert_array_equal(first, second)
    assert len(FIT_CALLS) == 1


@pytest.mark.parametrize("rewrite", [True, False], ids=["rewrite", "opt-out"])
def test_rewrite_flag_keeps_predictions(monkeypatch, taxi, expected, rewrite):
    monkeypatch.setattr(xo.options.ml, "rewrite", rewrite)

    actual = scored(taxi, Ridge()).execute()["predicted"]

    np.testing.assert_allclose(actual, expected)


def test_explain_fits_and_execute_reuses_the_model(taxi):
    FIT_CALLS.clear()
    expr = scored(taxi, CountingRidge(alpha=0.25))

    sql = explain(expr)
    expr.execute()

    assert "SELECT" in sql
    assert len(FIT_CALLS) == 1
