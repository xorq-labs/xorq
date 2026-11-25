from importlib.util import find_spec

import numpy as np
import pytest
from sklearn.linear_model import Ridge

import xorq.api as xo
from xorq.expr.ml.native.api import explain, fit, predict
from xorq.expr.ml.native.tests.taxi import FEATURES, TARGET, make_pipeline


@pytest.fixture
def predict_calls(monkeypatch):
    calls = []

    def counting(self, X, _predict=Ridge.predict):
        calls.append(len(X))
        return _predict(self, X)

    monkeypatch.setattr(Ridge, "predict", counting)
    return calls


def scored(taxi):
    fitted = fit(taxi, FEATURES, TARGET, make_pipeline(Ridge()))
    return taxi.mutate(predicted=predict(fitted, taxi))


def test_udf_scores_only_rows_passing_a_later_filter(
    monkeypatch, predict_calls, taxi_df
):
    monkeypatch.setattr(xo.options.ml, "rules", frozenset())
    expr = scored(xo.connect().create_table("taxi", taxi_df))

    actual = expr.filter(expr.hour < 6).execute()

    assert predict_calls == [len(actual)] and len(actual) < len(taxi_df)


@pytest.mark.skipif(find_spec("orbital") is None, reason="inlining needs orbital")
def test_inlined_filter_lands_in_the_scoring_select(taxi_df):
    expr = scored(xo.duckdb.connect().create_table("taxi", taxi_df))
    filtered = expr.filter(expr.hour < 6)

    sql = explain(filtered)

    assert sql.count("SELECT") == 1 and "WHERE" in sql
    np.testing.assert_allclose(
        filtered.execute()["predicted"],
        expr.execute().query("hour < 6")["predicted"],
    )
