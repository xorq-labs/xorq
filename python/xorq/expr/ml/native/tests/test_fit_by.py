import cloudpickle
import numpy as np
import pytest
from sklearn.linear_model import Ridge

import xorq.api as xo
from xorq.expr.ml.native.api import fit_by
from xorq.expr.ml.native.tests.taxi import FEATURES, TARGET, make_pipeline


CASES = (
    pytest.param(xo.connect, True, id="datafusion-rewrite"),
    pytest.param(xo.connect, False, id="datafusion-opt-out"),
    pytest.param(lambda: xo.duckdb.connect(), True, id="duckdb-rewrite"),
)


@pytest.mark.parametrize("connect,rewrite", CASES)
def test_fit_by_fits_one_model_per_group(monkeypatch, taxi_df, connect, rewrite):
    monkeypatch.setattr(xo.options.ml, "rewrite", rewrite)
    taxi = connect().create_table("taxi", taxi_df)

    models = fit_by(taxi, "payment_type", FEATURES, TARGET, make_pipeline(Ridge()))
    actual = models.execute().set_index("payment_type")["model"]

    assert sorted(actual.index) == sorted(taxi_df.payment_type.unique())
    for key, group in taxi_df.groupby("payment_type"):
        expected = make_pipeline(Ridge()).fit(group[list(FEATURES)], group[TARGET])
        np.testing.assert_allclose(
            cloudpickle.loads(actual[key]).predict(group[list(FEATURES)]),
            expected.predict(group[list(FEATURES)]),
        )


def test_fit_by_rejects_computed_keys(taxi_df):
    taxi = xo.duckdb.connect().create_table("taxi", taxi_df)

    with pytest.raises(TypeError, match="column names"):
        fit_by(taxi, taxi.payment_type.name("p"), FEATURES, TARGET, Ridge())
