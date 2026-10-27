import re
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

import xorq.api as xo
from xorq.expr.ml.native.api import explain, fit, predict
from xorq.expr.ml.native.tests.taxi import FEATURES, TARGET, make_pipeline


pytestmark = pytest.mark.skipif(
    find_spec("orbital") is None, reason="inlining needs orbital"
)
ROW_LEVEL_RULES = frozenset({"inline"})
# a per-group sum of the raw feature: the model was distributed over it
PUSHED_DOWN = re.compile(r'SUM\(\s*"t\d+"\."trip_distance"\s*\)')


@pytest.fixture
def taxi(taxi_df):
    with_nulls = taxi_df.assign(
        trip_distance=taxi_df.trip_distance.mask(taxi_df.index % 17 == 0)
    )
    return xo.duckdb.connect().create_table("taxi", with_nulls)


def per_hour(taxi, model, metrics, extra):
    fitted = fit(
        taxi.filter(taxi.trip_distance.notnull()),
        FEATURES,
        TARGET,
        make_pipeline(model),
    )
    expr = taxi.mutate(predicted=predict(fitted, taxi), **extra(taxi))
    return expr.group_by("hour").agg(**metrics(expr)).order_by("hour")


def at_row_level(monkeypatch, expr):
    with monkeypatch.context() as m:
        m.setattr(xo.options.ml, "rules", ROW_LEVEL_RULES)
        return expr.execute()


LINEAR = lambda e: {"mean": e.predicted.mean(), "total": e.predicted.sum()}  # noqa: E731
NONLINEAR = lambda e: {"top": e.predicted.max()}  # noqa: E731
NOTHING = lambda t: {}  # noqa: E731


def test_linear_model_mean_and_sum_push_below_scoring(monkeypatch, taxi):
    expr = per_hour(taxi, Ridge(), LINEAR, extra=NOTHING)

    actual = expr.execute()

    assert PUSHED_DOWN.search(explain(expr))
    expected = at_row_level(monkeypatch, expr)
    np.testing.assert_allclose(
        actual[["mean", "total"]], expected[["mean", "total"]], rtol=1e-9
    )


@pytest.mark.parametrize(
    "model,metrics",
    [
        pytest.param(
            DecisionTreeRegressor(max_depth=4, random_state=0), LINEAR, id="tree"
        ),
        pytest.param(Ridge(), NONLINEAR, id="max"),
    ],
)
def test_nonlinear_aggregates_stay_row_level(monkeypatch, taxi, model, metrics):
    expr = per_hour(taxi, model, metrics, extra=NOTHING)

    actual = expr.execute()

    assert not PUSHED_DOWN.search(explain(expr))
    expected = at_row_level(monkeypatch, expr)
    np.testing.assert_allclose(
        actual.drop(columns="hour"), expected.drop(columns="hour"), rtol=1e-9
    )


ALONGSIDE = (
    pytest.param(
        lambda e: {"total": e.predicted.sum(), "rows": e.count()}, NOTHING, id="count"
    ),
    pytest.param(
        lambda e: {"total": e.predicted.sum(), "m": e.predicted.cast("int64").mean()},
        NOTHING,
        id="int-cast",
    ),
    pytest.param(
        lambda e: {"total": e.predicted.sum(), "big": (-e.big).sum()},
        lambda t: {"big": t.hour.cast("int64") + 9007199254740993},
        id="exact-int",
    ),
    pytest.param(
        lambda e: {"total": e.predicted.sum(), "c": e.constant.sum()},
        lambda t: {"constant": xo.literal(1)},
        id="constant",
    ),
    pytest.param(
        lambda e: {"total": e.predicted.sum(), "halves": (e.hour // 2).mean()},
        NOTHING,
        id="floor-divide",
    ),
    pytest.param(
        lambda e: {
            "total": e.predicted.sum(),
            "rounded": (e.x.cast("float32") - 16777216.0).sum(),
        },
        lambda t: {"x": t.hour * 0 + 16777217.0},
        id="float32-cast",
    ),
    pytest.param(
        lambda e: {"total": e.predicted.sum(), "lagged": e.lagged.max()},
        lambda t: {"lagged": t.trip_distance.lag()},
        id="window",
    ),
)


@pytest.mark.parametrize("metrics,extra", ALONGSIDE)
def test_other_metrics_alongside_pushdown_keep_row_level_values(
    monkeypatch, taxi, metrics, extra
):
    expr = per_hour(taxi, Ridge(), metrics, extra)

    actual = expr.execute()

    pd.testing.assert_frame_equal(actual, at_row_level(monkeypatch, expr), rtol=1e-9)


def test_integer_arithmetic_alongside_pushdown_stays_exact(monkeypatch, taxi):
    wide = taxi.mutate(big=taxi.hour.cast("int64") * 0 + 9007199254740992)
    expr = per_hour(
        wide,
        Ridge(),
        lambda e: {
            "total": e.predicted.sum(),
            "gap": (e.big - 9007199254740993).mean(),
        },
        NOTHING,
    )

    actual = expr.execute()

    assert (actual.gap == -1).all()
    pd.testing.assert_frame_equal(actual, at_row_level(monkeypatch, expr), rtol=1e-9)
