"""
Runs linear regression fit/predict with deferred execution and caching, comparing uncached, cached, Step-based, and pinned APIs.

Traditional approach: You call sklearn's LinearRegression.fit() then .predict(),
manually managing train/test splits, saving models with pickle, and wiring prediction
arrays back into your DataFrame. Each of these steps is imperative and tightly coupled.

With Xorq: deferred_fit_predict wraps fit/predict in a deferred expression, so
predictions become composable Ibis columns you can .mutate() onto any table. Automatic
caching via ParquetCache means repeated runs are free, and the Step API provides a
higher-level interface for the same workflow.

Pinning goes one step further. A cached fit still lives outside the graph, so a
cache miss silently retrains and shipping the pipeline ships the training. `pin_caches`
(here via `expr.ls.pin_caches()`) executes the cached fit once and swaps it for a
deferred read on the cache parquet, tagged with its provenance, so the fitted model
becomes part of the expression and never retrains. `xorq build --pin` packs that
parquet into the artifact, so `xorq run` anywhere reproduces the exact model.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

import xorq.api as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import ParquetCache
from xorq.expr.ml.pipeline_lib import Step
from xorq.ml import deferred_fit_predict_sklearn


def make_data():
    import numpy as np  # noqa: PLC0415

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    df = pd.DataFrame(np.hstack((X, y[:, np.newaxis]))).rename(
        columns=lambda x: chr(x + ord("a"))
    )
    (*features, target) = df.columns
    return (df, features, target)


deferred_linear_regression = deferred_fit_predict_sklearn(
    cls=LinearRegression, return_type=dt.float64
)
step = Step(typ=LinearRegression)


con = xo.connect()
cache = ParquetCache.from_kwargs(source=con)
(df, features, target) = make_data()
t = con.register(df, "t")
kwargs = {
    "expr": t,
    "target": target,
    "features": features,
}


# uncached run
(deferred_model, model_udaf, predict) = deferred_linear_regression(
    **kwargs
).deferred_model_udaf_other
predicted = t.mutate(predict.on_expr(t).name("predicted"))


# cached run
(cached_deferred_model, cached_model_udaf, cached_predict) = deferred_linear_regression(
    cache=cache,
    **kwargs,
).deferred_model_udaf_other
cached_predicted = t.mutate(cached_predict.on_expr(t).name("predicted"))

# as step
fitted_step = step.fit(cache=cache, **kwargs)
step_predicted = t.mutate(fitted_step.predict_raw(t, name="predicted"))

# pinned run: execute the cached fit once and pin its parquet into the DAG, so
# the model ships with the expression and never retrains
pinned_predicted = cached_predicted.ls.pin_caches()


if __name__ == "__pytest_main__":
    # model = deferred_model.execute()
    # ((cached_model,),) = cached_deferred_model.execute().values
    predicted_df = predicted.execute()
    cached_predicted_df = cached_predicted.execute()
    step_predicted_df = step_predicted.execute()
    pinned_predicted_df = pinned_predicted.execute()
    assert predicted_df.equals(cached_predicted_df)
    assert predicted_df.equals(step_predicted_df)
    assert predicted_df.equals(pinned_predicted_df)
    # the cached fit is now a pinned read: verify each pinned parquet still
    # matches the content hash recorded when it was pinned
    assert pinned_predicted.ls.pinned_caches
    assert all(verification.ok for verification in pinned_predicted.ls.verify_pinned())
    pytest_examples_passed = True
