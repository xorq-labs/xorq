"""
Chains a TF-IDF transform step with an XGBoost regression predict step into a FittedPipeline over HackerNews data.

Traditional approach: You build an sklearn Pipeline and serve it via Flask or FastAPI,
manually serializing models with pickle or joblib, writing custom request/response
handlers, and managing the serving infrastructure yourself.

With xorq: Chain deferred_fit_transform and deferred_fit_predict as Step objects in a
FittedPipeline. Each step's model is automatically cached via ParquetCache, and the
pipeline's predict method composes all transformations into a single deferred expression
that can be executed on train or test data interchangeably.
"""

import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error

import xorq.api as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import ParquetCache
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.import_utils import import_python
from xorq.expr.ml.pipeline_lib import (
    FittedPipeline,
    Step,
)
from xorq.ml import (
    train_test_splits,
)


m = import_python(
    xo.options.pins.get_path("hackernews_lib", version="20250820T111457Z-1d66a")
)


def fit_xgboost_model(feature_df, target_series, seed=0):
    xgb_r = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric=mean_absolute_error,
        n_estimators=20,
        seed=seed,
    )
    xgb_r.fit(feature_df, target_series)
    return xgb_r


def predict_xgboost_model(model, df):
    return model.predict(df)


def make_splits(con):
    train_expr, test_expr = (
        deferred_read_parquet(
            xo.options.pins.get_path("hn-fetcher-input-small.parquet"),
            con,
            "fetcher-input",
        )
        # we still need to set inner_name, else we get unstable hash
        .pipe(m.do_hackernews_fetcher_udxf, inner_name="inner-named-flight-udxf")
        .pipe(
            train_test_splits,
            unique_key="id",
            test_sizes=(0.9, 0.1),
            random_seed=0,
        )
    )
    return train_expr, test_expr


def make_pipeline(
    train_expr, test_expr, transform_col, target, target_predicted, cache=None
):
    predict_features = (transformed_col,) = (f"{transform_col}_transformed",)
    transform_step = Step(TfidfVectorizer)
    predict_step = Step.from_fit_predict(
        fit=fit_xgboost_model,
        predict=predict_xgboost_model,
        return_type=dt.float64,
    )
    fitted_transform = transform_step.fit(
        train_expr,
        features=(transform_col,),
        dest_col=transformed_col,
        cache=cache,
    )
    fitted_predict = predict_step.fit(
        fitted_transform.transformed,
        features=predict_features,
        target=target,
        dest_col=target_predicted,
        cache=cache,
    )
    fitted_pipeline = FittedPipeline((fitted_transform, fitted_predict), train_expr)
    test_predicted = fitted_pipeline.predict(test_expr)
    return fitted_pipeline, test_predicted


transform_col = "title"
target = "descendants"
target_predicted = f"{target}_predicted"
con = xo.connect()
cache = ParquetCache.from_kwargs(source=con)
(train_expr, test_expr) = make_splits(con)
fitted_pipeline, test_predicted = make_pipeline(
    train_expr, test_expr, transform_col, target, target_predicted, cache
)


if __name__ == "__pytest_main__":
    fitted_transform, fitted_predict = fitted_pipeline.fitted_steps
    print(
        fitted_transform.deferred_model.ls.get_key(),
        fitted_transform.deferred_model.ls.exists(),
    )
    print(
        fitted_predict.deferred_model.ls.get_key(),
        fitted_predict.deferred_model.ls.exists(),
    )

    # EXECUTION
    df = fitted_pipeline.predict(train_expr).execute()
    df2 = fitted_pipeline.predict(test_expr).execute()
    print(df[[target, target_predicted]].corr())
    print(df2[[target, target_predicted]].corr())
    pytest_examples_passed = True
