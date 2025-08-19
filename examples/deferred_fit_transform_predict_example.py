import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error

import xorq.api as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import ParquetStorage
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
    xo.options.pins.get_path("hackernews_lib", version="20250604T223424Z-2e578")
)


def fit_xgboost_model(feature_df, target_series, seed=0):
    xgb_r = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric=mean_absolute_error,
        # max_depth=10,
        # learning_rate=1,
        n_estimators=20,
        seed=seed,
    )
    X = pd.DataFrame(feature_df.squeeze().tolist())
    xgb_r.fit(X, target_series)
    return xgb_r


def predict_xgboost_model(model, df):
    return model.predict(df.squeeze().tolist())


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
    train_expr, test_expr, transform_col, target, target_predicted, storage=None
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
        storage=storage,
    )
    fitted_predict = predict_step.fit(
        fitted_transform.transformed,
        features=predict_features,
        target=target,
        dest_col=target_predicted,
        storage=storage,
    )
    fitted_pipeline = FittedPipeline((fitted_transform, fitted_predict), train_expr)
    test_predicted = fitted_pipeline.predict(test_expr)
    return fitted_pipeline, test_predicted


transform_col = "title"
target = "descendants"
target_predicted = f"{target}_predicted"
con = xo.connect()
storage = ParquetStorage(source=con)
(train_expr, test_expr) = make_splits(con)
fitted_pipeline, test_predicted = make_pipeline(
    train_expr, test_expr, transform_col, target, target_predicted, storage
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
