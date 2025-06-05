import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error

import xorq as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import ParquetStorage
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.import_utils import import_python
from xorq.common.utils.toolz_utils import curry
from xorq.ml import (
    deferred_fit_predict,
    deferred_fit_transform_series_sklearn,
    train_test_splits,
)


m = import_python(
    xo.options.pins.get_path("hackernews_lib", version="20250604T223424Z-2e578")
)


@curry
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


@curry
def predict_xgboost_model(model, df):
    return model.predict(df.squeeze().tolist())


transform_col = "title"
features = (transformed_col,) = (f"{transform_col}_transformed",)
target = "descendants"
target_predicted = f"{target}_predicted"
deferred_fit_transform_tfidf = deferred_fit_transform_series_sklearn(
    col=transform_col,
    cls=TfidfVectorizer,
    return_type=dt.Array(dt.float64),
)
deferred_fit_predict_xgb = deferred_fit_predict(
    target=target,
    features=list(features),
    fit=fit_xgboost_model,
    predict=predict_xgboost_model,
    return_type=dt.float32,
)


con = xo.connect()
storage = ParquetStorage(source=con)
# storage = None


train_expr, test_expr = (
    deferred_read_parquet(
        con,
        xo.options.pins.get_path("hn-fetcher-input-small.parquet"),
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


# fit-transform
(deferred_tfidf_model, tfidf_udaf, deferred_tfidf_transform) = (
    deferred_fit_transform_tfidf(
        train_expr,
        storage=storage,
    )
)
train_tfidf_transformed = train_expr.mutate(
    **{transformed_col: deferred_tfidf_transform.on_expr}
)
# fit-predict
(deferred_xgb_model, xgb_udaf, deferred_xgb_predict) = deferred_fit_predict_xgb(
    train_tfidf_transformed,
    storage=storage,
)
train_xgb_predicted = (
    train_tfidf_transformed
    # if i add into backend here, i don't get ArrowNotImplementedError: Unsupported cast
    .into_backend(xo.connect()).mutate(
        **{target_predicted: deferred_xgb_predict.on_expr}
    )
)


# now we can define test pathway
test_xgb_predicted = (
    test_expr.mutate(**{transformed_col: deferred_tfidf_transform.on_expr})
    # if i add into backend here, i don't get ArrowNotImplementedError: Unsupported cast
    .into_backend(xo.connect())
    .mutate(**{target_predicted: deferred_xgb_predict.on_expr})
)


if __name__ == "__pytest_main__":
    print(deferred_tfidf_model.ls.get_key(), deferred_tfidf_model.ls.exists())
    print(deferred_xgb_model.ls.get_key(), deferred_xgb_model.ls.exists())

    # EXECUTION
    df = train_xgb_predicted.execute()
    df2 = test_xgb_predicted.execute()
    print(df[[target, target_predicted]].corr())
    print(df2[[target, target_predicted]].corr())
    pytest_examples_passed = True
