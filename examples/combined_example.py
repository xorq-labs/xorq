import pandas as pd
import toolz
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error

import xorq as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import (
    ParquetStorage,
    SourceStorage,
)
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.import_utils import import_python
from xorq.expr.ml import (
    deferred_fit_predict,
    deferred_fit_transform_series_sklearn,
    train_test_splits,
)


m = import_python(xo.options.pins.get_path("hackernews_lib"))
o = import_python("/home/dan/repos/github/xorq/examples/openai_lib.py")


@toolz.curry
def fit_xgboost_model(feature_df, target_series, seed=0):
    xgb_r = xgb.XGBRegressor(
        objective="multi:softmax",
        num_class=3,
        eval_metric=mean_absolute_error,
        max_depth=6,
        # learning_rate=1,
        n_estimators=10,
        seed=seed,
    )
    X = pd.DataFrame(feature_df.squeeze().tolist())
    xgb_r.fit(X, target_series)
    return xgb_r


@toolz.curry
def predict_xgboost_model(model, df):
    return model.predict(df.squeeze().tolist())


transform_col = "title"
features = (transformed_col,) = (f"{transform_col}_transformed",)
target = "sentiment_int"
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


name = "hn-fetcher-input-large"
con = xo.config._backend_init()
storage = ParquetStorage(source=con)
# pg.postgres.connect_env().create_catalog("caching")
pg = xo.postgres.connect_env(database="caching")
t = (
    deferred_read_parquet(
        con,
        xo.options.pins.get_path(name),
        name,
    )
    .pipe(m.do_hackernews_fetcher_udxf)
    .filter(xo._.text.notnull())
    .cache(storage=SourceStorage(pg))
    # .limit(100)
    .pipe(o.do_hackernews_sentiment_udxf, con=con)
    # commenting out this cache changes the hash of the subsequent hash
    .cache(storage=SourceStorage(pg))
    .cache(storage=ParquetStorage(con))
    .filter(~xo._.sentiment.contains("ERROR"))
    .mutate(
        sentiment_int=xo._.sentiment.cases(
            {"POSITIVE": 2, "NEUTRAL": 1, "NEGATIVE": 0}.items()
        ).cast(int)
    )
    # .mutate(sentiment_int=xo._.sentiment.cases({"POSITIVE": 1, "NEUTRAL": 1, "NEGATIVE": 0}.items()).cast(int))
)
(train_expr, test_expr) = t.pipe(
    train_test_splits,
    unique_key="id",
    test_sizes=(0.6, 0.4),
    random_seed=42,
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


x = train_xgb_predicted.execute()
y = test_xgb_predicted.execute()
print(x.groupby("sentiment_int").sentiment_int_predicted.describe().T)
print(y.groupby("sentiment_int").sentiment_int_predicted.describe().T)
