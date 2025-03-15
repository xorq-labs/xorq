import functools

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
from xorq.flight import (
    FlightServer,
    FlightUrl,
)


m = import_python(xo.options.pins.get_path("hackernews_lib"))
o = import_python(xo.options.pins.get_path("openai_lib"))


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


do_hackernews_fetcher_udxf = xo.expr.relations.flight_udxf(
    process_df=m.get_hackernews_stories_batch,
    # process_df=get_hackernews_stories_batch,
    maybe_schema_in=m.schema_in.to_pyarrow(),
    maybe_schema_out=m.schema_out.to_pyarrow(),
    name="HackerNewsFetcher",
)


name = "hn-fetcher-input-large"
con = xo.config._backend_init()
storage = ParquetStorage(source=con)
# pg.postgres.connect_env().create_catalog("caching")
pg = xo.postgres.connect_env(database="caching")
raw_expr = (
    deferred_read_parquet(
        con,
        xo.options.pins.get_path(name),
        name,
    )
    # .pipe(do_hackernews_fetcher_udxf)
    .pipe(m.do_hackernews_fetcher_udxf)
)
t = (
    raw_expr
    # most stories have a tile, but few have text
    # df.groupby("type").apply(lambda t: t.notnull().sum().div(len(t)))
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
    # why is this stable-name required?
    .into_backend(xo.connect(), name="stable-name")
    .mutate(**{target_predicted: deferred_xgb_predict.on_expr})
)


x = train_xgb_predicted.execute()
y = test_xgb_predicted.execute()
print(x.groupby("sentiment_int").sentiment_int_predicted.describe().T)
print(y.groupby("sentiment_int").sentiment_int_predicted.describe().T)


# fetch live and predict
z = (
    xo.memtable([{"maxitem": 43346282, "n": 1000}])
    .pipe(m.do_hackernews_fetcher_udxf)
    .filter(xo._.text.notnull())
    .mutate(
        **{
            "sentiment": xo.literal(None).cast(str),
            "sentiment_int": xo.literal(None).cast(int),
        }
    )
    #     .mutate(**{transformed_col: deferred_tfidf_transform.on_expr})
)


(transform_server, transform_do_exchange) = xo.expr.relations.flight_serve(
    # why is this stable-name required?
    test_expr.into_backend(xo.connect(), name="stable-name").mutate(
        **{transformed_col: deferred_tfidf_transform.on_expr}
    ),
    make_server=functools.partial(FlightServer, FlightUrl(port=8765)),
)
(predict_server, predict_do_exchange) = xo.expr.relations.flight_serve(
    test_xgb_predicted,
    make_server=functools.partial(FlightServer, FlightUrl(port=8766)),
)
(transform_command, predict_command) = (
    do_exchange.args[1] for do_exchange in (transform_do_exchange, predict_do_exchange)
)
# issue: do_exchange here takes expr, externally it takes RecordBatchReader
out = predict_do_exchange(xo.register(transform_do_exchange(z), "t")).read_pandas()

expected_transform_command = "execute-unbound-expr-4de287288267a2b06a16c7d8bcc2011b"
expected_predict_commnd = "execute-unbound-expr-d9ab9b66dc4a3beafe4de74f66b4edb9"
assert (transform_server.flight_url.port, transform_command) == (
    8765,
    expected_transform_command,
), (transform_command, expected_transform_command)
assert (predict_server.flight_url.port, predict_command) == (
    8766,
    expected_predict_commnd,
), (predict_command, expected_predict_commnd)
