import argparse
import functools

import pandas as pd
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
from xorq.common.utils.toolz_utils import curry
from xorq.expr.ml import (
    deferred_fit_predict,
    deferred_fit_transform_series_sklearn,
    train_test_splits,
)
from xorq.flight import (
    FlightServer,
    FlightUrl,
)
from xorq.flight.client import FlightClient


transform_port = 8765
predict_port = 8766
expected_transform_command = "execute-unbound-expr-d785a558027791af18dac689ed381d42"
expected_predict_command = "execute-unbound-expr-2f54734d557f2914929e8f0fc8784c42"


do_hackernews_fetcher_udxf = import_python(
    xo.options.pins.get_path("hackernews_lib", version="20250604T223424Z-2e578")
).do_hackernews_fetcher_udxf
do_hackernews_sentiment_udxf = import_python(
    xo.options.pins.get_path("openai_lib", version="20250604T223419Z-0ce44")
).do_hackernews_sentiment_udxf


@curry
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


@curry
def predict_xgboost_model(model, df):
    return model.predict(df.squeeze().tolist())


def make_outsample_expr():
    # simulated fetch live and predict
    z = (
        xo.memtable([{"maxitem": 43346282, "n": 1000}])
        .pipe(do_hackernews_fetcher_udxf)
        .filter(xo._.text.notnull())
    )
    return z


def get_confusion_matrix(df):
    return (
        df[[target, f"{target}_predicted"]]
        .astype(int)
        .value_counts()
        .unstack(f"{target}_predicted")
    )


SENTIMENT = "sentiment"
transform_col = "title"
target = f"{SENTIMENT}_int"
do_deferred_fit_transform_tfidf = deferred_fit_transform_series_sklearn(
    col=transform_col,
    cls=TfidfVectorizer,
    return_type=dt.Array(dt.float64),
    name="transform_tfidf",
)
do_deferred_fit_predict_xgb = deferred_fit_predict(
    target=target,
    features=[f"{transform_col}_transformed"],
    fit=fit_xgboost_model,
    predict=predict_xgboost_model,
    return_type=dt.float32,
    name="predict_xgb",
)
do_train_test_split = curry(
    train_test_splits,
    unique_key="id",
    test_sizes=(0.6, 0.4),
    random_seed=42,
)


def do_fit(expr, storage=None):
    (_, _, deferred_transform) = do_deferred_fit_transform_tfidf(
        expr,
        storage=storage,
    )
    transformed = expr.mutate(
        **{f"{transform_col}_transformed": deferred_transform.on_expr}
    )
    (_, _, deferred_predict) = do_deferred_fit_predict_xgb(
        transformed,
        storage=storage,
    )
    predicted = (
        transformed
        # into_backend prevents ArrowNotImplementedError: Unsupported cast
        .into_backend(xo.connect()).mutate(
            **{f"{target}_predicted": deferred_predict.on_expr}
        )
    )
    return (predicted, deferred_transform, deferred_predict)


def do_transform_predict(expr, deferred_transform, deferred_predict):
    return (
        expr.mutate(**{f"{transform_col}_transformed": deferred_transform.on_expr})
        # into_backend prevents ArrowNotImplementedError: Unsupported cast
        .into_backend(xo.connect())
        .mutate(**{f"{target}_predicted": deferred_predict.on_expr})
    )


def do_serve(expr, deferred_transform, deferred_predict, transform_port, predict_port):
    transformed = expr.into_backend(xo.connect()).mutate(
        **{f"{transform_col}_transformed": deferred_transform.on_expr},
    )
    predicted = transformed.into_backend(xo.connect()).mutate(
        **{f"{target}_predicted": deferred_predict.on_expr},
    )
    (transform_server, transform_do_exchange) = xo.expr.relations.flight_serve(
        transformed,
        make_server=functools.partial(FlightServer, FlightUrl(port=transform_port)),
    )
    (predict_server, predict_do_exchange) = xo.expr.relations.flight_serve(
        predicted,
        make_server=functools.partial(FlightServer, FlightUrl(port=predict_port)),
    )
    return (
        transform_server,
        transform_do_exchange,
        predict_server,
        predict_do_exchange,
    )


def make_exprs():
    name = "hn-fetcher-input-large"
    con = xo.connect()
    storage = ParquetStorage(source=con)
    # pg.postgres.connect_env().create_catalog("caching")
    pg = xo.postgres.connect_env(database="caching")

    (train_expr, test_expr) = (
        deferred_read_parquet(
            xo.options.pins.get_path(name),
            con,
            name,
        )
        .pipe(do_hackernews_fetcher_udxf)
        .filter(xo._.text.notnull())
        .cache(storage=SourceStorage(pg))
        .pipe(do_hackernews_sentiment_udxf, con=con)
        .cache(storage=SourceStorage(pg))
        .cache(storage=ParquetStorage(con))
        .filter(~xo._[SENTIMENT].contains("ERROR"))
        .mutate(
            **{
                target: (
                    xo._[SENTIMENT]
                    .cases(
                        (
                            ("POSITIVE", 2),
                            ("NEUTRAL", 1),
                            ("NEGATIVE", 0),
                        )
                    )
                    .cast(int)
                ),
            }
        )
        .pipe(do_train_test_split)
    )
    (train_predicted, deferred_transform, deferred_predict) = do_fit(
        train_expr, storage
    )
    test_predicted = do_transform_predict(
        test_expr, deferred_transform, deferred_predict
    )
    return (train_predicted, test_predicted, deferred_transform, deferred_predict)


def run():
    (train_predicted, test_predicted, deferred_transform, deferred_predict) = (
        make_exprs()
    )
    print(train_predicted.execute().pipe(get_confusion_matrix))
    print(test_predicted.execute().pipe(get_confusion_matrix))
    return (train_predicted, test_predicted, deferred_transform, deferred_predict)


def serve(transform_port, predict_port):
    (_, _, deferred_transform, deferred_predict) = make_exprs()
    z = make_outsample_expr()
    (transform_server, transform_do_exchange, predict_server, predict_do_exchange) = (
        do_serve(z, deferred_transform, deferred_predict, transform_port, predict_port)
    )
    validate_commands(transform_do_exchange, predict_do_exchange)
    return (
        transform_server,
        transform_do_exchange,
        predict_server,
        predict_do_exchange,
    )


def predict(transform_port, predict_port):
    transform_client = FlightClient(port=transform_port)
    transform_do_exchange = curry(
        transform_client.do_exchange, expected_transform_command
    )
    predict_client = FlightClient(port=predict_port)
    predict_do_exchange = curry(predict_client.do_exchange, expected_predict_command)
    predicted = predict_do_exchange(
        transform_do_exchange(make_outsample_expr().to_pyarrow_batches())[1]
    )[1].read_pandas()
    print(predicted)
    return predicted


def validate_commands(transform_do_exchange, predict_do_exchange):
    (transform_command, predict_command) = (
        do_exchange.args[1]
        for do_exchange in (transform_do_exchange, predict_do_exchange)
    )
    assert transform_command == expected_transform_command
    assert predict_command == expected_predict_command


def parse_args(override=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "serve", "predict"))
    parser.add_argument("-t", "--transform-port", default=transform_port, type=int)
    parser.add_argument("-p", "--predict-port", default=predict_port, type=int)
    args = parser.parse_args(override)
    return args


if __name__ == "__main__":
    args = parse_args()
    match args.command:
        case "run":
            (train_predicted, test_predicted, deferred_transform, deferred_predict) = (
                run()
            )
        case "serve":
            (
                transform_server,
                transform_do_exchange,
                predict_server,
                predict_do_exchange,
            ) = serve(args.transform_port, args.predict_port)
        case "predict":
            predicted = predict(args.transform_port, args.predict_port)
        case _:
            raise ValueError
