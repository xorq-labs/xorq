"""
HackerNews Sentiment Analysis Script This script loads HackerNews data,
analyzes post titles using a pre-trained TF-IDF transformer, and predicts
sentiment scores using another pre-trained XGBoost model.
"""

import pathlib
import pickle

import pandas as pd
import toolz
import xgboost as xgb

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.import_utils import import_python
from xorq.flight import FlightServer
from xorq.flight.exchanger import make_udxf


# paths
TFIDF_MODEL_PATH = pathlib.Path(xo.options.pins.get_path("hn_tfidf_fitted_model"))
XGB_MODEL_PATH = pathlib.Path(xo.options.pins.get_path("hn_sentiment_reg"))

HACKERNEWS_DATA_NAME = "hn-fetcher-input-small"

# import HackerNews library from pinned path
hackernews_lib = import_python(
    xo.options.pins.get_path("hackernews_lib", version="20250604T223424Z-2e578")
)


def load_models():
    transformer = pickle.loads(TFIDF_MODEL_PATH.read_bytes())

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)

    return transformer, xgb_model


def predict_sentiment(titles):
    transformer, xgb_model = load_models()
    return xgb_model.predict(transformer.transform(titles))


schema_in = xo.schema({"title": str})
schema_out = xo.schema({"sentiment_score": dt.double})


@xo.udf.make_pandas_udf(
    schema=xo.schema({"title": str}),
    return_type=dt.float64,
    name="title_transformed",
)
def transform_predict(df):
    return predict_sentiment(df["title"])


# For Flight server
def sentiment_analysis(df: pd.DataFrame):
    scores = predict_sentiment(df["title"])
    return pd.DataFrame({"sentiment_score": [float(scores)]})


def test_flight_service(do_sentiment, schema_in):
    test_data = xo.memtable(
        {"title": ["This is an amazing HackerNews post"]}, schema=schema_in
    )
    (_, rbr) = do_sentiment(test_data)
    res = rbr.read_pandas()
    print("Flight service test result:\n", res)


# connect to xorq's embedded engine
connection = xo.connect()

pipeline = (
    deferred_read_parquet(
        xo.options.pins.get_path(HACKERNEWS_DATA_NAME),
        connection,
        HACKERNEWS_DATA_NAME,
    )
    # process with HackerNews fetcher Exchanger that does a live fetch
    .pipe(hackernews_lib.do_hackernews_fetcher_udxf)
    # select only the title column
    .select(xo._.title)
    # add sentiment score prediction
    .mutate(sentiment_score=transform_predict.on_expr)
)

# Create the UDXF for Flight server
sentiment_udxf = make_udxf(
    sentiment_analysis,
    schema_in,
    schema_out,
)


if __name__ == "__pytest_main__":
    # demonstrate pipeline
    results = pipeline.execute()

    # Start the Flight server with our exchanger
    flight_server = FlightServer(exchangers=[sentiment_udxf])
    flight_server.serve()

    # Get a client to test the server
    client = flight_server.client
    do_sentiment = toolz.curry(client.do_exchange, sentiment_udxf.command)

    print("Testing Flight service...")
    test_flight_service(do_sentiment, schema_in)
    flight_server.close()
    pytest_examples_passed = True

"""
Next Steps: use the cli to build and see how things look like:

❯ xorq build scripts/hn_inference.py -e pipeline
Building pipeline from scripts/hn_inference.py
/nix/store/i7dqrcpgqll387lx48mfnhxq6nw5j1nb-xorq/lib/python3.10/site-packages/xgboost/core.py:265: FutureWarning: Your system has an old version of glibc (< 2.28). We will stop supporting Linux distros with glibc older than 2.28 after **May 31, 2025**. Please upgrade to a recent Linux distro (with glibc 2.28+) to use future versions of XGBoost.
Note: You have installed the 'manylinux2014' variant of XGBoost. Certain features such as GPU algorithms or federated learning are not available. To use these features, please upgrade to a recent Linux distro with glibc 2.28+, and install the 'manylinux_2_28' variant.
  warnings.warn(
Written 'pipeline' to builds/36293178ec4f

> xorq serve (coming soon)
"""
