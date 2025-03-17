"""
HackerNews Sentiment Analysis Script This script loads HackerNews data,
analyzes post titles using a pre-trained TF-IDF transformer, and predicts
sentiment scores using another pre-trained XGBoost model.
"""

import pathlib
import pickle

import xgboost as xgb

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.import_utils import import_python


# paths
TFIDF_MODEL_PATH = pathlib.Path(xo.options.pins.get_path("hn_tfidf_fitted_model"))
XGB_MODEL_PATH = pathlib.Path(xo.options.pins.get_path("hn_sentiment_reg"))

HACKERNEWS_DATA_NAME = "hn-fetcher-input-small"

# import HackerNews library from pinned path
hackernews_lib = import_python(xo.options.pins.get_path("hackernews_lib"))


def load_models():
    transformer = pickle.loads(TFIDF_MODEL_PATH.read_bytes())

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)

    return transformer, xgb_model


@xo.udf.make_pandas_udf(
    schema=xo.schema({"title": str}),
    return_type=dt.float64,
    name="title_transformed",
)
def transform_predict(df):
    transformer, xgb_model = load_models()
    return xgb_model.predict(transformer.transform(df["title"]))


# connect to xorq's embedded engine
connection = xo.connect()

pipeline = (
    deferred_read_parquet(
        connection,
        xo.options.pins.get_path(HACKERNEWS_DATA_NAME),
        HACKERNEWS_DATA_NAME,
    )
    # process with HackerNews fetcher Exchanger that does a live fetch
    .pipe(hackernews_lib.do_hackernews_fetcher_udxf)
    # select only the title column
    .select(xo._.title)
    # add sentiment score prediction
    .mutate(sentiment_score=transform_predict.on_expr)
)

results = pipeline.execute()

"""
Next Steps: use the cli to build and see how things look like:

❯ xorq build scripts/hn_inference.py -e pipeline
Building pipeline from scripts/hn_inference.py
/nix/store/i7dqrcpgqll387lx48mfnhxq6nw5j1nb-xorq/lib/python3.10/site-packages/xgboost/core.py:265: FutureWarning: Your system has an old version of glibc (< 2.28). We will stop supporting Linux distros with glibc older than 2.28 after **May 31, 2025**. Please upgrade to a recent Linux distro (with glibc 2.28+) to use future versions of XGBoost.
Note: You have installed the 'manylinux2014' variant of XGBoost. Certain features such as GPU algorithms or federated learning are not available. To use these features, please upgrade to a recent Linux distro with glibc 2.28+, and install the 'manylinux_2_28' variant.
  warnings.warn(
Written 'pipeline' to builds/36293178ec4f
"""
