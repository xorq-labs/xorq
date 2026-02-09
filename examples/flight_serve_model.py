"""Flight server serving TF-IDF model transformations on HackerNews data.

Traditional approach: You would train the model, save it to disk, build a
Flask or FastAPI endpoint, handle request parsing, batching, and response
serialization. Deploying the model as a service requires significant
boilerplate around the serving infrastructure.

With xorq: deferred_fit_transform produces an expression that captures both
training and inference. Calling flight_serve() on the bound expression turns
any deferred ML operation into a Flight endpoint in one step, with no separate
serving code required.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

import xorq.api as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import ParquetCache
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.import_utils import import_python
from xorq.expr.ml import (
    deferred_fit_transform_series_sklearn,
    train_test_splits,
)


m = import_python(
    xo.options.pins.get_path("hackernews_lib", version="20250820T111457Z-1d66a")
)


deferred_fit_transform_tfidf = deferred_fit_transform_series_sklearn(
    col="title", cls=TfidfVectorizer, return_type=dt.Array(dt.float64)
)


con = xo.connect()
train_expr, test_expr = (
    deferred_read_parquet(
        xo.options.pins.get_path("hn-fetcher-input-small.parquet"),
        con,
        "fetcher-input",
    )
    .pipe(m.do_hackernews_fetcher_udxf)
    .pipe(
        train_test_splits,
        unique_key="id",
        test_sizes=(0.9, 0.1),
        random_seed=0,
    )
)


(deferred_model, model_udaf, deferred_transform) = deferred_fit_transform_tfidf(
    train_expr,
    cache=ParquetCache.from_kwargs(source=con),
).deferred_model_udaf_other
bound_expr = test_expr.mutate(**{"transformed": deferred_transform.on_expr})


if __name__ in ("__pytest_main__", "__main__"):
    server, do_exchange = xo.expr.relations.flight_serve(bound_expr)
    df = do_exchange(test_expr).read_pandas()
    server.close()
    try:
        do_exchange(test_expr.limit(10)).read_pandas()
        raise Exception("do_exchange should have raised")
    except Exception:
        pass
    pytest_examples_passed = True
