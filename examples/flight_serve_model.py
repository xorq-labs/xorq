from sklearn.feature_extraction.text import TfidfVectorizer

import xorq as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import ParquetStorage
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.import_utils import import_python
from xorq.expr.ml import (
    deferred_fit_transform_series_sklearn,
    train_test_splits,
)


m = import_python(xo.options.pins.get_path("hackernews_lib"))


deferred_fit_transform_tfidf = deferred_fit_transform_series_sklearn(
    col="title", cls=TfidfVectorizer, return_type=dt.Array(dt.float64)
)


con = xo.connect()
train_expr, test_expr = (
    deferred_read_parquet(
        con,
        xo.options.pins.get_path("hn-fetcher-input-small.parquet"),
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
    storage=ParquetStorage(source=con),
)
bound_expr = test_expr.mutate(**{"transformed": deferred_transform.on_expr})


server, do_exchange = xo.expr.relations.flight_serve(bound_expr)
df = do_exchange(test_expr).read_pandas()
server.close()
try:
    do_exchange(test_expr.limit(10)).read_pandas()
    raise Exception("do_exchange should have raised")
except Exception:
    pass
