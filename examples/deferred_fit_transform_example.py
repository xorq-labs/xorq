import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

import xorq as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import ParquetStorage
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.import_utils import import_python
from xorq.ml import (
    deferred_fit_transform_series_sklearn,
    train_test_splits,
)


m = import_python(
    xo.options.pins.get_path("hackernews_lib", version="20250319T145842Z-023b8")
)


deferred_fit_transform_tfidf = deferred_fit_transform_series_sklearn(
    col="title", cls=TfidfVectorizer, return_type=dt.Array(dt.float64)
)


con = xo.connect()
storage = ParquetStorage(source=con)
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
)
(cached_deferred_model, cached_model_udaf, cached_deferred_transform) = (
    deferred_fit_transform_tfidf(train_expr, storage=storage)
)


if __name__ == "__pytest_main__":
    model = deferred_model.execute()
    transformed = test_expr.mutate(
        **{"transformed": deferred_transform.on_expr}
    ).execute()
    ((cached_model,),) = cached_deferred_model.execute().values
    cached_transformed = test_expr.mutate(
        **{"transformed": deferred_transform.on_expr}
    ).execute()

    assert transformed.equals(cached_transformed)
    (x, y) = (pickle.loads(el) for el in (model, cached_model))
    assert all(x.idf_ == y.idf_)
    assert x.vocabulary_ == y.vocabulary_
    pytest_examples_passed = True
