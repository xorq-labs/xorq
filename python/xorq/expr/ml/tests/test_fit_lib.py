from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.caching import ParquetStorage
from xorq.common.utils.defer_utils import (
    deferred_read_parquet,
)
from xorq.ml import (
    deferred_fit_predict_sklearn,
    deferred_fit_transform_series_sklearn,
)


sk_linear_model = pytest.importorskip("sklearn.linear_model")
sk_feature_extraction_text = pytest.importorskip("sklearn.feature_extraction.text")


def make_data():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    df = pd.DataFrame(np.hstack((X, y[:, np.newaxis]))).rename(
        columns=lambda x: chr(x + ord("a"))
    )
    (*features, target) = df.columns
    return (df, features, target)


deferred_linear_regression = deferred_fit_predict_sklearn(
    cls=sk_linear_model.LinearRegression, return_type=dt.float64
)


def test_deferred_fit_predict_linear_regression(tmp_path):
    con = xo.connect()
    (df, features, target) = make_data()
    t = con.register(df, "t")

    # uncached run
    (computed_kwargs_expr, _, predict_expr_udf) = deferred_linear_regression(
        t, target, features
    )
    model = computed_kwargs_expr.execute()
    predicted = t.mutate(predict_expr_udf.on_expr(t)).execute()

    # cached run
    storage = ParquetStorage(relative_path=tmp_path, source=con)
    (computed_kwargs_expr, _, predict_expr_udf) = deferred_linear_regression(
        t, target, features, storage=storage
    )
    ((cached_model,),) = computed_kwargs_expr.execute().values
    cached_predicted = t.mutate(predict_expr_udf.on_expr(t)).execute()

    assert predicted.equals(cached_predicted)
    np.testing.assert_almost_equal(
        pickle.loads(model).coef_, pickle.loads(cached_model).coef_
    )


def test_deferred_fit_predict_linear_regression_multi_into_backend():
    con = xo.connect()
    (df, features, target) = make_data()
    t = (
        con.register(df, "t")
        .into_backend(xo.connect())[lambda t: t[features[0]] > 0]
        .into_backend(xo.connect())
    )

    (_, _, predict_expr_udf) = deferred_linear_regression(t, target, features)
    predicted = t.mutate(predict_expr_udf.on_expr(t)).execute()
    assert not predicted.empty


def test_deferred_fit_transform_series_sklearn():
    cls = sk_feature_extraction_text.TfidfVectorizer
    transform_key = "transformed"
    col = "title"
    deferred_fit_transform_tfidf = deferred_fit_transform_series_sklearn(
        col=col, cls=cls, return_type=dt.Array(dt.float64)
    )
    con = xo.connect()
    train, test = deferred_read_parquet(
        xo.options.pins.get_path("hn-data-small.parquet"),
        con,
        "fetcher-input",
    ).pipe(
        xo.train_test_splits,
        unique_key="id",
        test_sizes=(0.9, 0.1),
    )
    (_, _, deferred_transform) = deferred_fit_transform_tfidf(
        train,
    )

    from_sklearn = test.execute().assign(
        **{
            transform_key: lambda t: (
                cls().fit(train.execute()[col]).transform(t[col]).toarray().tolist()
            )
        }
    )
    from_xo = test.mutate(**{transform_key: deferred_transform.on_expr}).execute()
    actual = from_xo[transform_key].apply(pd.Series)
    expected = from_sklearn[transform_key].apply(pd.Series)
    assert actual.equals(expected)
