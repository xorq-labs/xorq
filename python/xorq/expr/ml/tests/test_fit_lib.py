from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.caching import ParquetCache
from xorq.common.utils.defer_utils import (
    deferred_read_parquet,
)
from xorq.expr.ml.fit_lib import DeferredFitOther
from xorq.ml import (
    deferred_fit_predict_sklearn,
    deferred_fit_transform_series_sklearn,
)


sk_linear_model = pytest.importorskip("sklearn.linear_model")
sk_preprocessing = pytest.importorskip("sklearn.preprocessing")
sk_feature_selection = pytest.importorskip("sklearn.feature_selection")
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
    instance = deferred_linear_regression(t, target, features)
    (computed_kwargs_expr, predict_expr_udf) = (
        instance.deferred_model,
        instance.deferred_other,
    )
    model = computed_kwargs_expr.execute()
    predicted = t.mutate(predict_expr_udf.on_expr(t)).execute()

    # cached run
    cache = ParquetCache.from_kwargs(relative_path=tmp_path, source=con)
    instance = deferred_linear_regression(t, target, features, cache=cache)
    (computed_kwargs_expr, predict_expr_udf) = (
        instance.deferred_model,
        instance.deferred_other,
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

    instance = deferred_linear_regression(t, target, features)
    predict_expr_udf = instance.deferred_other
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
    instance = deferred_fit_transform_tfidf(train)
    deferred_transform = instance.deferred_other

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


class TestDeferredFitOtherChooseDeferred:
    """Tests for DeferredFitOther.choose_deferred_transform and choose_deferred_predict."""

    def test_choose_deferred_transform_known_schema(self):
        """Test choose_deferred_transform returns struct deferred for StandardScaler."""
        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        features = ("a", "b")

        deferred = DeferredFitOther.choose_deferred_transform(
            expr=t,
            features=features,
            sklearn_cls=sk_preprocessing.StandardScaler,
        )

        assert deferred is not None
        assert deferred.expr is t
        assert deferred.features == features
        # Known schema transformers should not be KV-encoded
        assert "struct" in deferred.name_infix or "transformed" in deferred.name_infix

    def test_choose_deferred_transform_kv_encoded(self):
        """Test choose_deferred_transform returns encoded deferred for OneHotEncoder."""
        t = xo.memtable({"cat": ["a", "b", "c"]})
        features = ("cat",)

        deferred = DeferredFitOther.choose_deferred_transform(
            expr=t,
            features=features,
            sklearn_cls=sk_preprocessing.OneHotEncoder,
        )

        assert deferred is not None
        assert deferred.expr is t
        assert deferred.features == features
        # KV-encoded transformers should have encoded in name
        assert "encoded" in deferred.name_infix

    def test_choose_deferred_transform_series_kv_encoded(self):
        """Test choose_deferred_transform returns series encoded deferred for TfidfVectorizer."""
        t = xo.memtable({"text": ["hello world", "foo bar"]})
        features = ("text",)

        deferred = DeferredFitOther.choose_deferred_transform(
            expr=t,
            features=features,
            sklearn_cls=sk_feature_extraction_text.TfidfVectorizer,
        )

        assert deferred is not None
        assert deferred.expr is t
        # TfidfVectorizer is series + KV-encoded
        assert "encoded" in deferred.name_infix

    def test_choose_deferred_transform_with_target(self):
        """Test choose_deferred_transform passes target for supervised transformers."""
        t = xo.memtable(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0], "y": [0, 0, 1, 1]}
        )
        features = ("a", "b")

        deferred = DeferredFitOther.choose_deferred_transform(
            expr=t,
            features=features,
            sklearn_cls=sk_feature_selection.SelectKBest,
            params=(("k", 1),),
            target="y",
        )

        assert deferred is not None
        assert deferred.target == "y"

    def test_choose_deferred_predict(self):
        """Test choose_deferred_predict returns predict deferred."""
        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [0.0, 1.0]})
        features = ("a", "b")

        deferred = DeferredFitOther.choose_deferred_predict(
            expr=t,
            features=features,
            sklearn_cls=sk_linear_model.LinearRegression,
            target="y",
            return_type=dt.float64,
        )

        assert deferred is not None
        assert deferred.expr is t
        assert deferred.features == features
        assert deferred.target == "y"

    def test_choose_deferred_transform_with_params(self):
        """Test choose_deferred_transform passes params correctly and executes."""
        t = xo.memtable({"cat": ["a", "b", "c"]})
        features = ("cat",)
        params = (("sparse_output", False),)

        deferred = DeferredFitOther.choose_deferred_transform(
            expr=t,
            features=features,
            sklearn_cls=sk_preprocessing.OneHotEncoder,
            params=params,
        )

        assert deferred is not None
        # Verify deferred can be used to transform
        col = deferred.deferred_other.on_expr(t)
        result = t.mutate(col).execute()
        # The column name is based on the UDF, check it exists and has KV-encoded data
        assert len(result.columns) == 2  # cat + transformed column
        transformed_col = [c for c in result.columns if c != "cat"][0]
        # Should have KV-encoded data
        first_row = result[transformed_col].iloc[0]
        assert all("key" in d and "value" in d for d in first_row)
