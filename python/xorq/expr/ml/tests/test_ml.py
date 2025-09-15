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
    make_quickgrove_udf,
)


sk_linear_model = pytest.importorskip("sklearn.linear_model")
sk_feature_extraction_text = pytest.importorskip("sklearn.feature_extraction.text")


def test_make_quickgrove_udf_predictions(feature_table, float_model_path):
    """quickgrove UDF predictions should match expected values"""
    predict_udf = make_quickgrove_udf(float_model_path)
    result = feature_table.mutate(pred=predict_udf.on_expr).execute()

    np.testing.assert_almost_equal(
        result["pred"].values, result["expected_pred"].values, decimal=3
    )


def test_make_quickgrove_udf_signature(float_model_path):
    """quickgrove UDF should have correct signature with float64 inputs, float32 output"""
    predict_fn = make_quickgrove_udf(float_model_path)

    assert predict_fn.__signature__.return_annotation == dt.float32
    assert all(
        p.annotation == dt.float64 for p in predict_fn.__signature__.parameters.values()
    )
    assert predict_fn.__name__ == "pretrained_model"


def test_make_quickgrove_udf_mixed_features(mixed_model_path):
    """quickgrove UDF should support int64 and boolean feature types"""
    predict_fn = make_quickgrove_udf(mixed_model_path)
    assert "i" in predict_fn.model.feature_types


def test_make_quickgrove_udf__repr(mixed_model_path):
    """quickgrove UDF repr should include model metadata"""
    predict_fn = make_quickgrove_udf(mixed_model_path)
    repr_str = repr(predict_fn)

    expected_info = [
        "Total number of trees:",
        "Average tree depth:",
        "Max tree depth:",
        "Total number of nodes:",
        "Model path:",
        "Signature:",
    ]

    for info in expected_info:
        assert info in repr_str


def test_quickgrove_hyphen_name(feature_table, hyphen_model_path):
    assert "-" in hyphen_model_path.name
    with pytest.raises(
        ValueError,
        match="The argument model_name was None and the name extracted from the path is not a valid Python identifier",
    ):
        make_quickgrove_udf(hyphen_model_path)

    with pytest.raises(
        ValueError, match="The argument model_name is not a valid Python identifier"
    ):
        make_quickgrove_udf(hyphen_model_path, "diamonds-model")

    predict_udf = make_quickgrove_udf(hyphen_model_path, model_name="diamonds_model")
    result = feature_table.mutate(pred=predict_udf.on_expr).execute()

    np.testing.assert_almost_equal(
        result["pred"].values, result["expected_pred"].values, decimal=3
    )


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


@pytest.mark.parametrize(
    "method",
    [
        "train_test_splits",
        "deferred_fit_predict",
        "deferred_fit_predict_sklearn",
        "deferred_fit_transform",
        "deferred_fit_transform_series_sklearn",
        "make_quickgrove_udf",
        "rewrite_quickgrove_expr",
    ],
)
def test_top_level_ml(method):
    assert hasattr(xo, method)
    assert hasattr(xo.ml, method)
