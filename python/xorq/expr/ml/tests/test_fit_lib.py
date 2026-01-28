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
from xorq.expr.ml.fit_lib import (
    DeferredFitOther,
    _get_named_transformers,
    _get_output_indices,
    transform_sklearn_hybrid,
)
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


class TestDeferredFitOtherFromFittedStep:
    """Tests for DeferredFitOther.from_fitted_step factory method."""

    def test_from_fitted_step_transform_known_schema(self):
        """Test from_fitted_step returns struct deferred for StandardScaler."""
        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        step = xo.Step.from_instance_name(
            sk_preprocessing.StandardScaler(), name="scaler"
        )
        fitted = step.fit(t, features=("a", "b"))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred is not None
        assert deferred.expr is fitted.expr
        assert deferred.features == fitted.features
        # Known schema transformers should have "transformed" in name
        assert "transformed" in deferred.name_infix

    def test_from_fitted_step_transform_kv_encoded(self):
        """Test from_fitted_step returns encoded deferred for OneHotEncoder."""
        t = xo.memtable({"cat": ["a", "b", "c"]})
        step = xo.Step.from_instance_name(sk_preprocessing.OneHotEncoder(), name="ohe")
        fitted = step.fit(t, features=("cat",))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred is not None
        assert deferred.expr is fitted.expr
        assert deferred.features == fitted.features
        # KV-encoded transformers should have encoded in name
        assert "encoded" in deferred.name_infix

    def test_from_fitted_step_transform_series_kv_encoded(self):
        """Test from_fitted_step returns series encoded deferred for TfidfVectorizer."""
        t = xo.memtable({"text": ["hello world", "foo bar"]})
        step = xo.Step.from_instance_name(
            sk_feature_extraction_text.TfidfVectorizer(), name="tfidf"
        )
        fitted = step.fit(t, features=("text",))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred is not None
        assert deferred.expr is fitted.expr
        # TfidfVectorizer is series + KV-encoded
        assert "encoded" in deferred.name_infix

    def test_from_fitted_step_transform_with_target(self):
        """Test from_fitted_step passes target for supervised transformers."""
        t = xo.memtable(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0], "y": [0, 0, 1, 1]}
        )
        step = xo.Step.from_instance_name(
            sk_feature_selection.SelectKBest(k=1), name="skb"
        )
        fitted = step.fit(t, features=("a", "b"), target="y")

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred is not None
        assert deferred.target == "y"

    def test_from_fitted_step_predict(self):
        """Test from_fitted_step returns predict deferred for predictor."""
        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [0.0, 1.0]})
        step = xo.Step.from_instance_name(sk_linear_model.LinearRegression(), name="lr")
        fitted = step.fit(t, features=("a", "b"), target="y")

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred is not None
        assert deferred.expr is fitted.expr
        assert deferred.features == fitted.features
        assert deferred.target == "y"
        assert "predict" in deferred.name_infix

    def test_from_fitted_step_transform_with_params_executes(self):
        """Test from_fitted_step passes params correctly and executes."""
        t = xo.memtable({"cat": ["a", "b", "c"]})
        step = xo.Step.from_instance_name(
            sk_preprocessing.OneHotEncoder(sparse_output=False), name="ohe"
        )
        fitted = step.fit(t, features=("cat",))

        deferred = DeferredFitOther.from_fitted_step(fitted)

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


sk_compose = pytest.importorskip("sklearn.compose")
sk_pipeline = pytest.importorskip("sklearn.pipeline")
sk_decomposition = pytest.importorskip("sklearn.decomposition")


class TestGetOutputIndices:
    """Tests for _get_output_indices helper function."""

    def test_column_transformer_uses_output_indices_attr(self):
        """ColumnTransformer should use its output_indices_ attribute."""
        ct = sk_compose.ColumnTransformer(
            [
                ("num", sk_preprocessing.StandardScaler(), ["a", "b"]),
                ("cat", sk_preprocessing.OneHotEncoder(sparse_output=False), ["c"]),
            ]
        )
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": ["x", "y"]})
        ct.fit(df)

        result = _get_output_indices(ct)

        # Should return the same as output_indices_
        assert result == ct.output_indices_
        assert "num" in result
        assert "cat" in result
        assert result["num"] == slice(0, 2)

    def test_feature_union_computes_indices(self):
        """FeatureUnion should compute indices from transformer feature counts."""
        fu = sk_pipeline.FeatureUnion(
            [
                ("scale", sk_preprocessing.StandardScaler()),
                ("pca", sk_decomposition.PCA(n_components=2)),
            ]
        )
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]}
        )
        fu.fit(df)

        result = _get_output_indices(fu)

        # scale outputs 3 features, pca outputs 2
        assert result["scale"] == slice(0, 3)
        assert result["pca"] == slice(3, 5)


class TestGetNamedTransformers:
    """Tests for _get_named_transformers helper function."""

    def test_column_transformer_uses_named_transformers_attr(self):
        """ColumnTransformer should use named_transformers_ attribute."""
        ct = sk_compose.ColumnTransformer(
            [
                ("num", sk_preprocessing.StandardScaler(), ["a"]),
                ("cat", sk_preprocessing.OneHotEncoder(sparse_output=False), ["b"]),
            ]
        )
        df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        ct.fit(df)

        result = _get_named_transformers(ct)

        assert result == ct.named_transformers_
        assert "num" in result
        assert "cat" in result
        assert isinstance(result["num"], sk_preprocessing.StandardScaler)

    def test_feature_union_uses_named_transformers_property(self):
        """FeatureUnion should use named_transformers property."""
        fu = sk_pipeline.FeatureUnion(
            [
                ("scale", sk_preprocessing.StandardScaler()),
                ("pca", sk_decomposition.PCA(n_components=2)),
            ]
        )
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        fu.fit(df)

        result = _get_named_transformers(fu)

        assert result == fu.named_transformers
        assert "scale" in result
        assert "pca" in result


class TestTransformSklearnHybrid:
    """Tests for transform_sklearn_hybrid function."""

    def test_column_transformer_mixed_children(self):
        """Test hybrid output with mixed known/KV-encoded children."""
        ct = sk_compose.ColumnTransformer(
            [
                ("num", sk_preprocessing.StandardScaler(), ["age", "income"]),
                (
                    "cat",
                    sk_preprocessing.OneHotEncoder(sparse_output=False),
                    ["category"],
                ),
            ]
        )
        df = pd.DataFrame(
            {
                "age": [25.0, 30.0, 35.0],
                "income": [50000.0, 60000.0, 70000.0],
                "category": ["a", "b", "a"],
            }
        )
        ct.fit(df)

        # cat is KV-encoded, num is known-schema
        result = transform_sklearn_hybrid(("cat",), ct, df)

        assert len(result) == 3
        # Known-schema fields should be unpacked directly
        assert "age" in result[0]
        assert "income" in result[0]
        # KV-encoded field should be a tuple of {key, value} dicts
        assert "cat" in result[0]
        assert isinstance(result[0]["cat"], tuple)
        assert all("key" in d and "value" in d for d in result[0]["cat"])

    def test_feature_union_mixed_children(self):
        """Test hybrid output with FeatureUnion."""
        fu = sk_pipeline.FeatureUnion(
            [
                ("scale", sk_preprocessing.StandardScaler()),
                ("pca", sk_decomposition.PCA(n_components=2)),
            ]
        )
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [5.0, 4.0, 3.0, 2.0, 1.0],
                "c": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        fu.fit(df)

        # pca is KV-encoded, scale is known-schema
        result = transform_sklearn_hybrid(("pca",), fu, df)

        assert len(result) == 5
        # scale outputs a, b, c as known fields
        assert "a" in result[0]
        assert "b" in result[0]
        assert "c" in result[0]
        # pca outputs as KV-encoded
        assert "pca" in result[0]
        assert isinstance(result[0]["pca"], tuple)
        assert len(result[0]["pca"]) == 2  # n_components=2

    def test_all_known_schema_children(self):
        """Test with no KV-encoded children (all known)."""
        ct = sk_compose.ColumnTransformer(
            [
                ("num1", sk_preprocessing.StandardScaler(), ["a"]),
                ("num2", sk_preprocessing.StandardScaler(), ["b"]),
            ]
        )
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        ct.fit(df)

        # No KV-encoded children
        result = transform_sklearn_hybrid((), ct, df)

        assert len(result) == 2
        # All fields should be unpacked directly
        assert "a" in result[0]
        assert "b" in result[0]
        # No KV-encoded fields
        assert not any(isinstance(v, tuple) for v in result[0].values())

    def test_all_kv_encoded_children(self):
        """Test with all KV-encoded children."""
        ct = sk_compose.ColumnTransformer(
            [
                ("cat1", sk_preprocessing.OneHotEncoder(sparse_output=False), ["a"]),
                ("cat2", sk_preprocessing.OneHotEncoder(sparse_output=False), ["b"]),
            ]
        )
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        ct.fit(df)

        # All children are KV-encoded
        result = transform_sklearn_hybrid(("cat1", "cat2"), ct, df)

        assert len(result) == 2
        # Both should be KV-encoded fields
        assert "cat1" in result[0]
        assert "cat2" in result[0]
        assert isinstance(result[0]["cat1"], tuple)
        assert isinstance(result[0]["cat2"], tuple)

    def test_sparse_matrix_handling(self):
        """Test that sparse matrices are converted to dense."""
        ct = sk_compose.ColumnTransformer(
            [
                ("num", sk_preprocessing.StandardScaler(), ["a"]),
                ("cat", sk_preprocessing.OneHotEncoder(sparse_output=True), ["b"]),
            ]
        )
        df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        ct.fit(df)

        # Should not raise even with sparse output
        result = transform_sklearn_hybrid(("cat",), ct, df)

        assert len(result) == 2
        assert "a" in result[0]
        assert "cat" in result[0]

    def test_feature_names_without_prefix(self):
        """Test that feature names don't include transformer prefix."""
        ct = sk_compose.ColumnTransformer(
            [
                ("scaler", sk_preprocessing.StandardScaler(), ["my_feature"]),
            ]
        )
        df = pd.DataFrame({"my_feature": [1.0, 2.0, 3.0]})
        ct.fit(df)

        result = transform_sklearn_hybrid((), ct, df)

        # Should use "my_feature" not "scaler__my_feature"
        assert "my_feature" in result[0]
        assert "scaler__my_feature" not in result[0]
