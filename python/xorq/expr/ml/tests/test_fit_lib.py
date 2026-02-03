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
    kv_encode_output,
)
from xorq.expr.ml.structer import KV_ENCODED_TYPE
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


class TestFromFittedStepMatchStatementBranches:
    """Tests verifying each branch of the match statement in from_fitted_step."""

    def test_series_kv_encoded_branch_name_infix(self):
        """Test (True, True, _) branch uses 'transformed_encoded' name_infix for TfidfVectorizer."""
        t = xo.memtable({"text": ["hello world", "foo bar"]})
        step = xo.Step.from_instance_name(
            sk_feature_extraction_text.TfidfVectorizer(), name="tfidf"
        )
        fitted = step.fit(t, features=("text",))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred.name_infix == "transformed_encoded"
        assert deferred.return_type == KV_ENCODED_TYPE

    def test_pure_kv_encoded_branch_name_infix(self):
        """Test (False, True, _) branch uses 'transformed_encoded' name_infix for OneHotEncoder."""
        t = xo.memtable({"cat": ["a", "b", "c"]})
        step = xo.Step.from_instance_name(sk_preprocessing.OneHotEncoder(), name="ohe")
        fitted = step.fit(t, features=("cat",))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred.name_infix == "transformed_encoded"
        assert deferred.return_type == KV_ENCODED_TYPE

    def test_struct_with_kv_fields_branch_name_infix(self):
        """Test (False, False, True) branch uses 'transformed_struct_kv' for ColumnTransformer with mixed output."""
        t = xo.memtable({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
        ct = sk_compose.ColumnTransformer(
            [
                ("scaler", sk_preprocessing.StandardScaler(), ["num"]),
                ("encoder", sk_preprocessing.OneHotEncoder(), ["cat"]),
            ]
        )
        step = xo.Step.from_instance_name(ct, name="ct")
        fitted = step.fit(t, features=("num", "cat"))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred.name_infix == "transformed_struct_kv"
        # Return type should be a struct containing both regular and KV-encoded fields
        assert deferred.return_type is not None

    def test_pure_struct_branch_name_infix(self):
        """Test default branch uses 'transformed' name_infix for StandardScaler."""
        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        step = xo.Step.from_instance_name(
            sk_preprocessing.StandardScaler(), name="scaler"
        )
        fitted = step.fit(t, features=("a", "b"))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred.name_infix == "transformed"
        # Return type should be a struct
        assert isinstance(deferred.return_type, dt.Struct)

    def test_predict_branch_name_infix(self):
        """Test predict branch uses 'predict' name_infix for LinearRegression."""
        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [0.0, 1.0]})
        step = xo.Step.from_instance_name(sk_linear_model.LinearRegression(), name="lr")
        fitted = step.fit(t, features=("a", "b"), target="y")

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred.name_infix == "predict"


class TestFromFittedStepFitFunctions:
    """Tests verifying correct fit function is used for each transformer type."""

    def test_series_transformer_uses_fit_sklearn_series(self):
        """Test TfidfVectorizer uses fit_sklearn_series with correct col parameter."""
        t = xo.memtable({"text": ["hello world", "foo bar"]})
        step = xo.Step.from_instance_name(
            sk_feature_extraction_text.TfidfVectorizer(), name="tfidf"
        )
        fitted = step.fit(t, features=("text",))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        # fit_sklearn_series is a curried function, verify it has the col parameter
        assert deferred.fit.keywords.get("col") == "text"

    def test_multi_column_transformer_uses_fit_sklearn_args(self):
        """Test StandardScaler with multiple columns uses fit_sklearn_args."""
        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        step = xo.Step.from_instance_name(
            sk_preprocessing.StandardScaler(), name="scaler"
        )
        fitted = step.fit(t, features=("a", "b"))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        # fit_sklearn_args uses cls and params keywords
        assert "cls" in deferred.fit.keywords
        assert deferred.fit.keywords["cls"] == sk_preprocessing.StandardScaler

    def test_predictor_uses_fit_sklearn(self):
        """Test LinearRegression uses fit_sklearn with target support."""
        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [0.0, 1.0]})
        step = xo.Step.from_instance_name(sk_linear_model.LinearRegression(), name="lr")
        fitted = step.fit(t, features=("a", "b"), target="y")

        deferred = DeferredFitOther.from_fitted_step(fitted)

        # fit_sklearn uses cls and params keywords
        assert "cls" in deferred.fit.keywords
        assert deferred.fit.keywords["cls"] == sk_linear_model.LinearRegression


class TestFromFittedStepOtherFunctions:
    """Tests verifying correct other (transform/predict) function is used."""

    def test_series_kv_encoded_uses_transform_sklearn_series_kv(self):
        """Test TfidfVectorizer uses transform_sklearn_series_kv."""
        t = xo.memtable({"text": ["hello world", "foo bar"]})
        step = xo.Step.from_instance_name(
            sk_feature_extraction_text.TfidfVectorizer(), name="tfidf"
        )
        fitted = step.fit(t, features=("text",))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        # transform_sklearn_series_kv has col keyword
        assert deferred.other.keywords.get("col") == "text"

    def test_pure_kv_encoded_uses_kv_encode_output(self):
        """Test OneHotEncoder uses kv_encode_output."""
        t = xo.memtable({"cat": ["a", "b", "c"]})
        step = xo.Step.from_instance_name(sk_preprocessing.OneHotEncoder(), name="ohe")
        fitted = step.fit(t, features=("cat",))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        # kv_encode_output is the function itself (not curried with keywords)
        assert deferred.other is kv_encode_output

    def test_pure_struct_uses_transform_sklearn_struct(self):
        """Test StandardScaler uses transform_sklearn_struct with convert_array."""
        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        step = xo.Step.from_instance_name(
            sk_preprocessing.StandardScaler(), name="scaler"
        )
        fitted = step.fit(t, features=("a", "b"))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        # transform_sklearn_struct is the base function (curried with convert_array as first arg)
        # Check that it's transform_sklearn_struct by verifying function name
        assert deferred.other.func.__name__ == "transform_sklearn_struct"


class TestFromFittedStepReturnTypes:
    """Tests verifying correct return types for each transformer type."""

    def test_kv_encoded_return_type(self):
        """Test KV-encoded transformers return KV_ENCODED_TYPE."""
        t = xo.memtable({"cat": ["a", "b", "c"]})
        step = xo.Step.from_instance_name(sk_preprocessing.OneHotEncoder(), name="ohe")
        fitted = step.fit(t, features=("cat",))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred.return_type == KV_ENCODED_TYPE
        # KV_ENCODED_TYPE is Array[Struct{key: string, value: float64}]
        assert isinstance(deferred.return_type, dt.Array)

    def test_struct_return_type_has_correct_fields(self):
        """Test struct transformers return type with correct field names."""
        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        step = xo.Step.from_instance_name(
            sk_preprocessing.StandardScaler(), name="scaler"
        )
        fitted = step.fit(t, features=("a", "b"))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert isinstance(deferred.return_type, dt.Struct)
        # Struct should have same field names as features
        field_names = set(deferred.return_type.names)
        assert field_names == {"a", "b"}

    def test_predict_return_type_matches_target(self):
        """Test predict return type matches the target column type."""
        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [0.0, 1.0]})
        step = xo.Step.from_instance_name(sk_linear_model.LinearRegression(), name="lr")
        fitted = step.fit(t, features=("a", "b"), target="y")

        deferred = DeferredFitOther.from_fitted_step(fitted)

        # Return type should be float64 to match target
        assert deferred.return_type == dt.float64


class TestFromFittedStepTargetHandling:
    """Tests verifying target is correctly passed through for supervised cases."""

    def test_transformer_with_target_passes_target(self):
        """Test supervised transformer (SelectKBest) preserves target."""
        t = xo.memtable(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0], "y": [0, 0, 1, 1]}
        )
        step = xo.Step.from_instance_name(
            sk_feature_selection.SelectKBest(k=1), name="skb"
        )
        fitted = step.fit(t, features=("a", "b"), target="y")

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred.target == "y"

    def test_unsupervised_transformer_has_no_target(self):
        """Test unsupervised transformer (StandardScaler) has None target."""
        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        step = xo.Step.from_instance_name(
            sk_preprocessing.StandardScaler(), name="scaler"
        )
        fitted = step.fit(t, features=("a", "b"))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        assert deferred.target is None

    def test_kv_encoded_transformer_target_cleared_when_not_needed(self):
        """Test KV-encoded transformer clears target when structer doesn't need it."""
        t = xo.memtable({"cat": ["a", "b", "c"]})
        step = xo.Step.from_instance_name(sk_preprocessing.OneHotEncoder(), name="ohe")
        # OneHotEncoder doesn't need target even if provided
        fitted = step.fit(t, features=("cat",))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        # Target should be None for unsupervised transformers
        assert deferred.target is None


class TestFromFittedStepColumnTransformerWithKvFields:
    """Tests for ColumnTransformer producing struct with KV-encoded fields."""

    def test_column_transformer_mixed_output_struct_has_kv_fields(self):
        """Test ColumnTransformer with scaler + encoder produces struct with KV fields."""
        t = xo.memtable({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
        ct = sk_compose.ColumnTransformer(
            [
                ("scaler", sk_preprocessing.StandardScaler(), ["num"]),
                ("encoder", sk_preprocessing.OneHotEncoder(), ["cat"]),
            ]
        )
        step = xo.Step.from_instance_name(ct, name="ct")
        fitted = step.fit(t, features=("num", "cat"))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        # Return type should be struct
        assert isinstance(deferred.return_type, dt.Struct)
        # Struct should have both regular fields and KV-encoded field
        field_types = dict(deferred.return_type.items())
        # scaler output should have regular float fields
        assert "scaler__num" in field_types
        # encoder output should be KV-encoded
        assert "encoder" in field_types
        assert field_types["encoder"] == KV_ENCODED_TYPE

    def test_column_transformer_all_kv_encoded_uses_struct_kv_branch(self):
        """Test ColumnTransformer with all KV-encoded outputs uses struct_kv branch."""
        t = xo.memtable({"cat1": ["a", "b", "c"], "cat2": ["x", "y", "z"]})
        ct = sk_compose.ColumnTransformer(
            [
                ("enc1", sk_preprocessing.OneHotEncoder(), ["cat1"]),
                ("enc2", sk_preprocessing.OneHotEncoder(), ["cat2"]),
            ]
        )
        step = xo.Step.from_instance_name(ct, name="ct")
        fitted = step.fit(t, features=("cat1", "cat2"))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        # Should use struct_kv branch since struct exists with KV fields
        assert deferred.name_infix == "transformed_struct_kv"
        # All fields should be KV-encoded
        field_types = dict(deferred.return_type.items())
        for name, typ in field_types.items():
            assert typ == KV_ENCODED_TYPE, f"Field {name} should be KV-encoded"

    def test_column_transformer_mixed_executes_correctly(self):
        """Test ColumnTransformer with mixed output produces correct result."""
        t = xo.memtable({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
        ct = sk_compose.ColumnTransformer(
            [
                ("scaler", sk_preprocessing.StandardScaler(), ["num"]),
                ("encoder", sk_preprocessing.OneHotEncoder(), ["cat"]),
            ]
        )
        step = xo.Step.from_instance_name(ct, name="ct")
        fitted = step.fit(t, features=("num", "cat"))

        deferred = DeferredFitOther.from_fitted_step(fitted)

        # Execute the transform
        col = deferred.deferred_other.on_expr(t)
        result = t.mutate(transformed=col).execute()

        # Verify we get struct output
        assert "transformed" in result.columns
        first_row = result["transformed"].iloc[0]
        # Should have scaler__num (float) and encoder (list of dicts)
        assert "scaler__num" in first_row
        assert "encoder" in first_row
        # encoder should be KV-encoded (list of dicts)
        assert isinstance(first_row["encoder"], list)
        assert all("key" in d and "value" in d for d in first_row["encoder"])
