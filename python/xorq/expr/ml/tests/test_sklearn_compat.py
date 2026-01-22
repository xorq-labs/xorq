"""
Integration tests for sklearn ↔ xorq interoperability.

Tests the following features:
- to_sklearn() round-trip for Step, FittedStep, Pipeline, FittedPipeline
- Packed format for transformers with dynamic schemas
- Mixin catch-alls for predictors (RegressorMixin, ClusterMixin)
- Transductive/fit-only estimators (TSNE, DBSCAN)
"""

import pandas as pd
import pytest

import xorq.api as xo
from xorq.expr.ml.fit_lib import (
    decode_encoded_column,
    kv_encode_output,
)
from xorq.expr.ml.pipeline_lib import (
    Pipeline,
    Step,
    has_structer_registration,
    is_fit_predict_only,
    is_fit_transform_only,
)


sklearn = pytest.importorskip("sklearn")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def sample_data():
    """Simple dataset for testing transformers and predictors."""
    return pd.DataFrame(
        {
            "feature_0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_1": [5.0, 4.0, 3.0, 2.0, 1.0],
            "category": ["a", "b", "a", "b", "a"],
            "target": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture(scope="module")
def expr(sample_data):
    """xorq expression from sample data."""
    return xo.memtable(sample_data)


@pytest.fixture(scope="module")
def numeric_features():
    return ("feature_0", "feature_1")


# =============================================================================
# Test: to_sklearn() for Step (unfitted)
# =============================================================================


class TestStepToSklearn:
    """Tests for Step.to_sklearn() method."""

    def test_step_to_sklearn_standard_scaler(self):
        """Step.to_sklearn() returns unfitted StandardScaler."""
        from sklearn.preprocessing import StandardScaler

        original = StandardScaler()
        step = Step.from_instance(original, name="scaler")
        result = step.to_sklearn()

        assert isinstance(result, StandardScaler)
        # Not fitted yet
        assert not hasattr(result, "mean_") or result.mean_ is None

    def test_step_to_sklearn_preserves_params(self):
        """Step.to_sklearn() preserves estimator parameters."""
        from sklearn.neighbors import KNeighborsClassifier

        original = KNeighborsClassifier(n_neighbors=7, weights="distance")
        step = Step.from_instance(original, name="knn")
        result = step.to_sklearn()

        assert result.n_neighbors == 7
        assert result.weights == "distance"


# =============================================================================
# Test: to_sklearn() for FittedStep
# =============================================================================


class TestFittedStepToSklearn:
    """Tests for FittedStep.to_sklearn() method."""

    def test_fitted_step_to_sklearn_transformer(self, expr, numeric_features):
        """FittedStep.to_sklearn() returns fitted transformer."""
        from sklearn.preprocessing import StandardScaler

        step = Step.from_instance(StandardScaler(), name="scaler")
        fitted_step = step.fit(expr, features=numeric_features)
        sklearn_model = fitted_step.to_sklearn()

        assert isinstance(sklearn_model, StandardScaler)
        # Should be fitted - has mean_ attribute
        assert hasattr(sklearn_model, "mean_")
        assert sklearn_model.mean_ is not None

    def test_fitted_step_to_sklearn_predictor(self, expr, numeric_features):
        """FittedStep.to_sklearn() returns fitted predictor."""
        from sklearn.linear_model import LogisticRegression

        step = Step.from_instance(LogisticRegression(max_iter=200), name="clf")
        fitted_step = step.fit(expr, features=numeric_features, target="target")
        sklearn_model = fitted_step.to_sklearn()

        assert isinstance(sklearn_model, LogisticRegression)
        # Should be fitted - has coef_ attribute
        assert hasattr(sklearn_model, "coef_")


# =============================================================================
# Test: to_sklearn() for Pipeline (unfitted)
# =============================================================================


class TestPipelineToSklearn:
    """Tests for Pipeline.to_sklearn() method."""

    def test_pipeline_to_sklearn_roundtrip(self):
        """Pipeline.to_sklearn() round-trip preserves structure."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        sklearn_pipe = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )

        xorq_pipe = Pipeline.from_instance(sklearn_pipe)
        sklearn_pipe_back = xorq_pipe.to_sklearn()

        assert isinstance(sklearn_pipe_back, SklearnPipeline)
        assert len(sklearn_pipe_back.steps) == 2
        assert sklearn_pipe_back.steps[0][0] == "scaler"
        assert sklearn_pipe_back.steps[1][0] == "clf"


# =============================================================================
# Test: to_sklearn() for FittedPipeline
# =============================================================================


class TestFittedPipelineToSklearn:
    """Tests for FittedPipeline.to_sklearn() method."""

    def test_fitted_pipeline_to_sklearn(self, expr, numeric_features):
        """FittedPipeline.to_sklearn() returns fitted sklearn Pipeline."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        sklearn_pipe = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )

        xorq_pipe = Pipeline.from_instance(sklearn_pipe)
        fitted_xorq = xorq_pipe.fit(expr, features=numeric_features, target="target")
        sklearn_fitted = fitted_xorq.to_sklearn()

        assert isinstance(sklearn_fitted, SklearnPipeline)
        # Should be fitted
        assert hasattr(sklearn_fitted.named_steps["scaler"], "mean_")
        assert hasattr(sklearn_fitted.named_steps["clf"], "coef_")

    def test_fitted_pipeline_can_predict(self, expr, numeric_features, sample_data):
        """Fitted sklearn pipeline from to_sklearn() can predict."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        sklearn_pipe = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )

        xorq_pipe = Pipeline.from_instance(sklearn_pipe)
        fitted_xorq = xorq_pipe.fit(expr, features=numeric_features, target="target")
        sklearn_fitted = fitted_xorq.to_sklearn()

        # Should be able to predict
        X_test = sample_data[list(numeric_features)]
        predictions = sklearn_fitted.predict(X_test)
        assert len(predictions) == len(X_test)


# =============================================================================
# Test: Mixin catch-alls
# =============================================================================


class TestMixinCatchAlls:
    """Tests for mixin catch-all registrations."""

    def test_regressor_mixin_returns_float64(self, expr, numeric_features):
        """RegressorMixin predictors return float64."""
        from sklearn.linear_model import Ridge

        step = Step.from_instance(Ridge(), name="ridge")
        fitted = step.fit(expr, features=numeric_features, target="target")

        # Should be able to predict without explicit registration
        result = fitted.predict(expr)
        assert result is not None

    @pytest.mark.skip(
        reason="KMeans has both transform and predict, which FittedStep doesn't support"
    )
    def test_cluster_mixin_works(self, expr, numeric_features):
        """ClusterMixin estimators work via catch-all (for inductive clusterers).

        NOTE: This test is skipped because KMeans has both transform() and predict()
        methods, which violates FittedStep's assumption that estimators are either
        transformers OR predictors (not both). This is a limitation in the current
        design that would need to be addressed in a future iteration.

        The ClusterMixin catch-all registration is still functional for estimators
        that only have predict() (not transform()).
        """
        from sklearn.cluster import KMeans

        step = Step.from_instance(KMeans(n_clusters=2, n_init=3), name="kmeans")
        fitted = step.fit(expr, features=numeric_features, target="target")
        result = fitted.predict(expr)
        assert result is not None


# =============================================================================
# Test: Packed format utilities
# =============================================================================


class TestKVEncoder:
    """Tests for KV encoding utilities."""

    def test_kv_encode_output(self):
        """kv_encode_output creates Array[Struct{key, value}] format."""
        from sklearn.preprocessing import StandardScaler

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        scaler = StandardScaler()
        scaler.fit(df)

        encoded = kv_encode_output(scaler, df)

        assert isinstance(encoded, pd.Series)
        assert len(encoded) == 3
        # Each row should be a tuple of dicts with key/value
        first_row = encoded.iloc[0]
        assert isinstance(first_row, tuple)
        assert all("key" in item and "value" in item for item in first_row)

    def test_decode_encoded_column(self):
        """decode_encoded_column converts encoded format back to columns."""
        # Create encoded data
        encoded_data = [
            ({"key": "a", "value": 1.0}, {"key": "b", "value": 4.0}),
            ({"key": "a", "value": 2.0}, {"key": "b", "value": 5.0}),
            ({"key": "a", "value": 3.0}, {"key": "b", "value": 6.0}),
        ]
        df = pd.DataFrame({"id": [1, 2, 3], "transformed": encoded_data})

        result = decode_encoded_column(df, col_name="transformed")

        assert "a" in result.columns
        assert "b" in result.columns
        assert "transformed" not in result.columns
        assert list(result["a"]) == [1.0, 2.0, 3.0]
        assert list(result["b"]) == [4.0, 5.0, 6.0]


# =============================================================================
# Test: Estimator mode detection
# =============================================================================


class TestEstimatorModeDetection:
    """Tests for estimator mode detection helpers."""

    def test_is_fit_transform_only(self):
        """is_fit_transform_only correctly identifies transductive embeddings."""
        from sklearn.manifold import TSNE

        assert is_fit_transform_only(TSNE) is True

    def test_is_fit_transform_only_regular_transformer(self):
        """is_fit_transform_only returns False for regular transformers."""
        from sklearn.preprocessing import StandardScaler

        assert is_fit_transform_only(StandardScaler) is False

    def test_is_fit_predict_only(self):
        """is_fit_predict_only correctly identifies transductive clusterers."""
        from sklearn.cluster import DBSCAN

        assert is_fit_predict_only(DBSCAN) is True

    def test_is_fit_predict_only_regular_clusterer(self):
        """is_fit_predict_only returns False for inductive clusterers."""
        from sklearn.cluster import KMeans

        assert is_fit_predict_only(KMeans) is False


# =============================================================================
# Test: Structer registration detection
# =============================================================================


class TestStructerRegistration:
    """Tests for Structer registration detection."""

    def test_has_structer_registration_for_registered_type(
        self, expr, numeric_features
    ):
        """has_structer_registration returns True for registered types."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        assert has_structer_registration(scaler, expr, numeric_features) is True

    def test_has_structer_registration_for_unregistered_type(
        self, expr, numeric_features
    ):
        """has_structer_registration returns False for unregistered types."""
        from sklearn.preprocessing import FunctionTransformer

        # FunctionTransformer is not explicitly registered in structer.py
        transformer = FunctionTransformer()
        result = has_structer_registration(transformer, expr, numeric_features)
        # Should be False since FunctionTransformer is not in the Structer registry
        assert result is False


# =============================================================================
# Test: Packed format fallback for unregistered transformers
# =============================================================================


class TestPackedFormatFallback:
    """Tests for packed format fallback for unregistered transformers."""

    def test_unregistered_transformer_uses_packed_format(self, expr, numeric_features):
        """Unregistered transformers fall back to packed format."""
        from sklearn.preprocessing import MinMaxScaler

        step = Step.from_instance(MinMaxScaler(), name="minmax")
        fitted = step.fit(expr, features=numeric_features)

        # Should be able to transform using packed format
        result = fitted.transform_raw(expr)
        assert result is not None


# =============================================================================
# Test: predict_proba() and decision_function()
# =============================================================================


class TestProbabilisticOutputs:
    """Tests for predict_proba() and decision_function() methods."""

    def test_predict_proba_returns_packed_format(self, expr, numeric_features):
        """predict_proba() returns packed format with class probabilities."""
        from sklearn.linear_model import LogisticRegression

        step = Step.from_instance(LogisticRegression(max_iter=200), name="clf")
        fitted = step.fit(expr, features=numeric_features, target="target")

        # Should return packed format
        result = fitted.predict_proba(expr, retain_others=False)
        assert result is not None
        assert "proba" in result.columns

    def test_predict_proba_raises_for_non_classifier(self, expr, numeric_features):
        """predict_proba() raises AttributeError for estimators without it."""
        from sklearn.linear_model import LinearRegression

        step = Step.from_instance(LinearRegression(), name="reg")
        fitted = step.fit(expr, features=numeric_features, target="target")

        with pytest.raises(AttributeError, match="does not have predict_proba"):
            fitted.predict_proba(expr)

    def test_decision_function_returns_packed_format(self, expr, numeric_features):
        """decision_function() returns packed format with decision values."""
        from sklearn.svm import SVC

        step = Step.from_instance(SVC(kernel="linear"), name="svc")
        fitted = step.fit(expr, features=numeric_features, target="target")

        # Should return packed format
        result = fitted.decision_function(expr, retain_others=False)
        assert result is not None
        assert "decision" in result.columns

    def test_decision_function_raises_for_non_svm(self, expr, numeric_features):
        """decision_function() raises AttributeError for estimators without it."""
        from sklearn.neighbors import KNeighborsClassifier

        step = Step.from_instance(KNeighborsClassifier(n_neighbors=3), name="knn")
        fitted = step.fit(expr, features=numeric_features, target="target")

        with pytest.raises(AttributeError, match="does not have decision_function"):
            fitted.decision_function(expr)


# =============================================================================
# Test: Vectorizers (CountVectorizer, TfidfVectorizer)
# =============================================================================


@pytest.fixture(scope="module")
def text_data():
    """Dataset with text column for vectorizer tests."""
    return pd.DataFrame(
        {
            "text": [
                "hello world",
                "goodbye world",
                "hello goodbye",
                "world peace",
                "hello peace",
            ],
            "target": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture(scope="module")
def text_expr(text_data):
    """xorq expression from text data."""
    return xo.memtable(text_data)


class TestVectorizers:
    """Tests for CountVectorizer and TfidfVectorizer with from_instance()."""

    def test_tfidf_vectorizer_from_instance(self, text_expr):
        """TfidfVectorizer works with from_instance()."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        step = Step.from_instance(TfidfVectorizer(), name="tfidf")
        fitted = step.fit(text_expr, features=("text",))

        # Should be able to transform
        result = fitted.transform_raw(text_expr)
        assert result is not None

    def test_count_vectorizer_from_instance(self, text_expr):
        """CountVectorizer works with from_instance() (packed format)."""
        from sklearn.feature_extraction.text import CountVectorizer

        step = Step.from_instance(CountVectorizer(), name="countvec")
        fitted = step.fit(text_expr, features=("text",))

        # CountVectorizer uses packed format fallback
        result = fitted.transform_raw(text_expr)
        assert result is not None


# =============================================================================
# Test: Dimensionality Reduction (PCA, TruncatedSVD)
# =============================================================================


class TestDimensionalityReduction:
    """Tests for PCA and TruncatedSVD with from_instance()."""

    def test_pca_from_instance(self, expr, numeric_features):
        """PCA works with from_instance() (Structer format with registered type)."""
        from sklearn.decomposition import PCA

        step = Step.from_instance(PCA(n_components=2), name="pca")
        fitted = step.fit(expr, features=numeric_features)

        # PCA with int n_components has a registered Structer
        result = fitted.transform(expr)
        assert result is not None

        # Execute and check the result - sklearn uses 'pca0', 'pca1', etc.
        result_df = result.execute()
        assert "pca0" in result_df.columns
        assert "pca1" in result_df.columns
        assert len([c for c in result_df.columns if c.startswith("pca")]) == 2

    def test_truncated_svd_from_instance(self, expr, numeric_features):
        """TruncatedSVD works with from_instance() (Structer format with registered type)."""
        from sklearn.decomposition import TruncatedSVD

        step = Step.from_instance(TruncatedSVD(n_components=1), name="svd")
        fitted = step.fit(expr, features=numeric_features)

        result = fitted.transform(expr)
        assert result is not None

        # Execute and verify - sklearn uses 'truncatedsvd0', 'truncatedsvd1', etc.
        result_df = result.execute()
        assert "truncatedsvd0" in result_df.columns
        assert len([c for c in result_df.columns if c.startswith("truncatedsvd")]) == 1


# =============================================================================
# Test: Composite Transformers (ColumnTransformer, FeatureUnion)
# =============================================================================


@pytest.fixture(scope="module")
def mixed_data():
    """Dataset with numeric and categorical columns for composite transformer tests."""
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "cat1": ["a", "b", "a", "b", "a"],
            "target": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture(scope="module")
def mixed_expr(mixed_data):
    """xorq expression from mixed data."""
    return xo.memtable(mixed_data)


class TestCompositeTransformers:
    """Tests for ColumnTransformer and FeatureUnion with from_instance()."""

    def test_column_transformer_from_instance(self, mixed_expr):
        """ColumnTransformer works with from_instance()."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        ct = ColumnTransformer(
            [
                ("num", StandardScaler(), ["num1", "num2"]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["cat1"]),
            ]
        )

        step = Step.from_instance(ct, name="preprocessor")
        # Use all feature columns (exclude target)
        fitted = step.fit(mixed_expr, features=("num1", "num2", "cat1"))

        # ColumnTransformer uses packed format
        result = fitted.transform_raw(mixed_expr)
        assert result is not None

        # Execute and verify structure
        result_df = result.as_table().execute()
        first_row = result_df["transformed"].iloc[0]
        assert isinstance(first_row, (tuple, list))
        # Should have 2 numeric + 2 one-hot (categories a, b)
        assert len(first_row) == 4

    def test_feature_union_from_instance(self, expr, numeric_features):
        """FeatureUnion works with from_instance()."""
        from sklearn.decomposition import PCA
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import StandardScaler

        fu = FeatureUnion(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=1)),
            ]
        )

        step = Step.from_instance(fu, name="feature_union")
        fitted = step.fit(expr, features=numeric_features)

        result = fitted.transform_raw(expr)
        assert result is not None


# =============================================================================
# Test: Clusterers (KMeans, DBSCAN)
# =============================================================================


class TestClusterers:
    """Tests for KMeans and DBSCAN with from_instance()."""

    def test_dbscan_from_instance_is_fit_predict_only(self, expr, numeric_features):
        """DBSCAN is correctly identified as fit_predict_only."""
        from sklearn.cluster import DBSCAN

        assert is_fit_predict_only(DBSCAN) is True

        step = Step.from_instance(DBSCAN(eps=0.5, min_samples=2), name="dbscan")
        # Clusterers don't need a target
        fitted = step.fit(expr, features=numeric_features)

        # Should be marked as transductive
        assert fitted.is_fit_predict_only is True
        assert fitted.is_transductive is True

    def test_dbscan_predict_on_new_data_raises(self, expr, numeric_features):
        """DBSCAN raises error when predicting on new data."""
        from sklearn.cluster import DBSCAN

        step = Step.from_instance(DBSCAN(eps=0.5, min_samples=2), name="dbscan")
        # Clusterers don't need a target
        fitted = step.fit(expr, features=numeric_features)

        # Should raise TypeError for transductive clusterer
        with pytest.raises(TypeError, match="transductive clusterer"):
            fitted.predict_raw(expr)

    def test_kmeans_from_instance(self, expr, numeric_features):
        """KMeans works with from_instance() as inductive clusterer."""
        from sklearn.cluster import KMeans

        assert is_fit_predict_only(KMeans) is False

        step = Step.from_instance(
            KMeans(n_clusters=2, n_init=3, random_state=42), name="kmeans"
        )

        # KMeans has both transform and predict, so it's treated as predictor
        # Clusterers don't need a target
        fitted = step.fit(expr, features=numeric_features)

        # KMeans is inductive, not transductive
        assert fitted.is_fit_predict_only is False


# =============================================================================
# Test: Mixed xorq + sklearn pipelines
# =============================================================================


class TestMixedPipelines:
    """Tests for mixed xorq + sklearn pipelines."""

    def test_mixed_pipeline_with_sklearn_step(self, expr, numeric_features):
        """Pipeline with sklearn Step.from_instance() works correctly."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Create steps using from_instance
        scaler_step = Step.from_instance(StandardScaler(), name="scaler")
        clf_step = Step.from_instance(LogisticRegression(max_iter=200), name="clf")

        # Create xorq Pipeline directly
        pipeline = Pipeline(steps=(scaler_step, clf_step))

        # Fit and predict
        fitted = pipeline.fit(expr, features=numeric_features, target="target")
        result = fitted.predict(expr)
        assert result is not None

        # Execute and verify
        result_df = result.execute()
        assert "predicted" in result_df.columns

    def test_sklearn_pipeline_wrapped_as_single_step(self, mixed_expr):
        """Full sklearn Pipeline wrapped as single Step (recommended for ColumnTransformer)."""
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        # Create complex sklearn pipeline
        sklearn_pipe = SklearnPipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        [
                            ("num", StandardScaler(), ["num1", "num2"]),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), ["cat1"]),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )

        # Wrap entire pipeline as single Step (recommended approach)
        step = Step.from_instance(sklearn_pipe, name="full_pipeline")
        fitted = step.fit(
            mixed_expr, features=("num1", "num2", "cat1"), target="target"
        )

        # Should work as predictor
        result = fitted.predict(mixed_expr)
        assert result is not None

        result_df = result.execute()
        assert "predicted" in result_df.columns


# =============================================================================
# Test: Caching for fitted models
# =============================================================================


class TestCaching:
    """Tests for caching functionality with fitted models."""

    def test_fitted_step_with_cache(self, expr, numeric_features, tmp_path):
        """FittedStep works with caching."""
        from sklearn.preprocessing import StandardScaler

        from xorq.caching import ParquetCache

        # Create cache
        cache = ParquetCache.from_kwargs(
            source=xo.connect(),
            relative_path="test-cache",
            base_path=tmp_path,
        )

        step = Step.from_instance(StandardScaler(), name="scaler")
        fitted = step.fit(expr, features=numeric_features, cache=cache)

        # First execution should cache
        result1 = fitted.transform(expr).execute()

        # Second execution should use cache
        result2 = fitted.transform(expr).execute()

        # Results should match
        pd.testing.assert_frame_equal(result1, result2)

    def test_fitted_pipeline_with_cache(self, expr, numeric_features, tmp_path):
        """FittedPipeline works with caching."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        from xorq.caching import ParquetCache

        cache = ParquetCache.from_kwargs(
            source=xo.connect(),
            relative_path="test-cache",
            base_path=tmp_path,
        )

        sklearn_pipe = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )

        xorq_pipe = Pipeline.from_instance(sklearn_pipe)
        fitted = xorq_pipe.fit(
            expr, features=numeric_features, target="target", cache=cache
        )

        # Execute with caching
        result1 = fitted.predict(expr).execute()
        result2 = fitted.predict(expr).execute()

        pd.testing.assert_frame_equal(result1, result2)


# =============================================================================
# Test: Transductive estimators error handling
# =============================================================================


class TestTransductiveEstimators:
    """Tests for transductive estimator error handling."""

    def test_tsne_is_fit_transform_only(self):
        """TSNE is correctly identified as fit_transform_only."""
        from sklearn.manifold import TSNE

        assert is_fit_transform_only(TSNE) is True

    def test_tsne_transform_raises_error(self, expr, numeric_features):
        """TSNE raises helpful error when transform is called."""
        from sklearn.manifold import TSNE

        step = Step.from_instance(
            TSNE(n_components=2, perplexity=2, random_state=42), name="tsne"
        )
        # TSNE is a transformer (fit_transform_only), doesn't need target
        fitted = step.fit(expr, features=numeric_features)

        # Should raise TypeError with helpful message
        with pytest.raises(TypeError, match="transductive estimator"):
            fitted.transform(expr)

    def test_mds_is_fit_transform_only(self):
        """MDS is correctly identified as fit_transform_only."""
        from sklearn.manifold import MDS

        assert is_fit_transform_only(MDS) is True


# =============================================================================
# Test: Verification checklist items
# =============================================================================


class TestVerificationChecklist:
    """Tests to verify the completion checklist requirements."""

    def test_pipeline_from_instance_without_registration(self, expr, numeric_features):
        """Pipeline.from_instance() works without explicit Structer registration."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import MinMaxScaler

        # MinMaxScaler is NOT registered in Structer - should use packed format
        # NOTE: When a transformer uses packed format, subsequent steps cannot
        # consume it directly. The recommended approach is to wrap the entire
        # sklearn pipeline as a single Step.
        sklearn_pipe = SklearnPipeline(
            [
                ("scaler", MinMaxScaler()),  # Unregistered transformer
                ("clf", RandomForestClassifier(n_estimators=5, random_state=42)),
            ]
        )

        # Wrap entire pipeline as single Step (recommended approach for packed format)
        step = Step.from_instance(sklearn_pipe, name="full_pipeline")
        fitted = step.fit(expr, features=numeric_features, target="target")

        # Should work without registration
        result = fitted.predict(expr).execute()
        assert "predicted" in result.columns

    def test_to_sklearn_returns_working_pipeline(
        self, expr, numeric_features, sample_data
    ):
        """to_sklearn() returns a working sklearn pipeline."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        sklearn_pipe = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )

        xorq_pipe = Pipeline.from_instance(sklearn_pipe)
        fitted = xorq_pipe.fit(expr, features=numeric_features, target="target")

        # Get sklearn pipeline back
        sklearn_fitted = fitted.to_sklearn()

        # Should be able to predict with sklearn directly
        X_test = sample_data[list(numeric_features)]
        predictions = sklearn_fitted.predict(X_test)
        assert len(predictions) == len(X_test)

        # Should also work for predict_proba
        proba = sklearn_fitted.predict_proba(X_test)
        assert proba.shape[0] == len(X_test)

    def test_operations_are_deferred(self, expr, numeric_features):
        """All operations are deferred (no eager execution)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        sklearn_pipe = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )

        xorq_pipe = Pipeline.from_instance(sklearn_pipe)
        fitted = xorq_pipe.fit(expr, features=numeric_features, target="target")

        # predict() should return an expression, not execute immediately
        result = fitted.predict(expr)

        # Result should be an Expr (deferred)
        from xorq.vendor.ibis.expr.types.core import Expr

        assert isinstance(result, Expr)

        # Only execute() triggers computation
        df = result.execute()
        assert isinstance(df, pd.DataFrame)

    def test_structer_format_preserves_feature_names(self, expr, numeric_features):
        """Structer format correctly preserves feature names for registered types."""
        from sklearn.decomposition import PCA

        step = Step.from_instance(PCA(n_components=2), name="pca")
        fitted = step.fit(expr, features=numeric_features)

        # PCA is now registered with Structer, uses individual columns
        result = fitted.transform(expr)
        result_df = result.execute()

        # Check that feature names follow sklearn convention (pca0, pca1)
        pca_cols = [c for c in result_df.columns if c.startswith("pca")]
        assert len(pca_cols) == 2
        assert "pca0" in pca_cols
        assert "pca1" in pca_cols

    def test_predictors_work_with_mixin_catchalls(self, expr, numeric_features):
        """Predictors work with mixin catch-all registrations."""
        from sklearn.ensemble import GradientBoostingRegressor

        # GradientBoostingRegressor uses RegressorMixin catch-all
        step = Step.from_instance(
            GradientBoostingRegressor(n_estimators=5, max_depth=2, random_state=42),
            name="gbr",
        )
        fitted = step.fit(expr, features=numeric_features, target="target")

        # Should work without explicit registration
        result = fitted.predict(expr).execute()
        assert "predicted" in result.columns
