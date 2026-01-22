"""Tests for structer_from_instance registrations."""

import pandas as pd
import pytest

import xorq.api as xo
from xorq.expr.ml.pipeline_lib import has_structer_registration
from xorq.expr.ml.structer import structer_from_instance


@pytest.fixture
def numeric_expr():
    """Create a simple numeric expression for testing."""
    data = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
            "c": [1.0, 1.0, 2.0, 2.0, 3.0],
        }
    )
    return xo.memtable(data)


@pytest.fixture
def text_expr():
    """Create a text expression for testing."""
    data = pd.DataFrame({"text": ["hello world", "foo bar", "test document"]})
    return xo.memtable(data)


class TestColumnPreservingRegistrations:
    """Test scalers/imputers that preserve column names."""

    @pytest.mark.parametrize(
        "cls_name,cls",
        [
            ("StandardScaler", "sklearn.preprocessing.StandardScaler"),
            ("MinMaxScaler", "sklearn.preprocessing.MinMaxScaler"),
            ("RobustScaler", "sklearn.preprocessing.RobustScaler"),
            ("MaxAbsScaler", "sklearn.preprocessing.MaxAbsScaler"),
            ("Normalizer", "sklearn.preprocessing.Normalizer"),
            ("PowerTransformer", "sklearn.preprocessing.PowerTransformer"),
            ("QuantileTransformer", "sklearn.preprocessing.QuantileTransformer"),
            ("Binarizer", "sklearn.preprocessing.Binarizer"),
            ("SimpleImputer", "sklearn.impute.SimpleImputer"),
            ("KNNImputer", "sklearn.impute.KNNImputer"),
        ],
    )
    def test_column_preserving_known_schema(self, numeric_expr, cls_name, cls):
        """Test that column-preserving transformers have known schema."""
        # Import the class dynamically
        module_path, class_name = cls.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        transformer_cls = getattr(module, class_name)

        instance = transformer_cls()
        features = ("a", "b", "c")

        assert has_structer_registration(instance, numeric_expr, features)
        structer = structer_from_instance(instance, numeric_expr, features=features)

        assert not structer.is_kv_encoded
        assert tuple(structer.dtype) == features


class TestNComponentsRegistrations:
    """Test dimensionality reduction methods with n_components."""

    @pytest.mark.parametrize(
        "cls_name,cls,prefix",
        [
            ("PCA", "sklearn.decomposition.PCA", "pca"),
            ("TruncatedSVD", "sklearn.decomposition.TruncatedSVD", "truncatedsvd"),
            ("KernelPCA", "sklearn.decomposition.KernelPCA", "kernelpca"),
            ("SparsePCA", "sklearn.decomposition.SparsePCA", "sparsepca"),
            (
                "MiniBatchSparsePCA",
                "sklearn.decomposition.MiniBatchSparsePCA",
                "minibatchsparsepca",
            ),
            ("FactorAnalysis", "sklearn.decomposition.FactorAnalysis", "factor"),
            ("FastICA", "sklearn.decomposition.FastICA", "ica"),
            ("NMF", "sklearn.decomposition.NMF", "nmf"),
            ("MiniBatchNMF", "sklearn.decomposition.MiniBatchNMF", "minibatchnmf"),
            (
                "LatentDirichletAllocation",
                "sklearn.decomposition.LatentDirichletAllocation",
                "lda",
            ),
        ],
    )
    def test_n_components_int_known_schema(self, numeric_expr, cls_name, cls, prefix):
        """Test that n_components as int produces known schema."""
        module_path, class_name = cls.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        transformer_cls = getattr(module, class_name)

        instance = transformer_cls(n_components=2)
        features = ("a", "b", "c")

        assert has_structer_registration(instance, numeric_expr, features)
        structer = structer_from_instance(instance, numeric_expr, features=features)

        assert not structer.is_kv_encoded
        assert len(structer.dtype) == 2
        # Check prefix naming
        assert all(name.startswith(prefix) for name in structer.dtype)

    @pytest.mark.parametrize(
        "cls_name,cls",
        [
            ("PCA", "sklearn.decomposition.PCA"),
            ("FactorAnalysis", "sklearn.decomposition.FactorAnalysis"),
            ("FastICA", "sklearn.decomposition.FastICA"),
        ],
    )
    def test_n_components_none_kv_encoded(self, numeric_expr, cls_name, cls):
        """Test that n_components=None produces KV-encoded schema."""
        module_path, class_name = cls.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        transformer_cls = getattr(module, class_name)

        instance = transformer_cls(n_components=None)
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)
        assert structer.is_kv_encoded


class TestRandomProjectionRegistrations:
    """Test random projection methods."""

    @pytest.mark.parametrize(
        "cls_name,cls,prefix",
        [
            (
                "GaussianRandomProjection",
                "sklearn.random_projection.GaussianRandomProjection",
                "gaussianrp",
            ),
            (
                "SparseRandomProjection",
                "sklearn.random_projection.SparseRandomProjection",
                "sparserp",
            ),
        ],
    )
    def test_random_projection_int_known_schema(
        self, numeric_expr, cls_name, cls, prefix
    ):
        """Test random projections with int n_components."""
        module_path, class_name = cls.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        transformer_cls = getattr(module, class_name)

        instance = transformer_cls(n_components=2)
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)

        assert not structer.is_kv_encoded
        assert len(structer.dtype) == 2
        assert all(name.startswith(prefix) for name in structer.dtype)

    @pytest.mark.parametrize(
        "cls_name,cls",
        [
            (
                "GaussianRandomProjection",
                "sklearn.random_projection.GaussianRandomProjection",
            ),
            (
                "SparseRandomProjection",
                "sklearn.random_projection.SparseRandomProjection",
            ),
        ],
    )
    def test_random_projection_auto_kv_encoded(self, numeric_expr, cls_name, cls):
        """Test random projections with auto n_components uses KV-encoded."""
        module_path, class_name = cls.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        transformer_cls = getattr(module, class_name)

        instance = transformer_cls(n_components="auto")
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)
        assert structer.is_kv_encoded


class TestKernelApproximationRegistrations:
    """Test kernel approximation methods."""

    @pytest.mark.parametrize(
        "cls_name,cls,prefix",
        [
            ("Nystroem", "sklearn.kernel_approximation.Nystroem", "nystroem"),
            ("RBFSampler", "sklearn.kernel_approximation.RBFSampler", "rbf"),
            (
                "SkewedChi2Sampler",
                "sklearn.kernel_approximation.SkewedChi2Sampler",
                "skewedchi2",
            ),
        ],
    )
    def test_kernel_approximation_known_schema(
        self, numeric_expr, cls_name, cls, prefix
    ):
        """Test kernel approximation methods have known schema."""
        module_path, class_name = cls.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        transformer_cls = getattr(module, class_name)

        instance = transformer_cls(n_components=5)
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)

        assert not structer.is_kv_encoded
        assert len(structer.dtype) == 5
        assert all(name.startswith(prefix) for name in structer.dtype)

    def test_additive_chi2_sampler_kv_encoded(self, numeric_expr):
        """Test AdditiveChi2Sampler uses KV-encoded (depends on n_features)."""
        from sklearn.kernel_approximation import AdditiveChi2Sampler

        instance = AdditiveChi2Sampler()
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)
        assert structer.is_kv_encoded


class TestPolynomialFeatures:
    """Test PolynomialFeatures registration."""

    def test_polynomial_features_degree_2(self, numeric_expr):
        """Test PolynomialFeatures with degree=2."""
        from sklearn.preprocessing import PolynomialFeatures

        instance = PolynomialFeatures(degree=2, include_bias=False)
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)

        assert not structer.is_kv_encoded
        # For 3 features, degree 2, no bias: C(3+2,2) - 1 = 10 - 1 = 9
        assert len(structer.dtype) == 9
        assert all(name.startswith("poly") for name in structer.dtype)

    def test_polynomial_features_with_bias(self, numeric_expr):
        """Test PolynomialFeatures with bias term."""
        from sklearn.preprocessing import PolynomialFeatures

        instance = PolynomialFeatures(degree=2, include_bias=True)
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)

        assert not structer.is_kv_encoded
        # For 3 features, degree 2, with bias: C(3+2,2) = 10
        assert len(structer.dtype) == 10

    def test_polynomial_features_interaction_only(self, numeric_expr):
        """Test PolynomialFeatures with interaction_only."""
        from sklearn.preprocessing import PolynomialFeatures

        instance = PolynomialFeatures(
            degree=2, include_bias=False, interaction_only=True
        )
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)

        assert not structer.is_kv_encoded
        # For 3 features, degree 2, interaction only, no bias: C(3,1) + C(3,2) = 3 + 3 = 6
        assert len(structer.dtype) == 6


class TestFeatureSelectionRegistrations:
    """Test feature selection registrations."""

    def test_select_k_best_known_schema(self, numeric_expr):
        """Test SelectKBest has known schema."""
        from sklearn.feature_selection import SelectKBest, f_classif

        instance = SelectKBest(score_func=f_classif, k=2)
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)

        assert not structer.is_kv_encoded
        assert len(structer.dtype) == 2

    def test_rfe_fixed_k_known_schema(self, numeric_expr):
        """Test RFE with fixed n_features_to_select has known schema."""
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression()
        instance = RFE(estimator=estimator, n_features_to_select=2)
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)

        assert not structer.is_kv_encoded
        assert len(structer.dtype) == 2

    def test_rfe_dynamic_k_kv_encoded(self, numeric_expr):
        """Test RFE with dynamic n_features_to_select uses KV-encoded."""
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression()
        instance = RFE(estimator=estimator, n_features_to_select=None)
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)
        assert structer.is_kv_encoded


class TestEncoderRegistrations:
    """Test encoder registrations."""

    @pytest.mark.parametrize(
        "cls_name,cls",
        [
            ("OneHotEncoder", "sklearn.preprocessing.OneHotEncoder"),
            ("OrdinalEncoder", "sklearn.preprocessing.OrdinalEncoder"),
            ("LabelBinarizer", "sklearn.preprocessing.LabelBinarizer"),
            ("MultiLabelBinarizer", "sklearn.preprocessing.MultiLabelBinarizer"),
        ],
    )
    def test_encoders_kv_encoded(self, numeric_expr, cls_name, cls):
        """Test encoders use KV-encoded format."""
        module_path, class_name = cls.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        transformer_cls = getattr(module, class_name)

        instance = transformer_cls()
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)
        assert structer.is_kv_encoded


class TestTextVectorizerRegistrations:
    """Test text vectorizer registrations."""

    @pytest.mark.parametrize(
        "cls_name,cls",
        [
            ("CountVectorizer", "sklearn.feature_extraction.text.CountVectorizer"),
            ("TfidfVectorizer", "sklearn.feature_extraction.text.TfidfVectorizer"),
            ("HashingVectorizer", "sklearn.feature_extraction.text.HashingVectorizer"),
        ],
    )
    def test_text_vectorizers_kv_encoded(self, text_expr, cls_name, cls):
        """Test text vectorizers use KV-encoded format."""
        module_path, class_name = cls.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        transformer_cls = getattr(module, class_name)

        instance = transformer_cls()
        features = ("text",)

        structer = structer_from_instance(instance, text_expr, features=features)
        assert structer.is_kv_encoded

    def test_dict_vectorizer_kv_encoded(self, numeric_expr):
        """Test DictVectorizer uses KV-encoded format."""
        from sklearn.feature_extraction import DictVectorizer

        instance = DictVectorizer()
        features = ("a",)

        structer = structer_from_instance(instance, numeric_expr, features=features)
        assert structer.is_kv_encoded


class TestOtherRegistrations:
    """Test other registrations."""

    def test_spline_transformer_kv_encoded(self, numeric_expr):
        """Test SplineTransformer uses KV-encoded format."""
        from sklearn.preprocessing import SplineTransformer

        instance = SplineTransformer()
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)
        assert structer.is_kv_encoded

    def test_missing_indicator_kv_encoded(self, numeric_expr):
        """Test MissingIndicator uses KV-encoded format."""
        from sklearn.impute import MissingIndicator

        instance = MissingIndicator()
        features = ("a", "b", "c")

        structer = structer_from_instance(instance, numeric_expr, features=features)
        assert structer.is_kv_encoded


class TestContainerRegistrations:
    """Test container type registrations as Steps."""

    def test_column_transformer_wraps_as_step(self, numeric_expr):
        """Test ColumnTransformer can be wrapped as a Step."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import Step

        instance = ColumnTransformer(
            [("num", StandardScaler(), ["a", "b"])], remainder="drop"
        )

        step = Step.from_instance(instance, name="ct")
        assert step.name == "ct"
        assert step.typ.__name__ == "ColumnTransformer"

    def test_feature_union_wraps_as_step(self, numeric_expr):
        """Test FeatureUnion can be wrapped as a Step."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import Step

        instance = FeatureUnion([("scaler", StandardScaler())])

        step = Step.from_instance(instance, name="fu")
        assert step.name == "fu"
        assert step.typ.__name__ == "FeatureUnion"

    def test_sklearn_pipeline_wraps_as_step(self, numeric_expr):
        """Test sklearn Pipeline can be wrapped as a Step."""
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import Step

        instance = SklearnPipeline([("scaler", StandardScaler())])

        step = Step.from_instance(instance, name="pipe")
        assert step.name == "pipe"
        assert step.typ.__name__ == "Pipeline"
