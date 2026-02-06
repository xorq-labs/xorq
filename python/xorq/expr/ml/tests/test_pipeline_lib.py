import operator
from numbers import Real

import pytest

import xorq.api as xo
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import Tag


sklearn = pytest.importorskip("sklearn")


TARGET = "target"
features = (feature0, feature1) = ("feature_0", "feature_1")


get_metadata = operator.attrgetter("metadata")


@pytest.fixture(scope="module")
def t():
    return xo.memtable(
        {
            feature0: [1, 2],
            feature1: [3, 4],
            TARGET: [0, 1],
        }
    )


step_typs = (
    sklearn.preprocessing.StandardScaler,
    sklearn.linear_model.LinearRegression,
)


@pytest.fixture(scope="module")
def sklearn_pipeline():
    sklearn_pipeline = sklearn.pipeline.make_pipeline(*(typ() for typ in step_typs))
    return sklearn_pipeline


@pytest.fixture(scope="module")
def fitted_xorq_pipeline(sklearn_pipeline, t):
    xorq_pipeline = xo.Pipeline.from_instance(sklearn_pipeline)
    return xorq_pipeline.fit(t, target=TARGET)


def test_infer_features(fitted_xorq_pipeline):
    assert all(
        features == step.features for step in fitted_xorq_pipeline.transform_steps
    )


@pytest.fixture(scope="module")
def all_tags(t, fitted_xorq_pipeline):
    expr = fitted_xorq_pipeline.predict(t)
    all_tags = walk_nodes((Tag,), expr)
    return all_tags


def test_all_tags(t, fitted_xorq_pipeline, all_tags):
    expr = fitted_xorq_pipeline.predict(t)
    actual = tuple(map(get_metadata, expr.ls.get_tags()))
    expected = tuple(map(get_metadata, all_tags))
    assert actual == expected


@pytest.mark.parametrize(
    "pairs",
    (
        (("tag", "FittedStep-transform"),),
        (("tag", "FittedStep-predict"),),
        (
            ("tag", "FittedStep-transform"),
            ("tag", "FittedStep-predict"),
        ),
    ),
)
def test_tagging_pipeline(pairs, t, fitted_xorq_pipeline):
    def contains_any_pairs(d, pairs=pairs):
        return set(pairs).intersection(d.items())

    def sort_and_tuplify(dcts):
        return tuple(sorted(tuple(sorted(dct.items())) for dct in dcts))

    actual = sort_and_tuplify(
        map(
            get_metadata,
            fitted_xorq_pipeline.predict(t).ls.get_tags(
                predicate=contains_any_pairs,
            ),
        )
    )
    expected = sort_and_tuplify(
        dct
        for dct in (
            fitted_step.tag_kwargs for fitted_step in fitted_xorq_pipeline.fitted_steps
        )
        if contains_any_pairs(dct)
    )
    assert actual and actual == expected


def test_score_expr_returns_metric(t, fitted_xorq_pipeline):
    score_expr = fitted_xorq_pipeline.score_expr(t)
    result = score_expr.execute()
    assert isinstance(result, Real)


class TestFittedStepTransform:
    """Tests for FittedStep.transform simplified logic."""

    def test_fitted_step_transform_known_schema_unpacks(self):
        """Test FittedStep.transform unpacks struct columns for known schema."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        step = xo.Step.from_instance_name(StandardScaler(), name="scaler")
        fitted = step.fit(t, features=("a", "b"))

        result = fitted.transform(t)
        df = result.execute()

        # Should have unpacked columns a and b, not a struct column
        assert "a" in df.columns
        assert "b" in df.columns
        assert "transformed" not in df.columns

    def test_fitted_step_transform_kv_encoded_no_unpack(self):
        """Test FittedStep.transform keeps KV-encoded column without unpacking."""
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["x", "y", "x", "z"]})
        step = xo.Step.from_instance_name(OneHotEncoder(), name="ohe")
        fitted = step.fit(t, features=("cat",))

        result = fitted.transform(t)
        df = result.execute()

        # Should have KV-encoded column named "transformed"
        assert "transformed" in df.columns
        # Should not have unpacked category columns
        assert "cat_x" not in df.columns
        assert "cat_y" not in df.columns
        assert "cat_z" not in df.columns

    def test_fitted_step_transform_retain_others_true(self):
        """Test FittedStep.transform retains other columns by default."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "other": ["x", "y"]})
        step = xo.Step.from_instance_name(StandardScaler(), name="scaler")
        fitted = step.fit(t, features=("a", "b"))

        result = fitted.transform(t, retain_others=True)
        df = result.execute()

        # Should retain the "other" column
        assert "other" in df.columns
        assert df["other"].tolist() == ["x", "y"]

    def test_fitted_step_transform_retain_others_false(self):
        """Test FittedStep.transform drops other columns when retain_others=False."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "other": ["x", "y"]})
        step = xo.Step.from_instance_name(StandardScaler(), name="scaler")
        fitted = step.fit(t, features=("a", "b"))

        result = fitted.transform(t, retain_others=False)
        df = result.execute()

        # Should not retain the "other" column
        assert "other" not in df.columns


class TestPipelineGetOutputColumns:
    """Tests for Pipeline using Structer.get_output_columns."""

    def test_pipeline_known_schema_features_propagate(self):
        """Test Pipeline correctly propagates features for known schema transformers."""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "y": [0.0, 1.0, 0.0]}
        )
        pipeline = xo.Pipeline.from_instance(
            sklearn.pipeline.make_pipeline(StandardScaler(), LinearRegression())
        )
        fitted = pipeline.fit(t, target="y")

        # The transform step should have features = ("a", "b") from the known schema
        transform_step = fitted.transform_steps[0]
        assert transform_step.structer.get_output_columns() == ("a", "b")

    def test_pipeline_kv_encoded_features_use_dest_col(self):
        """Test Pipeline correctly uses dest_col for KV-encoded transformers."""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b", "a"], "y": [0.0, 1.0, 0.0]})
        pipeline = xo.Pipeline.from_instance(
            sklearn.pipeline.make_pipeline(OneHotEncoder(), LinearRegression())
        )
        fitted = pipeline.fit(t, target="y")

        # The transform step should have features = ("transformed",) for KV-encoded
        transform_step = fitted.transform_steps[0]
        assert transform_step.structer.get_output_columns("transformed") == (
            "transformed",
        )

    def test_pipeline_mixed_transform_steps(self):
        """Test Pipeline with multiple transform steps propagates features correctly."""
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable(
            {"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0], "y": [0.0, 1.0, 0.0]}
        )
        pipeline = xo.Pipeline.from_instance(
            sklearn.pipeline.make_pipeline(
                SimpleImputer(), StandardScaler(), LinearRegression()
            )
        )
        fitted = pipeline.fit(t, target="y")

        # Both transform steps should have known schema
        for transform_step in fitted.transform_steps:
            assert not transform_step.structer.is_kv_encoded

        # Prediction should work
        result = fitted.predict(t)
        assert result.execute() is not None


class TestDeeplyNestedPipelines:
    """Tests for deeply nested sklearn pipelines with xorq.

    These tests verify that xorq produces identical predictions to sklearn
    for complex nested pipeline structures.
    """

    def test_kv_encoded_deeply_nested_pipeline(self):
        """Test depth-4 nested pipeline with KV-encoded ColumnTransformer.

        Pipeline structure:
        - ColumnTransformer (KV-encoded due to OneHotEncoder)
          - FeatureUnion
            - Pipeline (SimpleImputer -> StandardScaler)
            - Pipeline (SimpleImputer -> StandardScaler)
          - Pipeline (SimpleImputer -> OneHotEncoder)
        - SelectKBest
        - RandomForestClassifier
        """
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import FeatureUnion
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.pipeline_lib import Pipeline

        # Create sample data
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_samples).astype(float),
                "income": np.random.randint(20000, 150000, n_samples).astype(float),
                "credit_score": np.random.randint(300, 850, n_samples).astype(float),
                "years_employed": np.random.randint(0, 40, n_samples).astype(float),
                "education": np.random.choice(
                    ["high_school", "bachelor", "master", "phd"], n_samples
                ),
                "employment_type": np.random.choice(
                    ["full_time", "part_time", "contract", "self_employed"], n_samples
                ),
                "region": np.random.choice(
                    ["north", "south", "east", "west"], n_samples
                ),
                "approved": np.random.randint(0, 2, n_samples),
            }
        )

        numeric_features = ["age", "income", "credit_score", "years_employed"]
        categorical_features = ["education", "employment_type", "region"]
        all_features = tuple(numeric_features + categorical_features)

        # Build nested sklearn pipeline
        scaled_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        imputed_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        numeric_union = FeatureUnion(
            [
                ("scaled", scaled_pipeline),
                ("imputed", imputed_pipeline),
            ]
        )

        categorical_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("numeric", numeric_union, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ]
        )

        sklearn_pipe = SklearnPipeline(
            [
                ("preprocessor", preprocessor),
                ("selector", SelectKBest(f_classif, k=10)),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=50, random_state=42),
                ),
            ]
        )

        # Fit and predict with xorq
        expr = xo.memtable(data)
        xorq_pipeline = Pipeline.from_instance(sklearn_pipe)
        fitted_pipeline = xorq_pipeline.fit(
            expr, features=all_features, target="approved"
        )
        predictions = fitted_pipeline.predict(expr).execute()

        # Fit and predict with sklearn
        X = data[list(all_features)]
        y = data["approved"]
        sklearn_pipe.fit(X, y)
        sklearn_preds = sklearn_pipe.predict(X)

        # Assert predictions match
        assert np.array_equal(predictions["predicted"].values, sklearn_preds)

    def test_non_kv_deeply_nested_pipeline(self):
        """Test depth-4 nested pipeline with all known-schema transformers.

        Pipeline structure:
        - ColumnTransformer (known schema - no KV-encoded children)
          - Pipeline (SimpleImputer -> StandardScaler -> Pipeline)
            - Pipeline (SimpleImputer -> StandardScaler)
          - Pipeline (SimpleImputer -> StandardScaler)
        - RandomForestClassifier
        """
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import Pipeline

        # Create sample data
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_samples).astype(float),
                "income": np.random.randint(20000, 150000, n_samples).astype(float),
                "credit_score": np.random.randint(300, 850, n_samples).astype(float),
                "years_employed": np.random.randint(0, 40, n_samples).astype(float),
                "debt_ratio": np.random.uniform(0, 1, n_samples),
                "savings": np.random.randint(0, 100000, n_samples).astype(float),
                "approved": np.random.randint(0, 2, n_samples),
            }
        )

        numeric_features_a = ["age", "income", "credit_score"]
        numeric_features_b = ["years_employed", "debt_ratio", "savings"]
        all_features = tuple(numeric_features_a + numeric_features_b)

        # Build nested sklearn pipeline (depth 4)
        inner_pipeline = SklearnPipeline(
            [
                ("imputer2", SimpleImputer(strategy="mean")),
                ("scaler2", StandardScaler()),
            ]
        )

        numeric_a_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("inner", inner_pipeline),
            ]
        )

        numeric_b_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("numeric_a", numeric_a_pipeline, numeric_features_a),
                ("numeric_b", numeric_b_pipeline, numeric_features_b),
            ]
        )

        sklearn_pipe = SklearnPipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=50, random_state=42),
                ),
            ]
        )

        # Fit and predict with xorq
        expr = xo.memtable(data)
        xorq_pipeline = Pipeline.from_instance(sklearn_pipe)
        fitted_pipeline = xorq_pipeline.fit(
            expr, features=all_features, target="approved"
        )
        predictions = fitted_pipeline.predict(expr).execute()

        # Fit and predict with sklearn
        X = data[list(all_features)]
        y = data["approved"]
        sklearn_pipe.fit(X, y)
        sklearn_preds = sklearn_pipe.predict(X)

        # Assert predictions match
        assert np.array_equal(predictions["predicted"].values, sklearn_preds)


def get_scorers_by_type():
    """Categorize all sklearn scorers by their type based on internal module path."""
    from sklearn.metrics import get_scorer, get_scorer_names

    classification = set()
    regression = set()
    cluster = set()
    proba = set()
    multilabel = set()

    for name in get_scorer_names():
        scorer = get_scorer(name)
        module = scorer._score_func.__module__
        response = scorer._response_method

        # Check if needs predict_proba/decision_function
        if isinstance(response, tuple) or response in (
            "predict_proba",
            "decision_function",
        ):
            proba.add(name)

        # *_samples scorers need multilabel data
        if name.endswith("_samples"):
            multilabel.add(name)
            continue

        # Categorize by module
        if "cluster" in module:
            cluster.add(name)
        elif "_classification" in module:
            classification.add(name)
        elif "_regression" in module:
            regression.add(name)
        elif "_ranking" in module:
            classification.add(name)  # ranking metrics are for classifiers
        elif "_scorer" in module:
            classification.add(name)  # likelihood ratios

    return {
        "classification": frozenset(classification),
        "regression": frozenset(regression),
        "cluster": frozenset(cluster),
        "proba": frozenset(proba),
        "multilabel": frozenset(multilabel),
    }


SCORERS_BY_TYPE = get_scorers_by_type()


class TestPipelineScoringMatchSklearn:
    """Tests for pipeline scoring with all compatible scorers."""

    @pytest.fixture
    def scoring_data(self):
        """Generate dataset suitable for classification, regression, and clustering."""
        import numpy as np

        np.random.seed(42)
        n = 100
        return {
            "x1": np.random.randn(n).tolist(),
            "x2": np.random.randn(n).tolist(),
            "y_class": (np.random.randn(n) > 0).astype(int).tolist(),
            "y_reg": (
                np.abs(np.random.randn(n)) + 0.1
            ).tolist(),  # positive for log/deviance scorers
        }

    @pytest.mark.parametrize("scorer_name", SCORERS_BY_TYPE["classification"])
    def test_classifier_scorer(self, scoring_data, scorer_name):
        """Test classification scorers match sklearn."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import get_scorer
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        X = np.array([scoring_data["x1"], scoring_data["x2"]]).T
        y = np.array(scoring_data["y_class"])

        sklearn_pipe = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )
        sklearn_pipe.fit(X, y)

        t = xo.memtable(scoring_data)
        xorq_pipeline = xo.Pipeline.from_instance(sklearn_pipe)
        fitted_xorq = xorq_pipeline.fit(t, features=("x1", "x2"), target="y_class")

        scorer = get_scorer(scorer_name)
        sklearn_score = scorer(sklearn_pipe, X, y)
        xorq_score = fitted_xorq.score(X, y, scorer=scorer_name)

        np.testing.assert_allclose(xorq_score, sklearn_score, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("scorer_name", SCORERS_BY_TYPE["regression"])
    def test_regressor_scorer(self, scoring_data, scorer_name):
        """Test regression scorers match sklearn."""
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import get_scorer
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        X = np.array([scoring_data["x1"], scoring_data["x2"]]).T
        y = np.array(scoring_data["y_reg"])

        sklearn_pipe = SklearnPipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        )
        sklearn_pipe.fit(X, y)

        t = xo.memtable(scoring_data)
        xorq_pipeline = xo.Pipeline.from_instance(sklearn_pipe)
        fitted_xorq = xorq_pipeline.fit(t, features=("x1", "x2"), target="y_reg")

        scorer = get_scorer(scorer_name)
        sklearn_score = scorer(sklearn_pipe, X, y)
        xorq_score = fitted_xorq.score(X, y, scorer=scorer_name)

        np.testing.assert_allclose(xorq_score, sklearn_score, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("scorer_name", SCORERS_BY_TYPE["cluster"])
    def test_cluster_scorer(self, scoring_data, scorer_name):
        """Test clustering scorers match sklearn."""
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import get_scorer
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        X = np.array([scoring_data["x1"], scoring_data["x2"]]).T
        y = np.array(scoring_data["y_class"])

        sklearn_pipe = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clusterer", KMeans(n_clusters=2, random_state=42, n_init=10)),
            ]
        )
        sklearn_pipe.fit(X, y)

        t = xo.memtable(scoring_data)
        xorq_pipeline = xo.Pipeline.from_instance(sklearn_pipe)
        fitted_xorq = xorq_pipeline.fit(t, features=("x1", "x2"), target="y_class")

        scorer = get_scorer(scorer_name)
        sklearn_score = scorer(sklearn_pipe, X, y)
        xorq_score = fitted_xorq.score(X, y, scorer=scorer_name)

        np.testing.assert_allclose(xorq_score, sklearn_score, rtol=1e-9, atol=1e-12)


class TestScoreExpr:
    """Tests for FittedPipeline.score_expr and .score edge cases."""

    @pytest.fixture
    def fitted_classifier(self):
        """Fitted classifier pipeline."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        np.random.seed(42)
        data = {"x1": [0.0, 1.0], "x2": [0.0, 1.0], "y": [0, 1]}
        X = np.array([data["x1"], data["x2"]]).T
        y = np.array(data["y"])

        sklearn_pipe = SklearnPipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression())]
        )
        sklearn_pipe.fit(X, y)

        t = xo.memtable(data)
        fitted = xo.Pipeline.from_instance(sklearn_pipe).fit(
            t, features=("x1", "x2"), target="y"
        )
        return fitted, sklearn_pipe, X, y, t

    @pytest.fixture
    def fitted_regressor(self):
        """Fitted regressor pipeline."""
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        np.random.seed(42)
        data = {"x1": [0.0, 1.0], "x2": [0.0, 1.0], "y": [0.0, 1.0]}
        X = np.array([data["x1"], data["x2"]]).T
        y = np.array(data["y"])

        sklearn_pipe = SklearnPipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        )
        sklearn_pipe.fit(X, y)

        t = xo.memtable(data)
        fitted = xo.Pipeline.from_instance(sklearn_pipe).fit(
            t, features=("x1", "x2"), target="y"
        )
        return fitted, sklearn_pipe, X, y, t

    def test_default_scorer_classifier_is_accuracy(self, fitted_classifier):
        """Test default scorer for classifier is accuracy_score."""
        from sklearn.metrics import accuracy_score

        fitted, *_ = fitted_classifier
        scorer = fitted._get_default_scorer()
        assert scorer._score_func is accuracy_score

    def test_default_scorer_regressor_is_r2(self, fitted_regressor):
        """Test default scorer for regressor is r2_score."""
        from sklearn.metrics import r2_score

        fitted, *_ = fitted_regressor
        scorer = fitted._get_default_scorer()
        assert scorer._score_func is r2_score

    def test_default_scorer_cluster_is_adjusted_rand(self):
        """Test default scorer for clustering is adjusted_rand_score."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        t = xo.memtable({"x": [0.0, 1.0], "y": [0, 1]})
        fitted = xo.Pipeline.from_instance(
            sklearn.pipeline.make_pipeline(KMeans(n_clusters=2, n_init=1))
        ).fit(t, features=("x",), target="y")

        assert fitted._get_default_scorer()._score_func is adjusted_rand_score

    def test_string_scorer(self, fitted_classifier):
        """Test passing a scorer name string."""
        import numpy as np
        from sklearn.metrics import get_scorer

        fitted, sklearn_pipe, X, y, _ = fitted_classifier

        xorq_score = fitted.score(X, y, scorer="f1")
        sklearn_score = get_scorer("f1")(sklearn_pipe, X, y)
        np.testing.assert_allclose(xorq_score, sklearn_score, rtol=1e-9)

    def test_callable_scorer(self, fitted_classifier):
        """Test passing a raw callable metric function."""
        import numpy as np
        from sklearn.metrics import f1_score

        fitted, sklearn_pipe, X, y, _ = fitted_classifier

        xorq_score = fitted.score(X, y, scorer=f1_score)
        sklearn_score = f1_score(y, sklearn_pipe.predict(X))
        np.testing.assert_allclose(xorq_score, sklearn_score, rtol=1e-9)

    def test_make_scorer_object(self, fitted_classifier):
        """Test passing a make_scorer object directly."""
        import numpy as np
        from sklearn.metrics import f1_score, make_scorer

        fitted, sklearn_pipe, X, y, _ = fitted_classifier

        scorer = make_scorer(f1_score)
        xorq_score = fitted.score(X, y, scorer=scorer)
        sklearn_score = scorer(sklearn_pipe, X, y)
        np.testing.assert_allclose(xorq_score, sklearn_score, rtol=1e-9)

    def test_score_expr_returns_expression(self, fitted_classifier):
        """Test score_expr returns an ibis expression."""
        from xorq.vendor.ibis.expr.types import Expr

        fitted, _, _, _, t = fitted_classifier
        expr = fitted.score_expr(t, scorer="accuracy")

        assert isinstance(expr, Expr)
        assert isinstance(expr.execute(), (int, float))

    def test_default_scorer_raises_for_unknown_model(self):
        """Test _get_default_scorer raises ValueError for unknown model type."""
        from sklearn.base import BaseEstimator

        import xorq.expr.datatypes as dt

        # Custom estimator that isn't a Classifier/Regressor/Cluster
        class CustomEstimator(BaseEstimator):
            return_type = dt.int64  # needed for xorq to handle predict

            def fit(self, X, y=None):
                return self

            def predict(self, X):
                return [0] * len(X)

        t = xo.memtable({"x": [0.0, 1.0], "y": [0, 1]})
        fitted = xo.Pipeline.from_instance(
            sklearn.pipeline.make_pipeline(CustomEstimator())
        ).fit(t, features=("x",), target="y")

        with pytest.raises(ValueError, match="Cannot determine default scorer"):
            fitted._get_default_scorer()


class TestClusteringPredict:
    """Tests for clustering algorithm predict support."""

    @pytest.fixture
    def cluster_data(self):
        """Generate data with clear cluster structure."""
        import numpy as np

        np.random.seed(42)
        # Two well-separated clusters
        cluster1 = np.random.randn(10, 2) + [0, 0]
        cluster2 = np.random.randn(10, 2) + [10, 10]
        data = np.vstack([cluster1, cluster2])
        return {"num1": data[:, 0].tolist(), "num2": data[:, 1].tolist()}

    @pytest.mark.parametrize(
        "clusterer_cls,clusterer_kwargs",
        [
            pytest.param(
                "KMeans",
                {"n_clusters": 2, "random_state": 42, "n_init": 10},
                id="KMeans",
            ),
            pytest.param(
                "MiniBatchKMeans",
                {"n_clusters": 2, "random_state": 42, "n_init": 10},
                id="MiniBatchKMeans",
            ),
            pytest.param(
                "BisectingKMeans",
                {"n_clusters": 2, "random_state": 42},
                id="BisectingKMeans",
            ),
            pytest.param(
                "Birch",
                {"n_clusters": 2},
                id="Birch",
            ),
            pytest.param(
                "MeanShift",
                {},
                id="MeanShift",
            ),
            pytest.param(
                "AffinityPropagation",
                {"random_state": 42},
                id="AffinityPropagation",
            ),
        ],
    )
    def test_inductive_clustering_predict(
        self, cluster_data, clusterer_cls, clusterer_kwargs
    ):
        """Test that inductive clustering algorithms support predict."""
        import numpy as np
        from sklearn import cluster

        t = xo.memtable(cluster_data)
        features = ("num1", "num2")

        ClustererClass = getattr(cluster, clusterer_cls)
        clusterer = ClustererClass(**clusterer_kwargs)

        # xorq predict
        step = xo.Step.from_instance_name(clusterer, name="clusterer")
        fitted = step.fit(t, features=features)
        result = fitted.predict(t)
        xorq_labels = result.execute()["predicted"].values

        # sklearn predict
        X = np.array([cluster_data["num1"], cluster_data["num2"]]).T
        sklearn_clusterer = ClustererClass(**clusterer_kwargs)
        sklearn_clusterer.fit(X)
        sklearn_labels = sklearn_clusterer.predict(X)

        # Labels should match
        np.testing.assert_array_equal(xorq_labels, sklearn_labels)

    @pytest.mark.parametrize(
        "clusterer_cls,clusterer_kwargs",
        [
            pytest.param(
                "DBSCAN",
                {"eps": 3, "min_samples": 2},
                id="DBSCAN",
            ),
            pytest.param(
                "HDBSCAN",
                {"min_samples": 2},
                id="HDBSCAN",
            ),
            pytest.param(
                "AgglomerativeClustering",
                {"n_clusters": 2},
                id="AgglomerativeClustering",
            ),
            pytest.param(
                "SpectralClustering",
                {"n_clusters": 2, "random_state": 42},
                id="SpectralClustering",
            ),
            pytest.param(
                "OPTICS",
                {"min_samples": 2},
                id="OPTICS",
            ),
        ],
    )
    def test_transductive_clustering_rejected_at_fit(
        self, cluster_data, clusterer_cls, clusterer_kwargs
    ):
        """Test that transductive clustering algorithms are rejected at fit time."""
        from sklearn import cluster

        t = xo.memtable(cluster_data)
        features = ("num1", "num2")

        ClustererClass = getattr(cluster, clusterer_cls)
        clusterer = ClustererClass(**clusterer_kwargs)

        step = xo.Step.from_instance_name(clusterer, name="clusterer")

        with pytest.raises(ValueError, match="must have transform or predict method"):
            step.fit(t, features=features)
