import operator
from numbers import Real

import numpy as np
import pandas as pd
import pytest

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.ml.enums import ResponseMethod
from xorq.expr.ml.pipeline_lib import Pipeline, Step, make_estimator_typ
from xorq.expr.relations import Tag
from xorq.vendor.ibis.common.collections import FrozenOrderedDict
from xorq.vendor.ibis.expr.types import Expr


sklearn = pytest.importorskip("sklearn")

from xorq.expr.ml.sklearn_utils import ColumnRemapper  # noqa: E402


# sklearn submodule imports
KMeans = sklearn.cluster.KMeans
MiniBatchKMeans = sklearn.cluster.MiniBatchKMeans
RandomForestClassifier = sklearn.ensemble.RandomForestClassifier
SimpleImputer = sklearn.impute.SimpleImputer
LinearRegression = sklearn.linear_model.LinearRegression
LogisticRegression = sklearn.linear_model.LogisticRegression
accuracy_score = sklearn.metrics.accuracy_score
adjusted_rand_score = sklearn.metrics.adjusted_rand_score
f1_score = sklearn.metrics.f1_score
get_scorer = sklearn.metrics.get_scorer
get_scorer_names = sklearn.metrics.get_scorer_names
make_scorer = sklearn.metrics.make_scorer
r2_score = sklearn.metrics.r2_score
SklearnPipeline = sklearn.pipeline.Pipeline
ColumnTransformer = sklearn.compose.ColumnTransformer
SelectKBest = sklearn.feature_selection.SelectKBest
f_classif = sklearn.feature_selection.f_classif
FeatureUnion = sklearn.pipeline.FeatureUnion
OneHotEncoder = sklearn.preprocessing.OneHotEncoder
StandardScaler = sklearn.preprocessing.StandardScaler
BaseEstimator = sklearn.base.BaseEstimator
cluster = sklearn.cluster

TARGET = "target"
features = (feature0, feature1) = ("feature_0", "feature_1")

get_metadata = operator.attrgetter("metadata")


@pytest.fixture(scope="module")
def t_test():
    return xo.memtable(
        {
            feature0: [5, 6],
            feature1: [7, 8],
            TARGET: [1, 0],
        }
    )


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
def xorq_pipeline(sklearn_pipeline):
    xorq_pipeline = xo.Pipeline.from_instance(sklearn_pipeline)
    return xorq_pipeline


@pytest.fixture(scope="module")
def fitted_xorq_pipeline(xorq_pipeline, t):
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


def test_recover_pipeline_from_transform_expr(t, xorq_pipeline, fitted_xorq_pipeline):
    expr = fitted_xorq_pipeline.transform(t)
    assert expr.ls.pipeline == xorq_pipeline


def test_recover_pipeline_from_predict_expr(t, xorq_pipeline, fitted_xorq_pipeline):
    expr = fitted_xorq_pipeline.predict(t)
    assert expr.ls.pipeline == xorq_pipeline


def test_fitted_pipeline_pipeline_property(xorq_pipeline, fitted_xorq_pipeline):
    assert fitted_xorq_pipeline.pipeline == xorq_pipeline


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
def test_tagging_pipeline(pairs, t_test, fitted_xorq_pipeline):
    def contains_any_pairs(d, pairs=pairs):
        return set(pairs).intersection(d.items())

    def sort_and_tuplify(dcts):
        return tuple(sorted(tuple(sorted(dct.items())) for dct in dcts))

    actual = sort_and_tuplify(
        map(
            get_metadata,
            fitted_xorq_pipeline.predict(t_test).ls.get_tags(
                predicate=contains_any_pairs,
            ),
        )
    )
    expected = sort_and_tuplify(
        dct
        for dct in (
            fitted_step.get_tag_kwargs()
            for fitted_step in fitted_xorq_pipeline.fitted_steps
        )
        if contains_any_pairs(dct)
    )
    assert actual and set(actual) == set(expected)


def test_score_expr_returns_metric(t, fitted_xorq_pipeline):
    score_expr = fitted_xorq_pipeline.score_expr(t)
    result = score_expr.execute()
    assert isinstance(result, Real)


def test_fitted_step_transform_known_schema_unpacks():
    """Test FittedStep.transform unpacks struct columns for known schema."""

    t = xo.memtable({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    step = xo.Step.from_instance_name(StandardScaler(), name="scaler")
    fitted = step.fit(t, features=("a", "b"))

    result = fitted.transform(t)
    df = result.execute()

    # Should have unpacked columns a and b, not a struct column
    assert "a" in df.columns
    assert "b" in df.columns
    assert "transformed" not in df.columns


def test_fitted_step_transform_kv_encoded_no_unpack():
    """Test FittedStep.transform keeps KV-encoded column without unpacking."""

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


def test_fitted_step_transform_retain_others_true():
    """Test FittedStep.transform retains other columns by default."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "other": ["x", "y"]})
    step = xo.Step.from_instance_name(StandardScaler(), name="scaler")
    fitted = step.fit(t, features=("a", "b"))

    result = fitted.transform(t, retain_others=True)
    df = result.execute()

    # Should retain the "other" column
    assert "other" in df.columns
    assert df["other"].tolist() == ["x", "y"]


def test_fitted_step_transform_retain_others_false():
    """Test FittedStep.transform drops other columns when retain_others=False."""

    t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "other": ["x", "y"]})
    step = xo.Step.from_instance_name(StandardScaler(), name="scaler")
    fitted = step.fit(t, features=("a", "b"))

    result = fitted.transform(t, retain_others=False)
    df = result.execute()

    # Should not retain the "other" column
    assert "other" not in df.columns


def test_pipeline_get_output_columns_known_schema_features_propagate():
    """Test Pipeline correctly propagates features for known schema transformers."""

    t = xo.memtable({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "y": [0.0, 1.0, 0.0]})
    pipeline = xo.Pipeline.from_instance(
        sklearn.pipeline.make_pipeline(StandardScaler(), LinearRegression())
    )
    fitted = pipeline.fit(t, target="y")

    # The transform step should have features = ("a", "b") from the known schema
    transform_step = fitted.transform_steps[0]
    assert transform_step.structer.get_output_columns() == ("a", "b")


def test_pipeline_get_output_columns_kv_encoded_features_use_dest_col():
    """Test Pipeline correctly uses dest_col for KV-encoded transformers."""

    t = xo.memtable({"cat": ["a", "b", "a"], "y": [0.0, 1.0, 0.0]})
    pipeline = xo.Pipeline.from_instance(
        sklearn.pipeline.make_pipeline(OneHotEncoder(), LinearRegression())
    )
    fitted = pipeline.fit(t, target="y")

    # The transform step should have features = ("transformed",) for KV-encoded
    transform_step = fitted.transform_steps[0]
    assert transform_step.structer.get_output_columns("transformed") == ("transformed",)


def test_pipeline_get_output_columns_mixed_transform_steps():
    """Test Pipeline with multiple transform steps propagates features correctly."""

    t = xo.memtable({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0], "y": [0.0, 1.0, 0.0]})
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


def test_deeply_nested_kv_encoded_pipeline():
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
            "region": np.random.choice(["north", "south", "east", "west"], n_samples),
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
    fitted_pipeline = xorq_pipeline.fit(expr, features=all_features, target="approved")
    predictions = fitted_pipeline.predict(expr).execute()

    # Fit and predict with sklearn
    X = data[list(all_features)]
    y = data["approved"]
    sklearn_pipe.fit(X, y)
    sklearn_preds = sklearn_pipe.predict(X)

    # Assert predictions match
    assert np.array_equal(predictions[ResponseMethod.PREDICT].values, sklearn_preds)


def test_deeply_nested_non_kv_pipeline():
    """Test depth-4 nested pipeline with all known-schema transformers.

    Pipeline structure:
    - ColumnTransformer (known schema - no KV-encoded children)
      - Pipeline (SimpleImputer -> StandardScaler -> Pipeline)
        - Pipeline (SimpleImputer -> StandardScaler)
      - Pipeline (SimpleImputer -> StandardScaler)
    - RandomForestClassifier
    """

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
    fitted_pipeline = xorq_pipeline.fit(expr, features=all_features, target="approved")
    predictions = fitted_pipeline.predict(expr).execute()

    # Fit and predict with sklearn
    X = data[list(all_features)]
    y = data["approved"]
    sklearn_pipe.fit(X, y)
    sklearn_preds = sklearn_pipe.predict(X)

    # Assert predictions match
    assert np.array_equal(predictions[ResponseMethod.PREDICT].values, sklearn_preds)


def _scorer_info():
    """Return [(name, module, response_method), ...] for every registered scorer."""

    return [
        (
            name,
            get_scorer(name)._score_func.__module__,
            get_scorer(name)._response_method,
        )
        for name in get_scorer_names()
    ]


def get_scorers_by_type():
    """Categorize all sklearn scorers by their type based on internal module path."""
    info = _scorer_info()

    proba = {
        name
        for name, _, response in info
        if isinstance(response, tuple)
        or response in (ResponseMethod.PREDICT_PROBA, ResponseMethod.DECISION_FUNCTION)
    }

    multilabel = {name for name, _, _ in info if name.endswith("_samples")}

    # Non-multilabel scorers categorized by module
    non_ml = [
        (name, module) for name, module, _ in info if not name.endswith("_samples")
    ]

    classification = {
        name
        for name, module in non_ml
        if "_classification" in module or "_ranking" in module or "_scorer" in module
    }

    regression = {name for name, module in non_ml if "_regression" in module}

    cluster = {name for name, module in non_ml if "cluster" in module}

    return FrozenOrderedDict(
        **{
            k: tuple(sorted(v))
            for k, v in (
                ("classification", classification),
                ("regression", regression),
                ("cluster", cluster),
                ("proba", proba),
                ("multilabel", multilabel),
            )
        }
    )


SCORERS_BY_TYPE = get_scorers_by_type()


@pytest.fixture
def scoring_data():
    """Generate dataset suitable for classification, regression, and clustering."""

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
def test_pipeline_scoring_match_sklearn_classifier_scorer(scoring_data, scorer_name):
    """Test classification scorers match sklearn."""

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
def test_pipeline_scoring_match_sklearn_regressor_scorer(scoring_data, scorer_name):
    """Test regression scorers match sklearn."""

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
def test_pipeline_scoring_match_sklearn_cluster_scorer(scoring_data, scorer_name):
    """Test clustering scorers match sklearn."""

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


@pytest.fixture
def fitted_classifier():
    """Fitted classifier pipeline."""

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
def fitted_regressor():
    """Fitted regressor pipeline."""

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


def test_score_expr_default_scorer_classifier_is_accuracy(fitted_classifier):
    """Test default scorer for classifier is accuracy_score."""

    fitted, *_ = fitted_classifier
    scorer = fitted._get_default_scorer()
    assert scorer._score_func is accuracy_score


def test_score_expr_default_scorer_regressor_is_r2(fitted_regressor):
    """Test default scorer for regressor is r2_score."""

    fitted, *_ = fitted_regressor
    scorer = fitted._get_default_scorer()
    assert scorer._score_func is r2_score


def test_score_expr_default_scorer_cluster_is_adjusted_rand():
    """Test default scorer for clustering is adjusted_rand_score."""

    t = xo.memtable({"x": [0.0, 1.0], "y": [0, 1]})
    fitted = xo.Pipeline.from_instance(
        sklearn.pipeline.make_pipeline(KMeans(n_clusters=2, n_init=1))
    ).fit(t, features=("x",), target="y")

    assert fitted._get_default_scorer()._score_func is adjusted_rand_score


def test_score_expr_string_scorer(fitted_classifier):
    """Test passing a scorer name string."""

    fitted, sklearn_pipe, X, y, _ = fitted_classifier

    xorq_score = fitted.score(X, y, scorer="f1")
    sklearn_score = get_scorer("f1")(sklearn_pipe, X, y)
    np.testing.assert_allclose(xorq_score, sklearn_score, rtol=1e-9)


def test_score_expr_callable_scorer(fitted_classifier):
    """Test passing a raw callable metric function."""

    fitted, sklearn_pipe, X, y, _ = fitted_classifier

    xorq_score = fitted.score(X, y, scorer=f1_score)
    sklearn_score = f1_score(y, sklearn_pipe.predict(X))
    np.testing.assert_allclose(xorq_score, sklearn_score, rtol=1e-9)


def test_score_expr_make_scorer_object(fitted_classifier):
    """Test passing a make_scorer object directly."""

    fitted, sklearn_pipe, X, y, _ = fitted_classifier

    scorer = make_scorer(f1_score)
    xorq_score = fitted.score(X, y, scorer=scorer)
    sklearn_score = scorer(sklearn_pipe, X, y)
    np.testing.assert_allclose(xorq_score, sklearn_score, rtol=1e-9)


def test_score_expr_returns_expression(fitted_classifier):
    """Test score_expr returns an ibis expression."""

    fitted, _, _, _, t = fitted_classifier
    expr = fitted.score_expr(t, scorer="accuracy")

    assert isinstance(expr, Expr)
    assert isinstance(expr.execute(), (int, float))


def test_score_expr_default_scorer_raises_for_unknown_model():
    """Test _get_default_scorer raises ValueError for unknown model type."""

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


def test_step_from_fit_transform_creates_transform_type():
    """Step.from_fit_transform creates a type with transform (not predict)."""

    def my_fit(X, y=None):
        return np.mean(X, axis=0)

    def my_transform(model, X, y=None):
        return X - model

    step = Step.from_fit_transform(
        fit=my_fit,
        transform=my_transform,
        return_type=dt.Array(dt.float64),
        name="custom_transform",
    )

    # Verify the dynamically created type has correct attributes
    assert hasattr(step.instance, "transform")
    assert hasattr(step.instance, "fit")
    assert not hasattr(step.instance, "predict")
    assert step.instance.return_type == dt.Array(dt.float64)


def test_step_from_fit_predict_creates_predict_type():
    """Step.from_fit_predict creates a type with predict (not transform)."""

    def my_fit(X, y=None):
        return int(np.median(y))

    def my_predict(model, X, y=None):
        return np.full(len(X), model)

    step = Step.from_fit_predict(
        fit=my_fit,
        predict=my_predict,
        return_type=dt.int64,
        name="custom_predict",
    )

    assert hasattr(step.instance, "predict")
    assert hasattr(step.instance, "fit")
    assert not hasattr(step.instance, "transform")
    assert step.instance.return_type == dt.int64


def test_step_from_fit_make_estimator_typ_both_raises():
    """Passing both transform and predict raises ValueError."""

    with pytest.raises(ValueError):
        make_estimator_typ(
            fit=lambda X, y=None: None,
            return_type=dt.float64,
            transform=lambda m, X: X,
            predict=lambda m, X: X,
        )


def test_step_from_fit_make_estimator_typ_neither_raises():
    """Passing neither transform nor predict raises ValueError."""

    with pytest.raises(ValueError):
        make_estimator_typ(
            fit=lambda X, y=None: None,
            return_type=dt.float64,
        )


def test_step_from_fit_predict_end_to_end():
    """Step.from_fit_predict works end-to-end with dest_col."""

    def my_fit(X, y=None):
        return int(np.median(y))

    def my_predict(model, X, y=None):
        return np.full(len(X), model)

    step = Step.from_fit_predict(
        fit=my_fit,
        predict=my_predict,
        return_type=dt.int64,
        name="custom_predict",
    )

    t = xo.memtable({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "y": [0, 1, 1]})
    fitted = step.fit(t, features=("a", "b"), target="y", dest_col="pred")
    result = fitted.predict(t)
    df = result.execute()
    assert df is not None
    assert len(df) == 3


def test_feature_importances_fitted_step():
    """FittedStep.feature_importances returns importances for tree models."""

    t = xo.memtable(
        {
            "a": np.random.randn(50).tolist(),
            "b": np.random.randn(50).tolist(),
            "y": (np.random.randn(50) > 0).astype(int).tolist(),
        }
    )

    step = xo.Step.from_instance_name(
        RandomForestClassifier(n_estimators=5, random_state=42),
        name="rf",
    )
    fitted = step.fit(t, features=("a", "b"), target="y")
    result = fitted.feature_importances(t)
    df = result.execute()

    assert df is not None
    assert "feature_importances" in df.columns
    importances = df["feature_importances"].iloc[0]
    assert len(importances) == 2  # two features
    assert all(isinstance(v, float) for v in importances)


def test_feature_importances_fitted_pipeline():
    """FittedPipeline.feature_importances returns importances through pipeline."""

    t = xo.memtable(
        {
            "a": np.random.randn(50).tolist(),
            "b": np.random.randn(50).tolist(),
            "y": (np.random.randn(50) > 0).astype(int).tolist(),
        }
    )

    sklearn_pipe = SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=5, random_state=42)),
        ]
    )
    fitted = Pipeline.from_instance(sklearn_pipe).fit(
        t, features=("a", "b"), target="y"
    )

    result = fitted.feature_importances(t)
    df = result.execute()

    assert df is not None
    assert "feature_importances" in df.columns
    importances = df["feature_importances"].iloc[0]
    assert len(importances) == 2


@pytest.fixture
def cluster_data():
    """Generate data with clear cluster structure."""

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
def test_clustering_predict_inductive(cluster_data, clusterer_cls, clusterer_kwargs):
    """Test that inductive clustering algorithms support predict."""

    t = xo.memtable(cluster_data)
    features = ("num1", "num2")

    ClustererClass = getattr(cluster, clusterer_cls)
    clusterer = ClustererClass(**clusterer_kwargs)

    # xorq predict
    step = xo.Step.from_instance_name(clusterer, name="clusterer")
    fitted = step.fit(t, features=features)
    result = fitted.predict(t)
    xorq_labels = result.execute()[ResponseMethod.PREDICT].values

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
def test_clustering_predict_transductive_rejected_at_fit(
    cluster_data, clusterer_cls, clusterer_kwargs
):
    """Test that transductive clustering algorithms are rejected at fit time."""

    t = xo.memtable(cluster_data)
    features = ("num1", "num2")

    ClustererClass = getattr(cluster, clusterer_cls)
    clusterer = ClustererClass(**clusterer_kwargs)

    step = xo.Step.from_instance_name(clusterer, name="clusterer")

    with pytest.raises(ValueError, match="must have transform or predict method"):
        step.fit(t, features=features)


# ---------------------------------------------------------------------------
# remap_columns / remap_params
# ---------------------------------------------------------------------------

Ridge = sklearn.linear_model.Ridge
SimpleImputer = sklearn.impute.SimpleImputer


@pytest.fixture
def ct_pipeline():
    """Pipeline with a ColumnTransformer having num and cat slots."""
    numeric_transformer = SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, ["age", "fare"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["embarked", "sex"],
            ),
        ]
    )
    sk = SklearnPipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )
    return Pipeline.from_instance(sk)


def test_remap_columns_basic(ct_pipeline):
    remapped = ct_pipeline.remap_columns(
        {
            "preprocessor/num": ["distance", "flight_time"],
            "preprocessor/cat": ["carrier", "origin"],
        }
    )
    refs = ColumnRemapper.list_column_refs(remapped.instance)
    assert refs["preprocessor/num"] == ["distance", "flight_time"]
    assert refs["preprocessor/cat"] == ["carrier", "origin"]


def test_remap_columns_original_unchanged(ct_pipeline):
    ct_pipeline.remap_columns(
        {"preprocessor/num": ["x", "y"], "preprocessor/cat": ["z", "w"]}
    )
    refs = ColumnRemapper.list_column_refs(ct_pipeline.instance)
    assert refs["preprocessor/num"] == ["age", "fare"]
    assert refs["preprocessor/cat"] == ["embarked", "sex"]


def test_remap_columns_returns_pipeline(ct_pipeline):
    remapped = ct_pipeline.remap_columns(
        {"preprocessor/num": ["a", "b"], "preprocessor/cat": ["c"]}
    )
    assert isinstance(remapped, Pipeline)


def test_remap_columns_unknown_key_raises(ct_pipeline):
    with pytest.raises(ValueError, match="column_map keys not found"):
        ct_pipeline.remap_columns({"preprocessor/typo": ["a", "b"]})


def test_remap_columns_strict_raises_on_unmapped(ct_pipeline):
    with pytest.raises(ValueError, match="pipeline paths not covered"):
        ct_pipeline.remap_columns(
            {"preprocessor/num": ["a", "b"]},
            strict=True,
        )


def test_remap_columns_strict_passes_when_all_mapped(ct_pipeline):
    remapped = ct_pipeline.remap_columns(
        {"preprocessor/num": ["a", "b"], "preprocessor/cat": ["c", "d"]},
        strict=True,
    )
    assert isinstance(remapped, Pipeline)


def test_remap_columns_nested_slot(ct_pipeline):
    refs = ColumnRemapper.list_column_refs(ct_pipeline.instance)
    assert "preprocessor/num/imputer" not in refs  # imputer has no cols list
    assert "preprocessor/num" in refs


def test_remap_params_leaf_param(ct_pipeline):
    remapped = ct_pipeline.remap_params({"classifier__C": 0.01})
    assert remapped.instance.get_params()["classifier__C"] == 0.01


def test_remap_params_nested_param(ct_pipeline):
    remapped = ct_pipeline.remap_params(
        {"preprocessor__num__imputer__strategy": "mean"}
    )
    assert (
        remapped.instance.get_params()["preprocessor__num__imputer__strategy"] == "mean"
    )


def test_remap_params_whole_step_replacement(ct_pipeline):
    remapped = ct_pipeline.remap_params({"classifier": Ridge(alpha=0.5)})
    assert type(remapped.instance.named_steps["classifier"]) is Ridge
    assert remapped.instance.named_steps["classifier"].alpha == 0.5


def test_remap_params_original_unchanged(ct_pipeline):
    ct_pipeline.remap_params({"classifier__C": 0.001})
    assert ct_pipeline.instance.get_params()["classifier__C"] == 1.0


def test_remap_params_returns_pipeline(ct_pipeline):
    remapped = ct_pipeline.remap_params({"classifier__C": 0.5})
    assert isinstance(remapped, Pipeline)


def test_remap_params_unknown_key_raises(ct_pipeline):
    with pytest.raises(ValueError, match="param_map keys not found"):
        ct_pipeline.remap_params({"classifier__typo": 99})
