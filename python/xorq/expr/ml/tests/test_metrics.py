"""Tests for deferred sklearn metrics evaluation using Pipeline API."""

import numpy as np
import pandas as pd
import pytest

import xorq.expr.datatypes as dt
from xorq.expr import api
from xorq.expr.ml.enums import ResponseMethod
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline


# Skip all tests in this module if sklearn is not installed
sklearn = pytest.importorskip("sklearn")

# Access sklearn modules through the sklearn object to avoid E402 linting errors
make_classification = sklearn.datasets.make_classification
make_regression = sklearn.datasets.make_regression
RandomForestClassifier = sklearn.ensemble.RandomForestClassifier
RandomForestRegressor = sklearn.ensemble.RandomForestRegressor
LinearRegression = sklearn.linear_model.LinearRegression
LogisticRegression = sklearn.linear_model.LogisticRegression
LinearSVC = sklearn.svm.LinearSVC
StandardScaler = sklearn.preprocessing.StandardScaler
SkPipeline = sklearn.pipeline.Pipeline
accuracy_score = sklearn.metrics.accuracy_score
mean_absolute_error = sklearn.metrics.mean_absolute_error
mean_squared_error = sklearn.metrics.mean_squared_error
precision_score = sklearn.metrics.precision_score
r2_score = sklearn.metrics.r2_score
recall_score = sklearn.metrics.recall_score
roc_auc_score = sklearn.metrics.roc_auc_score
f1_score = sklearn.metrics.f1_score
train_test_split = sklearn.model_selection.train_test_split


@pytest.fixture
def classification_data():
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=3,
        n_classes=2,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    return train_df, test_df, feature_names


@pytest.fixture
def multiclass_data():
    """Generate synthetic multiclass data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=3,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    return train_df, test_df, feature_names


@pytest.fixture
def regression_data():
    """Generate synthetic regression data."""
    X, y = make_regression(
        n_samples=1000, n_features=10, n_informative=5, noise=10.0, random_state=42
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    return train_df, test_df, feature_names


def test_pipeline_classification_metrics(classification_data):
    """Test Pipeline API with classification metrics."""
    train_df, test_df, feature_names = classification_data

    # Register data with xorq
    train_expr = api.register(train_df, "train")
    test_expr = api.register(test_df, "test")

    # Create sklearn pipeline with scaler and classifier
    sklearn_pipeline = SkPipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )

    # Wrap with xorq Pipeline
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    # Fit the pipeline
    fitted_pipeline = xorq_pipeline.fit(
        train_expr, features=feature_names, target="target"
    )

    # Direct sklearn metrics for comparison
    X_train = train_df[feature_names]
    y_train = train_df["target"]
    X_test = test_df[feature_names]
    y_test = test_df["target"]

    sklearn_pipeline.fit(X_train, y_train)
    y_pred = sklearn_pipeline.predict(X_test)

    expected_accuracy = accuracy_score(y_test, y_pred)
    expected_precision = precision_score(y_test, y_pred)
    expected_recall = recall_score(y_test, y_pred)
    expected_f1 = f1_score(y_test, y_pred)

    # Get predictions once for all metrics
    expr_with_preds = fitted_pipeline.predict(test_expr)

    # Test accuracy using Pipeline API
    accuracy_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=accuracy_score,
    )
    actual_accuracy = accuracy_expr.execute()
    assert np.isclose(actual_accuracy, expected_accuracy)

    # Test precision
    precision_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=precision_score,
    )
    actual_precision = precision_expr.execute()
    assert np.isclose(actual_precision, expected_precision)

    # Test recall
    recall_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=recall_score,
    )
    actual_recall = recall_expr.execute()
    assert np.isclose(actual_recall, expected_recall)

    # Test F1 score
    f1_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=f1_score,
    )
    actual_f1 = f1_expr.execute()
    assert np.isclose(actual_f1, expected_f1)


def test_pipeline_regression_metrics(regression_data):
    """Test Pipeline API with regression metrics."""
    train_df, test_df, feature_names = regression_data

    # Register data with xorq
    train_expr = api.register(train_df, "train")
    test_expr = api.register(test_df, "test")

    # Create sklearn pipeline with scaler and regressor
    sklearn_pipeline = SkPipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
    )

    # Wrap with xorq Pipeline
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    # Fit the pipeline
    fitted_pipeline = xorq_pipeline.fit(
        train_expr, features=feature_names, target="target"
    )

    # Direct sklearn metrics for comparison
    X_train = train_df[feature_names]
    y_train = train_df["target"]
    X_test = test_df[feature_names]
    y_test = test_df["target"]

    sklearn_pipeline.fit(X_train, y_train)
    y_pred = sklearn_pipeline.predict(X_test)

    expected_mse = mean_squared_error(y_test, y_pred)
    expected_mae = mean_absolute_error(y_test, y_pred)
    expected_r2 = r2_score(y_test, y_pred)

    # Get predictions once for all metrics
    expr_with_preds = fitted_pipeline.predict(test_expr)

    # Test MSE using Pipeline API
    mse_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=mean_squared_error,
    )
    actual_mse = mse_expr.execute()
    assert np.isclose(actual_mse, expected_mse, rtol=1e-4)

    # Test MAE using Pipeline API
    mae_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=mean_absolute_error,
    )
    actual_mae = mae_expr.execute()
    assert np.isclose(actual_mae, expected_mae, rtol=1e-4)

    # Test R2 using Pipeline API
    r2_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=r2_score,
    )
    actual_r2 = r2_expr.execute()
    assert np.isclose(actual_r2, expected_r2, rtol=1e-4)


def test_pipeline_multiclass_metrics(multiclass_data):
    """Test Pipeline API with multiclass metrics."""
    train_df, test_df, feature_names = multiclass_data

    # Register data with xorq
    train_expr = api.register(train_df, "train")
    test_expr = api.register(test_df, "test")

    # Create sklearn pipeline with scaler and classifier
    sklearn_pipeline = SkPipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    # Wrap with xorq Pipeline
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    # Fit the pipeline
    fitted_pipeline = xorq_pipeline.fit(
        train_expr, features=feature_names, target="target"
    )

    # Direct sklearn metrics for comparison
    X_train = train_df[feature_names]
    y_train = train_df["target"]
    X_test = test_df[feature_names]
    y_test = test_df["target"]

    sklearn_pipeline.fit(X_train, y_train)
    y_pred = sklearn_pipeline.predict(X_test)

    expected_accuracy = accuracy_score(y_test, y_pred)
    expected_precision_macro = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    expected_recall_weighted = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Get predictions once for all metrics
    expr_with_preds = fitted_pipeline.predict(test_expr)

    # Test accuracy
    accuracy_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=accuracy_score,
    )
    actual_accuracy = accuracy_expr.execute()
    assert np.isclose(actual_accuracy, expected_accuracy)

    # Test precision with macro averaging
    precision_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=precision_score,
        metric_kwargs={"average": "macro", "zero_division": 0},
    )
    actual_precision = precision_expr.execute()
    assert np.isclose(actual_precision, expected_precision_macro)

    # Test recall with weighted averaging
    recall_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=recall_score,
        metric_kwargs={"average": "weighted", "zero_division": 0},
    )
    actual_recall = recall_expr.execute()
    assert np.isclose(actual_recall, expected_recall_weighted)


def test_custom_metric_with_pipeline():
    """Test Pipeline API with a custom metric function."""
    # Create simple test data
    df = pd.DataFrame(
        {
            "feature_0": [1, 2, 3, 4, 5],
            "feature_1": [2, 3, 4, 5, 6],
            "target": [0, 0, 1, 1, 1],
        }
    )

    test_expr = api.register(df, "test")
    feature_names = ["feature_0", "feature_1"]

    # Create sklearn pipeline
    sklearn_pipeline = SkPipeline([("classifier", LogisticRegression(random_state=42))])

    # Wrap with xorq Pipeline
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    # Fit the pipeline
    fitted_pipeline = xorq_pipeline.fit(
        test_expr, features=feature_names, target="target"
    )

    # Define a custom metric wrapped with make_scorer
    from sklearn.metrics import make_scorer

    def custom_metric(y_true, y_pred):
        """A custom metric that returns accuracy * 100."""
        return accuracy_score(y_true, y_pred) * 100

    custom_scorer = make_scorer(custom_metric)

    # Get predictions
    expr_with_preds = fitted_pipeline.predict(test_expr)

    # Test custom metric via make_scorer
    custom_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=custom_scorer,
        return_type=dt.float64,
    )
    result = custom_expr.execute()

    # Should return a float between 0 and 100
    assert isinstance(result, (float, np.floating))
    assert 0 <= result <= 100


def test_deferred_nature_with_pipeline():
    """Test that the Pipeline API maintains deferred execution."""
    # Create simple test data
    df = pd.DataFrame(
        {
            "feature_0": range(100),
            "feature_1": range(100),
            "target": [i % 2 for i in range(100)],
        }
    )

    test_expr = api.register(df, "test")
    feature_names = ["feature_0", "feature_1"]

    # Create sklearn pipeline
    sklearn_pipeline = SkPipeline([("classifier", LogisticRegression(random_state=42))])

    # Wrap with xorq Pipeline
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    # Fit the pipeline
    fitted_pipeline = xorq_pipeline.fit(
        test_expr, features=feature_names, target="target"
    )

    # Get predictions
    expr_with_preds = fitted_pipeline.predict(test_expr)

    # Create a deferred metric expression
    accuracy_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=accuracy_score,
    )

    # The expression should have an execute method (deferred)
    assert hasattr(accuracy_expr, "execute")

    # It should not be a computed value yet
    assert not isinstance(accuracy_expr, (float, int, np.number))

    # Can create multiple metrics without execution
    precision_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=precision_score,
        metric_kwargs={"average": "macro", "zero_division": 0},
    )

    # Both should be expressions
    assert hasattr(precision_expr, "execute")

    # Now execute one
    accuracy_value = accuracy_expr.execute()
    assert isinstance(accuracy_value, (float, np.floating))

    # The other is still deferred
    assert hasattr(precision_expr, "execute")
    assert not isinstance(precision_expr, (float, int, np.number))


def test_predict_proba_metrics():
    """Test that probability-based metrics work with predict_proba."""
    # Create binary classification data
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, n_classes=2, random_state=42
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    train_df = df.iloc[:150]
    test_df = df.iloc[150:]

    train_expr = api.register(train_df, "train")
    test_expr = api.register(test_df, "test")

    # Create pipeline with classifier that supports predict_proba
    sklearn_pipeline = SkPipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    # Wrap with xorq Pipeline
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
    features = [f"feature_{i}" for i in range(X.shape[1])]

    # Fit the pipeline
    fitted_pipeline = xorq_pipeline.fit(train_expr, features=features, target="target")

    # Get sklearn predictions for comparison
    X_train = train_df[features].values
    y_train = train_df["target"].values
    X_test = test_df[features].values
    y_test = test_df["target"].values

    sklearn_pipeline.fit(X_train, y_train)
    y_proba = sklearn_pipeline.predict_proba(X_test)[:, 1]

    expected_auc = roc_auc_score(y_test, y_proba)

    # Test AUC with predict_proba - explicit API
    expr_with_proba = fitted_pipeline.predict_proba(test_expr)
    auc_expr = deferred_sklearn_metric(
        expr=expr_with_proba,
        target="target",
        pred=ResponseMethod.PREDICT_PROBA,
        metric=roc_auc_score,
    )

    actual_auc = auc_expr.execute()

    # Verify AUC matches
    assert np.isclose(actual_auc, expected_auc, rtol=1e-5), (
        f"AUC mismatch: {actual_auc} vs {expected_auc}"
    )

    # Also test that we can get the predict_proba expression directly
    proba_expr = fitted_pipeline.predict_proba(test_expr)
    assert hasattr(proba_expr, "execute")

    # Execute and check shape
    proba_result = proba_expr[ResponseMethod.PREDICT_PROBA].execute()
    assert len(proba_result) == len(test_df)


def test_probability_metric_without_predict_proba():
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, n_classes=2, random_state=7
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    train_df = df.iloc[:150]
    test_df = df.iloc[150:]

    train_expr = api.register(train_df, "svm_train")
    test_expr = api.register(test_df, "svm_test")

    features = [f"feature_{i}" for i in range(X.shape[1])]

    sklearn_pipeline = SkPipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(random_state=7, max_iter=5000)),
        ]
    )

    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
    fitted_pipeline = xorq_pipeline.fit(train_expr, features=features, target="target")

    X_train = train_df[features].values
    y_train = train_df["target"].values
    X_test = test_df[features].values
    y_test = test_df["target"].values

    sklearn_pipeline.fit(X_train, y_train)
    y_scores = sklearn_pipeline.decision_function(X_test)

    # ROC-AUC works with raw decision function scores for binary classification
    expected_auc = roc_auc_score(y_test, y_scores)

    # Use decision_function - works directly with ROC-AUC for binary
    expr_with_scores = fitted_pipeline.decision_function(test_expr)
    auc_expr = deferred_sklearn_metric(
        expr=expr_with_scores,
        target="target",
        pred=ResponseMethod.DECISION_FUNCTION,
        metric=roc_auc_score,
    )

    actual_auc = auc_expr.execute()
    assert np.isclose(actual_auc, expected_auc)


def test_probability_metric_multiclass_with_predict_proba():
    """Test multi-class ROC-AUC using a classifier with predict_proba.

    For multi-class ROC-AUC, sklearn requires probabilities that sum to 1.
    We use LogisticRegression which natively supports predict_proba.
    """
    X, y = make_classification(
        n_samples=240,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=11,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    train_df = df.iloc[:180]
    test_df = df.iloc[180:]

    train_expr = api.register(train_df, "multiclass_train")
    test_expr = api.register(test_df, "multiclass_test")

    features = [f"feature_{i}" for i in range(X.shape[1])]

    # Use LogisticRegression which has predict_proba
    sklearn_pipeline = SkPipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=11, max_iter=5000)),
        ]
    )

    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
    fitted_pipeline = xorq_pipeline.fit(train_expr, features=features, target="target")

    X_train = train_df[features].values
    y_train = train_df["target"].values
    X_test = test_df[features].values
    y_test = test_df["target"].values

    sklearn_pipeline.fit(X_train, y_train)
    sklearn_proba = sklearn_pipeline.predict_proba(X_test)

    expected_auc = roc_auc_score(y_test, sklearn_proba, multi_class="ovr")

    # Use predict_proba - sklearn handles normalization internally
    expr_with_proba = fitted_pipeline.predict_proba(test_expr)
    auc_expr = deferred_sklearn_metric(
        expr=expr_with_proba,
        target="target",
        pred=ResponseMethod.PREDICT_PROBA,
        metric=roc_auc_score,
        metric_kwargs={"multi_class": "ovr"},
    )

    actual_auc = auc_expr.execute()
    assert np.isclose(actual_auc, expected_auc)


def test_metric_kwargs_with_pipeline():
    """Test that metric kwargs work correctly with Pipeline API."""
    # Create simple test data
    df = pd.DataFrame(
        {
            "feature_0": range(20),
            "feature_1": range(20, 40),
            "target": [i % 3 for i in range(20)],  # 3 classes
        }
    )

    test_expr = api.register(df, "test")
    feature_names = ["feature_0", "feature_1"]

    # Create sklearn pipeline
    sklearn_pipeline = SkPipeline(
        [("classifier", LogisticRegression(random_state=42, max_iter=200))]
    )

    # Wrap with xorq Pipeline
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    # Fit the pipeline
    fitted_pipeline = xorq_pipeline.fit(
        test_expr, features=feature_names, target="target"
    )

    # Get predictions
    expr_with_preds = fitted_pipeline.predict(test_expr)

    # Test precision with different averaging methods
    precision_macro = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=precision_score,
        metric_kwargs={"average": "macro", "zero_division": 0},
    ).execute()

    precision_weighted = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=precision_score,
        metric_kwargs={"average": "weighted", "zero_division": 0},
    ).execute()

    # Different averaging should give different results
    assert precision_macro != precision_weighted

    # Both should be valid precision scores
    assert 0 <= precision_macro <= 1
    assert 0 <= precision_weighted <= 1


def test_sign_auto_detected():
    """Test that sign is automatically extracted from the scorer.

    Bare callable (mean_squared_error) resolves with sign=1.
    String scorer ("neg_mean_squared_error") resolves with sign=-1.
    """
    df = pd.DataFrame(
        {
            "feature_0": [0.0, 1.0, 2.0, 3.0],
            "feature_1": [0.0, 1.0, 2.0, 3.0],
            "target": [0.0, 1.0, 2.0, 3.0],
        }
    )

    test_expr = api.register(df, "sign_test")
    feature_names = ["feature_0", "feature_1"]

    sklearn_pipeline = SkPipeline([("regressor", LinearRegression())])
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
    fitted_pipeline = xorq_pipeline.fit(
        test_expr, features=feature_names, target="target"
    )

    expr_with_preds = fitted_pipeline.predict(test_expr)

    # Bare callable — sign=1 (raw metric value)
    mse_positive = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric=mean_squared_error,
    ).execute()

    # String scorer — sign=-1 (negated by convention)
    mse_negative = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred=ResponseMethod.PREDICT,
        metric="neg_mean_squared_error",
    ).execute()

    assert mse_negative == -mse_positive
    assert mse_positive >= 0  # MSE is always non-negative


def test_scorer_from_spec_string_input():
    """str -> resolved scorer with sign, kwargs, response_method."""
    from xorq.expr.ml.metrics import Scorer

    s = Scorer.from_spec("accuracy")
    assert s.metric_fn is accuracy_score
    assert s.sign == 1
    assert s.response_method == ResponseMethod.PREDICT


def test_scorer_from_spec_string_neg_scorer():
    """neg_* string -> sign=-1."""
    from xorq.expr.ml.metrics import Scorer

    s = Scorer.from_spec("neg_mean_squared_error")
    assert s.metric_fn is mean_squared_error
    assert s.sign == -1
    assert s.response_method == ResponseMethod.PREDICT


def test_scorer_from_spec_string_proba_scorer():
    """roc_auc string -> response_method from scorer."""
    from xorq.expr.ml.metrics import Scorer

    s = Scorer.from_spec("roc_auc")
    assert s.metric_fn is roc_auc_score
    assert s.sign == 1
    # roc_auc uses decision_function or predict_proba
    assert s.response_method in (
        ResponseMethod.DECISION_FUNCTION,
        ResponseMethod.PREDICT_PROBA,
    )


def test_scorer_from_spec_base_scorer_input():
    """_BaseScorer -> extracts metric_fn, sign, kwargs."""
    from sklearn.metrics import make_scorer

    from xorq.expr.ml.metrics import Scorer

    scorer_obj = make_scorer(accuracy_score)
    s = Scorer.from_spec(scorer_obj)
    assert s.metric_fn is accuracy_score
    assert s.sign == 1


def test_scorer_from_spec_known_callable():
    """Known bare callable -> sign=1, kwargs=(), response_method=predict."""
    from xorq.expr.ml.metrics import Scorer

    s = Scorer.from_spec(accuracy_score)
    assert s.metric_fn is accuracy_score
    assert s.sign == 1
    assert s.kwargs == ()
    assert s.response_method == ResponseMethod.PREDICT


def test_scorer_from_spec_unknown_callable_raises():
    """Unknown callable -> ValueError."""
    from sklearn.metrics import confusion_matrix

    from xorq.expr.ml.metrics import Scorer

    with pytest.raises(ValueError, match="not a known sklearn scorer function"):
        Scorer.from_spec(confusion_matrix)


def test_scorer_from_spec_scorer_passthrough():
    """Scorer input -> returns same instance."""
    from xorq.expr.ml.metrics import Scorer

    original = Scorer(
        metric_fn=accuracy_score,
        sign=1,
        kwargs={},
        response_method=ResponseMethod.PREDICT,
    )
    result = Scorer.from_spec(original)
    assert result is original


def test_scorer_from_spec_none_classifier():
    """None with classifier -> accuracy_score."""
    from sklearn.linear_model import LogisticRegression

    from xorq.expr.ml.metrics import Scorer

    model = LogisticRegression()
    model.fit([[0], [1]], [0, 1])
    s = Scorer.from_spec(None, model=model)
    assert s.metric_fn is accuracy_score


def test_scorer_from_spec_none_regressor():
    """None with regressor -> r2_score."""
    from sklearn.linear_model import LinearRegression

    from xorq.expr.ml.metrics import Scorer

    model = LinearRegression()
    model.fit([[0], [1]], [0.0, 1.0])
    s = Scorer.from_spec(None, model=model)
    assert s.metric_fn is r2_score


def test_scorer_from_spec_none_cluster():
    """None with clusterer -> adjusted_rand_score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    from xorq.expr.ml.metrics import Scorer

    model = KMeans(n_clusters=2, n_init=1)
    model.fit([[0], [1]])
    s = Scorer.from_spec(None, model=model)
    assert s.metric_fn is adjusted_rand_score


def test_scorer_from_spec_invalid_input_raises():
    """Non-callable, non-string -> ValueError."""
    from xorq.expr.ml.metrics import Scorer

    with pytest.raises(ValueError, match="scorer must be"):
        Scorer.from_spec(42)


def test_scorer_from_spec_string_neg_mean_absolute_error():
    """neg_mean_absolute_error string -> sign=-1, metric_fn=mean_absolute_error."""
    from xorq.expr.ml.metrics import Scorer

    s = Scorer.from_spec("neg_mean_absolute_error")
    assert s.metric_fn is mean_absolute_error
    assert s.sign == -1
    assert s.response_method == ResponseMethod.PREDICT


def test_scorer_from_spec_string_r2():
    """r2 string -> sign=1, metric_fn=r2_score."""
    from xorq.expr.ml.metrics import Scorer

    s = Scorer.from_spec("r2")
    assert s.metric_fn is r2_score
    assert s.sign == 1
    assert s.response_method == ResponseMethod.PREDICT


def test_scorer_from_spec_string_precision():
    """precision string -> sign=1, metric_fn=precision_score."""
    from xorq.expr.ml.metrics import Scorer

    s = Scorer.from_spec("precision")
    assert s.metric_fn is precision_score
    assert s.sign == 1
    assert s.response_method == ResponseMethod.PREDICT


def test_scorer_from_spec_string_recall():
    """recall string -> sign=1, metric_fn=recall_score."""
    from xorq.expr.ml.metrics import Scorer

    s = Scorer.from_spec("recall")
    assert s.metric_fn is recall_score
    assert s.sign == 1
    assert s.response_method == ResponseMethod.PREDICT


def test_scorer_from_spec_string_f1():
    """f1 string -> sign=1, metric_fn=f1_score."""
    from xorq.expr.ml.metrics import Scorer

    s = Scorer.from_spec("f1")
    assert s.metric_fn is f1_score
    assert s.sign == 1
    assert s.response_method == ResponseMethod.PREDICT


def test_scorer_from_spec_callable_mean_squared_error():
    """Known bare callable mean_squared_error -> sign=1."""
    from xorq.expr.ml.metrics import Scorer

    s = Scorer.from_spec(mean_squared_error)
    assert s.metric_fn is mean_squared_error
    assert s.sign == 1
    assert s.response_method == ResponseMethod.PREDICT


def test_scorer_from_spec_callable_r2_score():
    """Known bare callable r2_score -> sign=1."""
    from xorq.expr.ml.metrics import Scorer

    s = Scorer.from_spec(r2_score)
    assert s.metric_fn is r2_score
    assert s.sign == 1
    assert s.response_method == ResponseMethod.PREDICT


def test_scorer_from_spec_callable_roc_auc_score():
    """Known bare callable roc_auc_score -> sign=1, response_method=predict."""
    from xorq.expr.ml.metrics import Scorer

    s = Scorer.from_spec(roc_auc_score)
    assert s.metric_fn is roc_auc_score
    assert s.sign == 1
    # bare callable doesn't carry response_method metadata
    assert s.response_method == ResponseMethod.PREDICT


def test_scorer_from_spec_make_scorer_with_kwargs():
    """make_scorer with extra kwargs -> kwargs preserved."""
    from sklearn.metrics import make_scorer

    from xorq.expr.ml.metrics import Scorer

    scorer_obj = make_scorer(precision_score, average="macro", zero_division=0)
    s = Scorer.from_spec(scorer_obj)
    assert s.metric_fn is precision_score
    assert s.sign == 1


def test_scorer_from_spec_make_scorer_greater_is_better_false():
    """make_scorer(greater_is_better=False) -> sign=-1."""
    from sklearn.metrics import make_scorer

    from xorq.expr.ml.metrics import Scorer

    scorer_obj = make_scorer(mean_squared_error, greater_is_better=False)
    s = Scorer.from_spec(scorer_obj)
    assert s.metric_fn is mean_squared_error
    assert s.sign == -1


class TestCustomPredColName:
    """Test that custom name= on prediction methods flows through to deferred_sklearn_metric."""

    def test_classifier_custom_pred(self, classification_data):
        """Classifier: predict(name='my_pred') -> pred='my_pred'."""
        train_df, test_df, feature_names = classification_data
        train_expr = api.register(train_df, "cls_train")
        test_expr = api.register(test_df, "cls_test")

        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=feature_names, target="target"
        )

        preds = fitted.predict(test_expr, name="my_pred")
        assert "my_pred" in preds.columns

        result = deferred_sklearn_metric(
            expr=preds,
            target="target",
            pred="my_pred",
            metric=accuracy_score,
        ).execute()
        assert 0 <= result <= 1

    def test_regressor_custom_pred(self, regression_data):
        """Regressor: predict(name='my_pred') -> pred='my_pred'."""
        train_df, test_df, feature_names = regression_data
        train_expr = api.register(train_df, "reg_train")
        test_expr = api.register(test_df, "reg_test")

        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=feature_names, target="target"
        )

        preds = fitted.predict(test_expr, name="my_pred")
        assert "my_pred" in preds.columns

        result = deferred_sklearn_metric(
            expr=preds,
            target="target",
            pred="my_pred",
            metric=r2_score,
        ).execute()
        assert isinstance(result, (float, np.floating))

    def test_clusterer_custom_pred(self):
        """Clusterer: predict(name='my_labels') -> pred='my_labels'."""
        KMeans = sklearn.cluster.KMeans
        adjusted_rand_score = sklearn.metrics.adjusted_rand_score

        X, y = make_classification(
            n_samples=200, n_features=5, n_informative=3, n_classes=2, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]).assign(target=y)
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

        train_expr = api.register(train_df, "clu_train")
        test_expr = api.register(test_df, "clu_test")

        features = [f"f{i}" for i in range(5)]
        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("clu", KMeans(n_clusters=2, random_state=42, n_init=10)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=features, target="target"
        )

        preds = fitted.predict(test_expr, name="my_labels")
        assert "my_labels" in preds.columns

        result = deferred_sklearn_metric(
            expr=preds,
            target="target",
            pred="my_labels",
            metric=adjusted_rand_score,
        ).execute()
        assert isinstance(result, (float, np.floating))

    def test_predict_proba_custom_pred(self, classification_data):
        """predict_proba(name='my_proba') -> pred='my_proba'."""
        train_df, test_df, feature_names = classification_data
        train_expr = api.register(train_df, "proba_train")
        test_expr = api.register(test_df, "proba_test")

        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=feature_names, target="target"
        )

        preds = fitted.predict_proba(test_expr, name="my_proba")
        assert "my_proba" in preds.columns

        result = deferred_sklearn_metric(
            expr=preds,
            target="target",
            pred="my_proba",
            metric=roc_auc_score,
        ).execute()
        assert 0 <= result <= 1

    def test_decision_function_custom_pred(self):
        """decision_function(name='my_scores') -> pred='my_scores'."""
        X, y = make_classification(
            n_samples=200, n_features=5, n_informative=3, n_classes=2, random_state=7
        )
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]).assign(target=y)
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

        train_expr = api.register(train_df, "df_train")
        test_expr = api.register(test_df, "df_test")

        features = [f"f{i}" for i in range(5)]
        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", LinearSVC(random_state=7, max_iter=5000)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=features, target="target"
        )

        preds = fitted.decision_function(test_expr, name="my_scores")
        assert "my_scores" in preds.columns

        result = deferred_sklearn_metric(
            expr=preds,
            target="target",
            pred="my_scores",
            metric=roc_auc_score,
        ).execute()
        assert isinstance(result, (float, np.floating))


# ---------------------------------------------------------------------------
# Tests for metric_fn path (non-scorer metrics)
# ---------------------------------------------------------------------------

cohen_kappa_score = sklearn.metrics.cohen_kappa_score
hamming_loss = sklearn.metrics.hamming_loss
hinge_loss = sklearn.metrics.hinge_loss
zero_one_loss = sklearn.metrics.zero_one_loss
fbeta_score = sklearn.metrics.fbeta_score
mean_pinball_loss = sklearn.metrics.mean_pinball_loss
mean_tweedie_deviance = sklearn.metrics.mean_tweedie_deviance
d2_pinball_score = sklearn.metrics.d2_pinball_score
d2_tweedie_score = sklearn.metrics.d2_tweedie_score
d2_log_loss_score = sklearn.metrics.d2_log_loss_score
calinski_harabasz_score = sklearn.metrics.calinski_harabasz_score
davies_bouldin_score = sklearn.metrics.davies_bouldin_score
silhouette_score = sklearn.metrics.silhouette_score
coverage_error = sklearn.metrics.coverage_error
dcg_score = sklearn.metrics.dcg_score
ndcg_score = sklearn.metrics.ndcg_score
label_ranking_average_precision_score = (
    sklearn.metrics.label_ranking_average_precision_score
)
label_ranking_loss = sklearn.metrics.label_ranking_loss
class_likelihood_ratios = sklearn.metrics.class_likelihood_ratios
confusion_matrix = sklearn.metrics.confusion_matrix
homogeneity_completeness_v_measure = sklearn.metrics.homogeneity_completeness_v_measure
pair_confusion_matrix = sklearn.metrics.pair_confusion_matrix
roc_curve = sklearn.metrics.roc_curve
precision_recall_curve = sklearn.metrics.precision_recall_curve
det_curve = sklearn.metrics.det_curve
silhouette_samples = sklearn.metrics.silhouette_samples
KMeans = sklearn.cluster.KMeans


class TestMetricValidation:
    """Validate dispatch errors for the ``metric`` parameter."""

    def test_unknown_callable_raises(self):
        """Unknown callable -> ValueError."""

        def my_custom_fn(y_true, y_pred):
            return 0.0

        df = pd.DataFrame({"target": [0, 1], "predict": [0, 1]})
        expr = api.register(df, "unknown_callable")
        with pytest.raises(ValueError, match="Unknown callable"):
            deferred_sklearn_metric(
                expr=expr,
                target="target",
                pred="predict",
                metric=my_custom_fn,
            )

    def test_invalid_type_raises(self):
        """Non-callable, non-string -> TypeError."""
        df = pd.DataFrame({"target": [0, 1], "predict": [0, 1]})
        expr = api.register(df, "invalid_type")
        with pytest.raises(TypeError, match="metric must be"):
            deferred_sklearn_metric(
                expr=expr,
                target="target",
                pred="predict",
                metric=42,
            )


class TestMetricFnYPred:
    """Non-scorer metrics that take (y_true, y_pred) -> scalar float."""

    @pytest.fixture
    def classification_expr(self, classification_data):
        train_df, test_df, feature_names = classification_data
        train_expr = api.register(train_df, "mfn_train")
        test_expr = api.register(test_df, "mfn_test")

        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=10, random_state=42),
                ),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=feature_names, target="target"
        )

        sklearn_pipeline.fit(train_df[feature_names], train_df["target"])
        y_pred = sklearn_pipeline.predict(test_df[feature_names])

        return (
            fitted.predict(test_expr),
            test_df["target"].values,
            y_pred,
        )

    @pytest.mark.parametrize(
        "metric_func,kwargs",
        (
            (cohen_kappa_score, {}),
            (hamming_loss, {}),
            (zero_one_loss, {}),
            (fbeta_score, {"beta": 1.0}),
            (mean_pinball_loss, {}),
            (d2_pinball_score, {}),
            (mean_tweedie_deviance, {}),
            (d2_tweedie_score, {}),
        ),
        ids=lambda p: p.__name__ if callable(p) else str(p),
    )
    def test_y_pred_metric(self, classification_expr, metric_func, kwargs):
        expr_with_preds, y_true, y_pred = classification_expr

        result = deferred_sklearn_metric(
            expr=expr_with_preds,
            target="target",
            pred=ResponseMethod.PREDICT,
            metric=metric_func,
            metric_kwargs=kwargs,
        ).execute()

        expected = metric_func(y_true, y_pred, **kwargs)
        assert abs(result - expected) < 1e-10


class TestMetricFnYScore:
    """Non-scorer metrics that take (y_true, y_score/pred_decision)."""

    def test_hinge_loss(self, classification_data):
        train_df, test_df, feature_names = classification_data
        train_expr = api.register(train_df, "hinge_train")
        test_expr = api.register(test_df, "hinge_test")

        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", LinearSVC(random_state=42, max_iter=5000)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=feature_names, target="target"
        )

        expr_with_scores = fitted.decision_function(test_expr)
        result = deferred_sklearn_metric(
            expr=expr_with_scores,
            target="target",
            pred=ResponseMethod.DECISION_FUNCTION,
            metric=hinge_loss,
        ).execute()

        sklearn_pipeline.fit(train_df[feature_names], train_df["target"])
        y_score = sklearn_pipeline.decision_function(test_df[feature_names])
        expected = hinge_loss(test_df["target"], y_score)
        assert abs(result - expected) < 1e-10

    def test_d2_log_loss_score(self, classification_data):
        train_df, test_df, feature_names = classification_data
        train_expr = api.register(train_df, "d2ll_train")
        test_expr = api.register(test_df, "d2ll_test")

        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=feature_names, target="target"
        )

        expr_with_proba = fitted.predict_proba(test_expr)
        result = deferred_sklearn_metric(
            expr=expr_with_proba,
            target="target",
            pred=ResponseMethod.PREDICT_PROBA,
            metric=d2_log_loss_score,
        ).execute()

        sklearn_pipeline.fit(train_df[feature_names], train_df["target"])
        y_proba = sklearn_pipeline.predict_proba(test_df[feature_names])[:, 1]
        expected = d2_log_loss_score(test_df["target"], y_proba)
        assert abs(result - expected) < 1e-10


class TestMetricFnClustering:
    """Clustering metrics that take (X, labels) via tuple target."""

    @pytest.fixture
    def clustering_expr(self):
        from sklearn.datasets import make_blobs

        X, _ = make_blobs(n_samples=200, centers=3, n_features=4, random_state=42)
        feature_names = tuple(f"f{i}" for i in range(X.shape[1]))
        df = pd.DataFrame(X, columns=list(feature_names))
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X)
        expr = api.register(df, "cluster_test")
        return expr, feature_names, X, df["cluster"].values

    @pytest.mark.parametrize(
        "metric_func",
        (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        ),
        ids=lambda p: p.__name__,
    )
    def test_clustering_metric(self, clustering_expr, metric_func):
        expr, feature_names, X, labels = clustering_expr

        result = deferred_sklearn_metric(
            expr=expr,
            target=feature_names,
            pred="cluster",
            metric=metric_func,
        ).execute()

        expected = metric_func(X, labels)
        assert abs(result - expected) < 1e-10


class TestMetricFnSignAutoDetect:
    """When metric_fn is a known scorer func, sign is auto-detected."""

    def test_known_scorer_fn_via_metric_fn_gets_sign(self):
        """mean_squared_error is in _build_known_scorer_funcs;
        Scorer.from_spec resolves it with sign=1 (bare callable convention).
        metric_fn path picks up that same sign=1."""
        df = pd.DataFrame(
            {
                "target": [0.0, 1.0, 2.0, 3.0],
                "predict": [0.1, 0.9, 2.1, 3.2],
            }
        )
        expr = api.register(df, "sign_auto_test")

        result_metric_fn = deferred_sklearn_metric(
            expr=expr,
            target="target",
            pred="predict",
            metric=mean_squared_error,
        ).execute()

        # Bare callable through scorer path also uses sign=1
        result_scorer = deferred_sklearn_metric(
            expr=expr,
            target="target",
            pred="predict",
            metric=mean_squared_error,
        ).execute()

        assert abs(result_metric_fn - result_scorer) < 1e-10

        # neg_mean_squared_error string scorer uses sign=-1
        result_neg = deferred_sklearn_metric(
            expr=expr,
            target="target",
            pred="predict",
            metric="neg_mean_squared_error",
        ).execute()

        assert abs(result_metric_fn + result_neg) < 1e-10

    def test_non_scorer_fn_has_no_sign(self):
        df = pd.DataFrame(
            {
                "target": [0, 1, 0, 1],
                "predict": [0, 1, 1, 1],
            }
        )
        expr = api.register(df, "no_sign_test")

        result = deferred_sklearn_metric(
            expr=expr,
            target="target",
            pred="predict",
            metric=cohen_kappa_score,
        ).execute()

        expected = cohen_kappa_score([0, 1, 0, 1], [0, 1, 1, 1])
        assert abs(result - expected) < 1e-10


class TestMetricFnMultilabel:
    """Multilabel/ranking metrics that take (y_true_2D, y_score_2D).

    Both arguments are 2D matrices. target is a tuple of label column names;
    pred is a single array column whose rows are reconstructed into a
    2D ndarray via _reconstruct_matrix (no positive-class extraction).
    """

    @pytest.fixture
    def multilabel_expr(self):
        from sklearn.multiclass import OneVsRestClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        Y = rng.randint(0, 2, size=(100, 3))

        clf = OneVsRestClassifier(LogisticRegression(random_state=42))
        clf.fit(X[:70], Y[:70])

        y_true = Y[70:]
        y_score = clf.predict_proba(X[70:])

        label_names = tuple(f"label_{i}" for i in range(Y.shape[1]))
        df = pd.DataFrame(y_true, columns=list(label_names))
        # Store y_score as a single array column (mimics predict_proba storage)
        df["predict_proba"] = [row for row in y_score]

        expr = api.register(df, "multilabel_test")
        return expr, label_names, y_true, y_score

    @pytest.mark.parametrize(
        "metric_func",
        (
            coverage_error,
            dcg_score,
            ndcg_score,
            label_ranking_average_precision_score,
            label_ranking_loss,
        ),
        ids=lambda p: p.__name__,
    )
    def test_multilabel_metric(self, multilabel_expr, metric_func):
        expr, label_names, y_true, y_score = multilabel_expr

        result = deferred_sklearn_metric(
            expr=expr,
            target=label_names,
            pred="predict_proba",
            metric=metric_func,
        ).execute()

        expected = metric_func(y_true, y_score)
        assert abs(result - expected) < 1e-10

    @pytest.mark.parametrize(
        "metric_func",
        (
            coverage_error,
            dcg_score,
            ndcg_score,
            label_ranking_average_precision_score,
            label_ranking_loss,
        ),
        ids=lambda p: p.__name__,
    )
    def test_multilabel_metric_tuple_pred(self, metric_func):
        """Scores stored as separate columns; pred is a tuple of str."""
        from sklearn.multiclass import OneVsRestClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        Y = rng.randint(0, 2, size=(100, 3))

        clf = OneVsRestClassifier(LogisticRegression(random_state=42))
        clf.fit(X[:70], Y[:70])

        y_true = Y[70:]
        y_score = clf.predict_proba(X[70:])

        label_names = tuple(f"label_{i}" for i in range(Y.shape[1]))
        score_names = tuple(f"score_{i}" for i in range(y_score.shape[1]))
        df = pd.DataFrame(y_true, columns=list(label_names))
        for i, name in enumerate(score_names):
            df[name] = y_score[:, i]

        expr = api.register(df, "multilabel_tuple_pred")

        result = deferred_sklearn_metric(
            expr=expr,
            target=label_names,
            pred=score_names,
            metric=metric_func,
        ).execute()

        expected = metric_func(y_true, y_score)
        assert abs(result - expected) < 1e-10


# ---------------------------------------------------------------------------
# Tests for non-scalar return metrics (auto-detected via registry)
# ---------------------------------------------------------------------------


class TestNonScalarMetrics:
    """Metrics that return structs, arrays, or per-sample values.

    return_type is auto-resolved from _build_metric_return_types().
    """

    # -- tuple of scalars -> Struct -------------------------------------------

    def test_class_likelihood_ratios(self, classification_data):
        """class_likelihood_ratios -> Struct(positive_lr, negative_lr)."""
        train_df, test_df, feature_names = classification_data
        train_expr = api.register(train_df, "clr_train")
        test_expr = api.register(test_df, "clr_test")

        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=feature_names, target="target"
        )

        expr_with_preds = fitted.predict(test_expr)
        result = deferred_sklearn_metric(
            expr=expr_with_preds,
            target="target",
            pred=ResponseMethod.PREDICT,
            metric=class_likelihood_ratios,
        ).execute()

        sklearn_pipeline.fit(train_df[feature_names], train_df["target"])
        y_pred = sklearn_pipeline.predict(test_df[feature_names])
        expected = class_likelihood_ratios(test_df["target"], y_pred)

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "positive_likelihood_ratio",
            "negative_likelihood_ratio",
        }
        assert np.isclose(result["positive_likelihood_ratio"], expected[0])
        assert np.isclose(result["negative_likelihood_ratio"], expected[1])

    def test_homogeneity_completeness_v_measure(self):
        """homogeneity_completeness_v_measure -> Struct(h, c, v)."""
        y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        y_pred = [0, 0, 1, 1, 1, 0, 2, 2, 2]
        df = pd.DataFrame({"target": y_true, "predict": y_pred})
        expr = api.register(df, "hcv_test")

        result = deferred_sklearn_metric(
            expr=expr,
            target="target",
            pred="predict",
            metric=homogeneity_completeness_v_measure,
        ).execute()

        expected = homogeneity_completeness_v_measure(y_true, y_pred)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"homogeneity", "completeness", "v_measure"}
        assert np.isclose(result["homogeneity"], expected[0])
        assert np.isclose(result["completeness"], expected[1])
        assert np.isclose(result["v_measure"], expected[2])

    # -- matrix -> Array(Array(int64)) ----------------------------------------

    def test_confusion_matrix(self, classification_data):
        """confusion_matrix -> Array(Array(int64))."""
        train_df, test_df, feature_names = classification_data
        train_expr = api.register(train_df, "cm_train")
        test_expr = api.register(test_df, "cm_test")

        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=feature_names, target="target"
        )

        expr_with_preds = fitted.predict(test_expr)
        result = deferred_sklearn_metric(
            expr=expr_with_preds,
            target="target",
            pred=ResponseMethod.PREDICT,
            metric=confusion_matrix,
        ).execute()

        sklearn_pipeline.fit(train_df[feature_names], train_df["target"])
        y_pred = sklearn_pipeline.predict(test_df[feature_names])
        expected = confusion_matrix(test_df["target"], y_pred)

        assert isinstance(result, list)
        assert np.array_equal(np.array(result), expected)

    def test_pair_confusion_matrix(self):
        """pair_confusion_matrix -> Array(Array(int64))."""
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 2, 2, 2]
        df = pd.DataFrame({"target": y_true, "predict": y_pred})
        expr = api.register(df, "pcm_test")

        result = deferred_sklearn_metric(
            expr=expr,
            target="target",
            pred="predict",
            metric=pair_confusion_matrix,
        ).execute()

        expected = pair_confusion_matrix(y_true, y_pred)

        assert isinstance(result, list)
        assert np.array_equal(np.array(result), expected)

    # -- curves -> Struct of arrays -------------------------------------------

    def test_roc_curve(self, classification_data):
        """roc_curve -> Struct(fpr, tpr, thresholds)."""
        train_df, test_df, feature_names = classification_data
        train_expr = api.register(train_df, "roc_train")
        test_expr = api.register(test_df, "roc_test")

        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=feature_names, target="target"
        )

        expr_with_proba = fitted.predict_proba(test_expr)
        result = deferred_sklearn_metric(
            expr=expr_with_proba,
            target="target",
            pred=ResponseMethod.PREDICT_PROBA,
            metric=roc_curve,
        ).execute()

        sklearn_pipeline.fit(train_df[feature_names], train_df["target"])
        y_proba = sklearn_pipeline.predict_proba(test_df[feature_names])[:, 1]
        expected = roc_curve(test_df["target"], y_proba)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"fpr", "tpr", "thresholds"}
        assert np.allclose(result["fpr"], expected[0])
        assert np.allclose(result["tpr"], expected[1])
        assert np.allclose(result["thresholds"], expected[2])

    def test_precision_recall_curve(self, classification_data):
        """precision_recall_curve -> Struct(precision, recall, thresholds)."""
        train_df, test_df, feature_names = classification_data
        train_expr = api.register(train_df, "prc_train")
        test_expr = api.register(test_df, "prc_test")

        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=feature_names, target="target"
        )

        expr_with_proba = fitted.predict_proba(test_expr)
        result = deferred_sklearn_metric(
            expr=expr_with_proba,
            target="target",
            pred=ResponseMethod.PREDICT_PROBA,
            metric=precision_recall_curve,
        ).execute()

        sklearn_pipeline.fit(train_df[feature_names], train_df["target"])
        y_proba = sklearn_pipeline.predict_proba(test_df[feature_names])[:, 1]
        expected = precision_recall_curve(test_df["target"], y_proba)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"precision", "recall", "thresholds"}
        assert np.allclose(result["precision"], expected[0])
        assert np.allclose(result["recall"], expected[1])
        assert np.allclose(result["thresholds"], expected[2])

    def test_det_curve(self, classification_data):
        """det_curve -> Struct(fpr, fnr, thresholds)."""
        train_df, test_df, feature_names = classification_data
        train_expr = api.register(train_df, "det_train")
        test_expr = api.register(test_df, "det_test")

        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=feature_names, target="target"
        )

        expr_with_proba = fitted.predict_proba(test_expr)
        result = deferred_sklearn_metric(
            expr=expr_with_proba,
            target="target",
            pred=ResponseMethod.PREDICT_PROBA,
            metric=det_curve,
        ).execute()

        sklearn_pipeline.fit(train_df[feature_names], train_df["target"])
        y_proba = sklearn_pipeline.predict_proba(test_df[feature_names])[:, 1]
        expected = det_curve(test_df["target"], y_proba)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"fpr", "fnr", "thresholds"}
        assert np.allclose(result["fpr"], expected[0])
        assert np.allclose(result["fnr"], expected[1])
        assert np.allclose(result["thresholds"], expected[2])

    # -- per-sample -> Array(float64) -----------------------------------------

    def test_silhouette_samples(self):
        """silhouette_samples -> Array(float64)."""
        from sklearn.datasets import make_blobs

        X, _ = make_blobs(n_samples=50, centers=3, n_features=4, random_state=42)
        labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)

        feature_names = tuple(f"f{i}" for i in range(X.shape[1]))
        df = pd.DataFrame(X, columns=list(feature_names))
        df["cluster"] = labels
        expr = api.register(df, "silhouette_samples_test")

        result = deferred_sklearn_metric(
            expr=expr,
            target=feature_names,
            pred="cluster",
            metric=silhouette_samples,
        ).execute()

        expected = silhouette_samples(X, labels)

        assert isinstance(result, list)
        assert np.allclose(result, expected)


class TestDeferredAucFromCurve:
    """Tests for deferred_auc_from_curve — auc computed from deferred curves."""

    @pytest.fixture
    def proba_expr(self, classification_data):
        train_df, test_df, feature_names = classification_data
        train_expr = api.register(train_df, "auc_train")
        test_expr = api.register(test_df, "auc_test")

        sklearn_pipeline = SkPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42)),
            ]
        )
        fitted = Pipeline.from_instance(sklearn_pipeline).fit(
            train_expr, features=feature_names, target="target"
        )
        expr_with_proba = fitted.predict_proba(test_expr)

        # Also compute sklearn reference
        sklearn_pipeline.fit(train_df[feature_names], train_df["target"])
        y_proba = sklearn_pipeline.predict_proba(test_df[feature_names])[:, 1]
        y_true = test_df["target"]
        return expr_with_proba, y_true, y_proba

    def test_roc_auc(self, proba_expr):
        from sklearn.metrics import auc, roc_curve

        from xorq.expr.ml.metrics import deferred_auc_from_curve

        expr_with_proba, y_true, y_proba = proba_expr
        deferred_roc = deferred_sklearn_metric(
            expr=expr_with_proba,
            target="target",
            pred=ResponseMethod.PREDICT_PROBA,
            metric=roc_curve,
        )
        result = deferred_auc_from_curve(deferred_roc).execute()

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        expected = auc(fpr, tpr)
        assert abs(result - expected) < 1e-10

    def test_precision_recall_auc(self, proba_expr):
        from sklearn.metrics import auc, precision_recall_curve

        from xorq.expr.ml.metrics import deferred_auc_from_curve

        expr_with_proba, y_true, y_proba = proba_expr
        deferred_pr = deferred_sklearn_metric(
            expr=expr_with_proba,
            target="target",
            pred=ResponseMethod.PREDICT_PROBA,
            metric=precision_recall_curve,
        )
        result = deferred_auc_from_curve(deferred_pr).execute()

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        expected = auc(recall, precision)
        assert abs(result - expected) < 1e-10

    def test_det_auc(self, proba_expr):
        from sklearn.metrics import auc, det_curve

        from xorq.expr.ml.metrics import deferred_auc_from_curve

        expr_with_proba, y_true, y_proba = proba_expr
        deferred_det = deferred_sklearn_metric(
            expr=expr_with_proba,
            target="target",
            pred=ResponseMethod.PREDICT_PROBA,
            metric=det_curve,
        )
        result = deferred_auc_from_curve(deferred_det).execute()

        fpr, fnr, _ = det_curve(y_true, y_proba)
        expected = auc(fpr, fnr)
        assert abs(result - expected) < 1e-10

    def test_compose_into_table(self, proba_expr):
        """auc composes with other metrics via .as_scalar() + .mutate()."""
        from sklearn.metrics import roc_curve

        from xorq.expr.ml.metrics import deferred_auc_from_curve

        expr_with_proba, _, _ = proba_expr
        deferred_roc = deferred_sklearn_metric(
            expr=expr_with_proba,
            target="target",
            pred=ResponseMethod.PREDICT_PROBA,
            metric=roc_curve,
        )
        deferred_roc_auc = deferred_auc_from_curve(deferred_roc)
        deferred_roc_auc_score = deferred_sklearn_metric(
            expr=expr_with_proba,
            target="target",
            pred=ResponseMethod.PREDICT_PROBA,
            metric=roc_auc_score,
        )

        table = (
            deferred_roc_auc_score.as_scalar()
            .name("roc_auc_score")
            .as_table()
            .mutate(roc_auc=deferred_roc_auc.as_scalar())
        )
        result = table.execute()
        assert "roc_auc_score" in result.columns
        assert "roc_auc" in result.columns
        assert len(result) == 1

    def test_invalid_input_non_struct(self):
        """Raises TypeError for non-Struct expressions."""
        from xorq.expr.ml.metrics import deferred_auc_from_curve

        df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        expr = api.register(df, "auc_invalid")
        deferred_acc = deferred_sklearn_metric(
            expr=expr,
            target="target",
            pred="a",
            metric=accuracy_score,
        )
        with pytest.raises(TypeError, match="Expected a Struct"):
            deferred_auc_from_curve(deferred_acc)
