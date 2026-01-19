"""Tests for deferred sklearn metrics evaluation using Pipeline API."""

import numpy as np
import pandas as pd
import pytest

import xorq.expr.datatypes as dt
from xorq.expr import api
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline


# Skip all tests in this module if sklearn is not installed
sklearn = pytest.importorskip("sklearn")

# Access sklearn modules through the sklearn object to avoid E402 linting errors
make_classification = sklearn.datasets.make_classification
make_regression = sklearn.datasets.make_regression
RandomForestClassifier = sklearn.ensemble.RandomForestClassifier
RandomForestRegressor = sklearn.ensemble.RandomForestRegressor
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
        pred_col="predicted",
        metric_fn=accuracy_score,
    )
    actual_accuracy = accuracy_expr.execute()
    assert np.isclose(actual_accuracy, expected_accuracy)

    # Test precision
    precision_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred_col="predicted",
        metric_fn=precision_score,
    )
    actual_precision = precision_expr.execute()
    assert np.isclose(actual_precision, expected_precision)

    # Test recall
    recall_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred_col="predicted",
        metric_fn=recall_score,
    )
    actual_recall = recall_expr.execute()
    assert np.isclose(actual_recall, expected_recall)

    # Test F1 score
    f1_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred_col="predicted",
        metric_fn=f1_score,
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
        pred_col="predicted",
        metric_fn=mean_squared_error,
    )
    actual_mse = mse_expr.execute()
    assert np.isclose(actual_mse, expected_mse, rtol=1e-4)

    # Test MAE using Pipeline API
    mae_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred_col="predicted",
        metric_fn=mean_absolute_error,
    )
    actual_mae = mae_expr.execute()
    assert np.isclose(actual_mae, expected_mae, rtol=1e-4)

    # Test R2 using Pipeline API
    r2_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred_col="predicted",
        metric_fn=r2_score,
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
        pred_col="predicted",
        metric_fn=accuracy_score,
    )
    actual_accuracy = accuracy_expr.execute()
    assert np.isclose(actual_accuracy, expected_accuracy)

    # Test precision with macro averaging
    precision_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred_col="predicted",
        metric_fn=precision_score,
        metric_kwargs={"average": "macro", "zero_division": 0},
    )
    actual_precision = precision_expr.execute()
    assert np.isclose(actual_precision, expected_precision_macro)

    # Test recall with weighted averaging
    recall_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred_col="predicted",
        metric_fn=recall_score,
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

    # Define a custom metric
    def custom_metric(y_true, y_pred):
        """A custom metric that returns accuracy * 100."""
        return accuracy_score(y_true, y_pred) * 100

    # Get predictions
    expr_with_preds = fitted_pipeline.predict(test_expr)

    # Test custom metric
    custom_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred_col="predicted",
        metric_fn=custom_metric,
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
        pred_col="predicted",
        metric_fn=accuracy_score,
    )

    # The expression should have an execute method (deferred)
    assert hasattr(accuracy_expr, "execute")

    # It should not be a computed value yet
    assert not isinstance(accuracy_expr, (float, int, np.number))

    # Can create multiple metrics without execution
    precision_expr = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred_col="predicted",
        metric_fn=precision_score,
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
        pred_col="predicted_proba",
        metric_fn=roc_auc_score,
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
    proba_result = proba_expr["predicted_proba"].execute()
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
        pred_col="decision_function",
        metric_fn=roc_auc_score,
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
        pred_col="predicted_proba",
        metric_fn=roc_auc_score,
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
        pred_col="predicted",
        metric_fn=precision_score,
        metric_kwargs={"average": "macro", "zero_division": 0},
    ).execute()

    precision_weighted = deferred_sklearn_metric(
        expr=expr_with_preds,
        target="target",
        pred_col="predicted",
        metric_fn=precision_score,
        metric_kwargs={"average": "weighted", "zero_division": 0},
    ).execute()

    # Different averaging should give different results
    assert precision_macro != precision_weighted

    # Both should be valid precision scores
    assert 0 <= precision_macro <= 1
    assert 0 <= precision_weighted <= 1
