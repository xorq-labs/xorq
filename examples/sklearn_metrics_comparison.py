#!/usr/bin/env python
"""
Comparison of scikit-learn metrics using xorq's minimal deferred metrics API.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import xorq.api as xo
from xorq.expr.ml import Pipeline
from xorq.expr.ml.metrics import deferred_sklearn_metric


target = "target"
feature_prefix = "feature_"

metric_configs = (
    ("accuracy", accuracy_score, {}),
    ("precision_macro", precision_score, {"average": "macro", "zero_division": 0}),
    (
        "precision_weighted",
        precision_score,
        {"average": "weighted", "zero_division": 0},
    ),
    ("recall_macro", recall_score, {"average": "macro", "zero_division": 0}),
    ("recall_weighted", recall_score, {"average": "weighted", "zero_division": 0}),
    ("f1_macro", f1_score, {"average": "macro", "zero_division": 0}),
    ("f1_weighted", f1_score, {"average": "weighted", "zero_division": 0}),
)


def make_exprs(X_train, y_train, X_test, y_test):
    con = xo.connect()
    (train, test) = (
        con.register(
            pd.DataFrame(X)
            .rename(columns=(feature_prefix + "{}").format)
            .assign(**{target: y}),
            name,
        )
        for (X, y, name) in (
            (X_train, y_train, "train"),
            (X_test, y_test, "test"),
        )
    )
    features = train.select(xo.selectors.startswith(feature_prefix)).columns
    return (train, test, features)


def compute_metrics(clf, X_train, X_test, y_train, y_test):
    # Sklearn
    sklearn_pipeline = make_pipeline(clf).fit(X_train, y_train)
    y_pred = sklearn_pipeline.predict(X_test)
    sklearn_metrics = {
        name: metric_fn(y_test, y_pred, **kwargs)
        for name, metric_fn, kwargs in metric_configs
    }
    # xorq
    (train, test, features) = make_exprs(X_train, y_train, X_test, y_test)
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline).fit(
        train, features=features, target=target
    )
    expr_with_preds = xorq_pipeline.predict(test, name="predicted")
    xorq_metrics = {
        name: deferred_sklearn_metric(
            expr=expr_with_preds,
            target=target,
            pred_col="predicted",
            scorer=metric_fn,
            metric_kwargs=kwargs if kwargs else (),
        ).execute()
        for name, metric_fn, kwargs in metric_configs
    }
    return {
        "sklearn_metrics": sklearn_metrics,
        "xorq_metrics": xorq_metrics,
    }


def gen_datasets():
    yield from (
        (
            f"{n_samples}x{n_features}x{n_classes}",
            train_test_split(
                *X_y,
                test_size=0.2,
                random_state=42,
                stratify=X_y[1],
            ),
        )
        for (n_samples, n_features, n_classes) in (
            (5000, 10, 3),
            (3000, 5, 2),
        )
        for X_y in (
            make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=min(n_features - 2, int(n_features * 0.7)),
                n_redundant=min(
                    n_features - min(n_features - 2, int(n_features * 0.7)) - 1,
                    int(n_features * 0.2),
                ),
                n_classes=n_classes,
                n_clusters_per_class=2,
                random_state=42,
                flip_y=0.1,
            ),
        )
    )


names_classifiers = (
    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
    (
        "RandomForest",
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    ),
)


def make_comparison_df():
    df = pd.DataFrame.from_dict(
        {
            (ds_name, clf_name): compute_metrics(clf, X_train, X_test, y_train, y_test)
            for (ds_name, (X_train, X_test, y_train, y_test)) in gen_datasets()
            for (clf_name, clf) in names_classifiers
        },
        orient="index",
    )
    return df


def validate_metrics_match(df, tolerance=0.0001):
    mismatches = []
    for idx, row in df.iterrows():
        for metric_name in row["sklearn_metrics"]:
            if metric_name in row["xorq_metrics"]:
                diff = abs(
                    row["sklearn_metrics"][metric_name]
                    - row["xorq_metrics"][metric_name]
                )
                if diff > tolerance:
                    mismatches.append(
                        {
                            "dataset": idx[0],
                            "model": idx[1],
                            "metric": metric_name,
                            "sklearn": row["sklearn_metrics"][metric_name],
                            "xorq": row["xorq_metrics"][metric_name],
                            "diff": diff,
                        }
                    )
    return mismatches


def test_predict_proba():
    """Test predict_proba() for probability-based metrics like ROC-AUC."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
        flip_y=0.1,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Sklearn
    sklearn_pipeline = make_pipeline(
        LogisticRegression(max_iter=1000, random_state=42)
    ).fit(X_train, y_train)
    sklearn_auc = roc_auc_score(y_test, sklearn_pipeline.predict_proba(X_test)[:, 1])

    # xorq
    (train, test, features) = make_exprs(X_train, y_train, X_test, y_test)
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline).fit(
        train, features=features, target=target
    )
    expr_with_proba = xorq_pipeline.predict_proba(test, name="predicted_proba")
    xorq_auc = deferred_sklearn_metric(
        expr=expr_with_proba,
        target=target,
        pred_col="predicted_proba",
        scorer=roc_auc_score,
    ).execute()

    return sklearn_auc, xorq_auc


def test_decision_function():
    """Test decision_function() for models like LinearSVC."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=123,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Sklearn
    sklearn_pipeline = make_pipeline(LinearSVC(random_state=123, max_iter=5000)).fit(
        X_train, y_train
    )
    sklearn_auc = roc_auc_score(y_test, sklearn_pipeline.decision_function(X_test))

    # xorq
    (train, test, features) = make_exprs(X_train, y_train, X_test, y_test)
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline).fit(
        train, features=features, target=target
    )
    expr_with_scores = xorq_pipeline.decision_function(test)
    xorq_auc = deferred_sklearn_metric(
        expr=expr_with_scores,
        target=target,
        pred_col="decision_function",
        scorer=roc_auc_score,
    ).execute()

    return sklearn_auc, xorq_auc


def test_feature_importances():
    """Test feature_importances for tree-based models."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        flip_y=0.05,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Sklearn
    sklearn_pipeline = make_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=42)
    ).fit(X_train, y_train)
    sklearn_importances = sklearn_pipeline.named_steps[
        "randomforestclassifier"
    ].feature_importances_

    # xorq
    (train, test, features) = make_exprs(X_train, y_train, X_test, y_test)
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline).fit(
        train, features=features, target=target
    )
    importances_expr = xorq_pipeline.feature_importances(test)
    xorq_importances = np.array(
        importances_expr.execute()["feature_importances"].iloc[0]
    )

    return sklearn_importances, xorq_importances


if __name__ == "__pytest_main__":
    df = make_comparison_df()
    mismatches = validate_metrics_match(df)
    assert not mismatches, f"Metrics mismatch: {mismatches}"

    sklearn_auc_proba, xorq_auc_proba = test_predict_proba()
    assert np.isclose(sklearn_auc_proba, xorq_auc_proba), "predict_proba AUC mismatch"

    sklearn_auc_decision, xorq_auc_decision = test_decision_function()
    assert np.isclose(sklearn_auc_decision, xorq_auc_decision), (
        "decision_function AUC mismatch"
    )

    sklearn_importances, xorq_importances = test_feature_importances()
    assert np.allclose(sklearn_importances, xorq_importances), (
        "feature_importances mismatch"
    )

    pytest_examples_passed = True
