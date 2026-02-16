"""Deferred AUC from Curve

Demonstrates deferred_auc_from_curve — computing AUC from deferred curve
metrics (roc_curve, precision_recall_curve, det_curve) without eager execution.

The user builds a deferred curve expression with deferred_sklearn_metric,
then passes it to deferred_auc_from_curve which applies sklearn.metrics.auc
on the curve's x/y fields via a pyarrow scalar UDF.

All results compose into a single table via .agg().
"""

import pandas as pd
import toolz
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    det_curve,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.expr.ml.metrics import deferred_auc_from_curve, deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline


con = xo.connect()
target_name, pred_name = "target", "scores"

# --- Data setup ---
n_features = 10
feature_names = [f"f{i}" for i in range(n_features)]
X, y = make_classification(
    n_samples=500, n_features=n_features, n_informative=5, random_state=42
)
train_df, test_df = train_test_split(
    pd.DataFrame(X, columns=feature_names).assign(target=y),
    test_size=0.3,
    random_state=42,
)
train_expr = con.register(train_df, "train")
test_expr = con.register(test_df, "test")

# --- Fit and predict_proba ---
fitted = Pipeline.from_instance(
    SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
).fit(train_expr, features=tuple(feature_names), target=target_name)
proba_expr = fitted.predict_proba(test_expr, name=pred_name)


# --- Deferred curve metrics ---
# deferred_sklearn_metric is curried, so partial application returns a callable
make_roc_auc, make_pr_auc, make_det_auc = (
    toolz.compose(
        deferred_auc_from_curve,
        deferred_sklearn_metric(target=target_name, pred=pred_name, metric=metric),
    )
    for metric in (roc_curve, precision_recall_curve, det_curve)
)
# --- For comparison: roc_auc_score via the scorer path ---
make_roc_auc_score = deferred_sklearn_metric(
    target=target_name,
    pred=pred_name,
    metric=roc_auc_score,
)
# --- Compose into a single table ---
auc_metrics = proba_expr.agg(
    roc_auc=make_roc_auc,
    pr_auc=make_pr_auc,
    det_auc=make_det_auc,
    roc_auc_score=make_roc_auc_score,
)


if __name__ == "__pytest_main__":
    result = auc_metrics.execute().iloc[0]
    print("AUC Metrics:")
    print(f"  ROC AUC (from curve):  {result['roc_auc']:.4f}")
    print(f"  ROC AUC (from scorer): {result['roc_auc_score']:.4f}")
    print(f"  PR AUC:                {result['pr_auc']:.4f}")
    print(f"  DET AUC:               {result['det_auc']:.4f}")

    pytest_examples_passed = True
