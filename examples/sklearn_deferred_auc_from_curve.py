"""Deferred AUC from Curve

Demonstrates deferred_auc_from_curve — computing AUC from deferred curve
metrics (roc_curve, precision_recall_curve, det_curve) without eager execution.

The user builds a deferred curve expression with deferred_sklearn_metric,
then passes it to deferred_auc_from_curve which applies sklearn.metrics.auc
on the curve's x/y fields via a pyarrow scalar UDF.

All results compose into a single table via .as_scalar() + .mutate().
"""

import pandas as pd
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

# --- Data setup ---
feature_names = [f"f{i}" for i in range(10)]
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=5, random_state=42
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
).fit(train_expr, features=tuple(feature_names), target="target")

proba_expr = fitted.predict_proba(test_expr, name="scores")

# --- Deferred curve metrics ---
deferred_roc = deferred_sklearn_metric(
    expr=proba_expr,
    target="target",
    pred="scores",
    metric=roc_curve,
)
deferred_pr = deferred_sklearn_metric(
    expr=proba_expr,
    target="target",
    pred="scores",
    metric=precision_recall_curve,
)
deferred_det = deferred_sklearn_metric(
    expr=proba_expr,
    target="target",
    pred="scores",
    metric=det_curve,
)

# --- AUC from each curve ---
deferred_roc_auc = deferred_auc_from_curve(deferred_roc)
deferred_pr_auc = deferred_auc_from_curve(deferred_pr)
deferred_det_auc = deferred_auc_from_curve(deferred_det)

# --- For comparison: roc_auc_score via the scorer path ---
deferred_roc_auc_score = deferred_sklearn_metric(
    expr=proba_expr,
    target="target",
    pred="scores",
    metric=roc_auc_score,
)

# --- Compose into a single table ---
auc_metrics = (
    deferred_roc_auc.as_scalar()
    .name("roc_auc")
    .as_table()
    .mutate(pr_auc=deferred_pr_auc.as_scalar())
    .mutate(det_auc=deferred_det_auc.as_scalar())
    .mutate(roc_auc_score=deferred_roc_auc_score.as_scalar())
)


if __name__ == "__pytest_main__":
    result = auc_metrics.execute()
    print("AUC Metrics:")
    print(f"  ROC AUC (from curve):  {result['roc_auc'].iloc[0]:.4f}")
    print(f"  ROC AUC (from scorer): {result['roc_auc_score'].iloc[0]:.4f}")
    print(f"  PR AUC:                {result['pr_auc'].iloc[0]:.4f}")
    print(f"  DET AUC:               {result['det_auc'].iloc[0]:.4f}")

    pytest_examples_passed = True
