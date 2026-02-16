"""Sklearn Deferred Metrics — Advanced

Demonstrates every (target, pred) dispatch combination supported by
deferred_sklearn_metric, plus non-scalar return types and how to unpack them.

Dispatch combinations
---------------------
1. (str, str)           — standard classification / regression
2. (tuple, str)         — clustering (features as target, labels as pred)
3. (tuple, str-array)   — multilabel with array-valued prediction column
4. (tuple, tuple)       — multilabel with separate score columns

Non-scalar return types
-----------------------
- Array(Array(int64))   — confusion_matrix
- Struct(fields...)     — class_likelihood_ratios, homogeneity_completeness_v_measure
- Array(float64)        — silhouette_samples

All return types compose via .agg() into a single table.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    class_likelihood_ratios,
    confusion_matrix,
    coverage_error,
    homogeneity_completeness_v_measure,
    label_ranking_loss,
    ndcg_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline


con = xo.connect()

# ============================================================================
# Section 1 — (str, str): classifier with non-scalar return types
# ============================================================================

feature_names = [f"f{i}" for i in range(10)]
X_cls, y_cls = make_classification(
    n_samples=500, n_features=10, n_informative=5, random_state=42
)
train_cls, test_cls = train_test_split(
    pd.DataFrame(X_cls, columns=feature_names).assign(target=y_cls),
    test_size=0.3,
    random_state=42,
)
train_cls_expr = con.register(train_cls, "train_cls")
test_cls_expr = con.register(test_cls, "test_cls")

fitted_clf = Pipeline.from_instance(
    SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )
).fit(train_cls_expr, features=tuple(feature_names), target="target")

clf_preds = fitted_clf.predict(test_cls_expr, name="pred")

# deferred_sklearn_metric is curried — partial application returns a callable
make_clf_metric = deferred_sklearn_metric(target="target", pred="pred")

clf_metrics = clf_preds.agg(
    # Scalar
    accuracy=make_clf_metric(metric=accuracy_score),
    # Non-scalar: 2D int64 matrix
    confusion_matrix=make_clf_metric(metric=confusion_matrix),
    # Non-scalar: Struct with two float64 fields
    class_likelihood_ratios=make_clf_metric(metric=class_likelihood_ratios),
)

# ============================================================================
# Section 2 — (tuple, str): clustering metrics (features as target)
# ============================================================================

X_cluster, _ = make_blobs(n_samples=200, centers=3, n_features=4, random_state=42)
cluster_features = tuple(f"f{i}" for i in range(X_cluster.shape[1]))
cluster_df = pd.DataFrame(X_cluster, columns=list(cluster_features))
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_df["cluster"] = kmeans.fit_predict(X_cluster)
cluster_expr = con.register(cluster_df, "clusters")

make_cluster_metric = deferred_sklearn_metric(
    target=cluster_features,
    pred="cluster",
)

# Note: h/c/v needs ground-truth labels, not features — use (str, str) here
cluster_df["true_label"] = np.repeat([0, 1, 2], [67, 67, 66])
cluster_labeled_expr = con.register(cluster_df, "clusters_labeled")

cluster_metrics = cluster_expr.agg(
    # target=tuple of feature columns, pred=single label column
    silhouette=make_cluster_metric(metric=silhouette_score),
    calinski_harabasz=make_cluster_metric(metric=calinski_harabasz_score),
    # Non-scalar: per-sample float64 array
    silhouette_samples=make_cluster_metric(metric=silhouette_samples),
)

# HCV uses (str, str) with ground-truth labels
hcv_metric = cluster_labeled_expr.agg(
    hcv=deferred_sklearn_metric(
        target="true_label",
        pred="cluster",
        metric=homogeneity_completeness_v_measure,
    ),
)

# ============================================================================
# Section 3 — (tuple, str): multilabel with array-valued prediction column
# ============================================================================

rng = np.random.RandomState(42)
X_ml = rng.randn(100, 5)
Y_ml = rng.randint(0, 2, size=(100, 3))

ovr = OneVsRestClassifier(LogisticRegression(random_state=42))
ovr.fit(X_ml[:70], Y_ml[:70])

y_true_ml = Y_ml[70:]
y_score_ml = ovr.predict_proba(X_ml[70:])

label_names = tuple(f"label_{i}" for i in range(Y_ml.shape[1]))
ml_df = pd.DataFrame(y_true_ml, columns=list(label_names))
# Store scores as a single array-valued column (mimics predict_proba storage)
ml_df["scores"] = [row for row in y_score_ml]
ml_expr = con.register(ml_df, "multilabel_array")

make_ml_metric = deferred_sklearn_metric(target=label_names, pred="scores")

ml_array_metrics = ml_expr.agg(
    # target=tuple of label columns, pred=single array column
    coverage_error=make_ml_metric(metric=coverage_error),
    ndcg=make_ml_metric(metric=ndcg_score),
    label_ranking_loss=make_ml_metric(metric=label_ranking_loss),
)

# ============================================================================
# Section 4 — (tuple, tuple): multilabel with separate score columns
# ============================================================================

score_names = tuple(f"score_{i}" for i in range(y_score_ml.shape[1]))
ml_cols_df = pd.DataFrame(y_true_ml, columns=list(label_names))
for i, name in enumerate(score_names):
    ml_cols_df[name] = y_score_ml[:, i]
ml_cols_expr = con.register(ml_cols_df, "multilabel_cols")

make_ml_cols_metric = deferred_sklearn_metric(target=label_names, pred=score_names)

ml_cols_metrics = ml_cols_expr.agg(
    # target=tuple of label columns, pred=tuple of score columns
    coverage_error=make_ml_cols_metric(metric=coverage_error),
    ndcg=make_ml_cols_metric(metric=ndcg_score),
    label_ranking_loss=make_ml_cols_metric(metric=label_ranking_loss),
)


if __name__ == "__pytest_main__":
    # --- Section 1: (str, str) — classifier non-scalar metrics ---
    print("=== Section 1: (str, str) — Classifier non-scalar metrics ===")
    clf_result = clf_metrics.execute().iloc[0]
    print(f"  accuracy: {clf_result['accuracy']}")

    # Unpack confusion matrix: nested list -> 2D ndarray
    cm = np.array(clf_result["confusion_matrix"])
    print(f"  confusion_matrix:\n{cm}")

    # Unpack Struct: dict with named fields
    clr = clf_result["class_likelihood_ratios"]
    print(f"  class_likelihood_ratios: {clr}")

    # --- Section 2: (tuple, str) — clustering metrics ---
    print("\n=== Section 2: (tuple, str) — Clustering metrics ===")
    clu_result = cluster_metrics.execute().iloc[0]
    print(f"  silhouette:        {clu_result['silhouette']:.4f}")
    print(f"  calinski_harabasz: {clu_result['calinski_harabasz']:.1f}")

    # Unpack per-sample array
    sil = np.array(clu_result["silhouette_samples"])
    print(f"  silhouette_samples: shape={sil.shape}, mean={sil.mean():.4f}")

    # Unpack Struct
    hcv_result = hcv_metric.execute().iloc[0]
    print(f"  hcv: {hcv_result['hcv']}")

    # --- Section 3: (tuple, str) — multilabel array pred ---
    print("\n=== Section 3: (tuple, str) — Multilabel (array pred) ===")
    print(ml_array_metrics.execute().to_string(index=False))

    # --- Section 4: (tuple, tuple) — multilabel separate columns ---
    print("\n=== Section 4: (tuple, tuple) — Multilabel (tuple pred) ===")
    print(ml_cols_metrics.execute().to_string(index=False))

    pytest_examples_passed = True
