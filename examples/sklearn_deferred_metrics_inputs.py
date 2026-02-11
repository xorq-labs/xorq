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

All return types compose via .as_scalar() + .mutate() into a single table.
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

# Scalar
deferred_accuracy = deferred_sklearn_metric(
    expr=clf_preds,
    target="target",
    pred="pred",
    metric=accuracy_score,
)
# Non-scalar: 2D int64 matrix
deferred_cm = deferred_sklearn_metric(
    expr=clf_preds,
    target="target",
    pred="pred",
    metric=confusion_matrix,
)
# Non-scalar: Struct with two float64 fields
deferred_clr = deferred_sklearn_metric(
    expr=clf_preds,
    target="target",
    pred="pred",
    metric=class_likelihood_ratios,
)

clf_metrics = (
    deferred_accuracy.as_scalar()
    .name("accuracy")
    .as_table()
    .mutate(confusion_matrix=deferred_cm.as_scalar())
    .mutate(class_likelihood_ratios=deferred_clr.as_scalar())
)

# ============================================================================
# Section 2 — (tuple, str): clustering metrics (features as target)
# ============================================================================

X_blobs, _ = make_blobs(n_samples=200, centers=3, n_features=4, random_state=42)
blob_features = tuple(f"f{i}" for i in range(X_blobs.shape[1]))
blob_df = pd.DataFrame(X_blobs, columns=list(blob_features))
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
blob_df["cluster"] = kmeans.fit_predict(X_blobs)
blob_expr = con.register(blob_df, "blobs")

# target=tuple of feature columns, pred=single label column
deferred_silhouette = deferred_sklearn_metric(
    expr=blob_expr,
    target=blob_features,
    pred="cluster",
    metric=silhouette_score,
)
deferred_calinski = deferred_sklearn_metric(
    expr=blob_expr,
    target=blob_features,
    pred="cluster",
    metric=calinski_harabasz_score,
)
# Non-scalar: per-sample float64 array
deferred_sil_samples = deferred_sklearn_metric(
    expr=blob_expr,
    target=blob_features,
    pred="cluster",
    metric=silhouette_samples,
)
# Non-scalar: Struct(homogeneity, completeness, v_measure)
# Note: h/c/v needs ground-truth labels, not features — use (str, str) here
blob_df_with_labels = blob_df.copy()
blob_df_with_labels["true_label"] = np.repeat([0, 1, 2], [67, 67, 66])
blob_labeled_expr = con.register(blob_df_with_labels, "blobs_labeled")
deferred_hcv = deferred_sklearn_metric(
    expr=blob_labeled_expr,
    target="true_label",
    pred="cluster",
    metric=homogeneity_completeness_v_measure,
)

cluster_metrics = (
    deferred_silhouette.as_scalar()
    .name("silhouette")
    .as_table()
    .mutate(calinski_harabasz=deferred_calinski.as_scalar())
    .mutate(silhouette_samples=deferred_sil_samples.as_scalar())
    .mutate(hcv=deferred_hcv.as_scalar())
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

# target=tuple of label columns, pred=single array column
deferred_coverage = deferred_sklearn_metric(
    expr=ml_expr,
    target=label_names,
    pred="scores",
    metric=coverage_error,
)
deferred_ndcg = deferred_sklearn_metric(
    expr=ml_expr,
    target=label_names,
    pred="scores",
    metric=ndcg_score,
)
deferred_lr_loss = deferred_sklearn_metric(
    expr=ml_expr,
    target=label_names,
    pred="scores",
    metric=label_ranking_loss,
)

ml_array_metrics = (
    deferred_coverage.as_scalar()
    .name("coverage_error")
    .as_table()
    .mutate(ndcg=deferred_ndcg.as_scalar())
    .mutate(label_ranking_loss=deferred_lr_loss.as_scalar())
)

# ============================================================================
# Section 4 — (tuple, tuple): multilabel with separate score columns
# ============================================================================

score_names = tuple(f"score_{i}" for i in range(y_score_ml.shape[1]))
ml_cols_df = pd.DataFrame(y_true_ml, columns=list(label_names))
for i, name in enumerate(score_names):
    ml_cols_df[name] = y_score_ml[:, i]
ml_cols_expr = con.register(ml_cols_df, "multilabel_cols")

# target=tuple of label columns, pred=tuple of score columns
deferred_coverage_t = deferred_sklearn_metric(
    expr=ml_cols_expr,
    target=label_names,
    pred=score_names,
    metric=coverage_error,
)
deferred_ndcg_t = deferred_sklearn_metric(
    expr=ml_cols_expr,
    target=label_names,
    pred=score_names,
    metric=ndcg_score,
)
deferred_lr_loss_t = deferred_sklearn_metric(
    expr=ml_cols_expr,
    target=label_names,
    pred=score_names,
    metric=label_ranking_loss,
)

ml_cols_metrics = (
    deferred_coverage_t.as_scalar()
    .name("coverage_error")
    .as_table()
    .mutate(ndcg=deferred_ndcg_t.as_scalar())
    .mutate(label_ranking_loss=deferred_lr_loss_t.as_scalar())
)


if __name__ == "__pytest_main__":
    # --- Section 1: (str, str) — classifier non-scalar metrics ---
    print("=== Section 1: (str, str) — Classifier non-scalar metrics ===")
    clf_result = clf_metrics.execute()
    print(clf_result[["accuracy"]].to_string(index=False))

    # Unpack confusion matrix: nested list -> 2D ndarray
    cm = np.array(clf_result["confusion_matrix"].iloc[0])
    print(f"  confusion_matrix:\n{cm}")

    # Unpack Struct: dict with named fields
    clr = clf_result["class_likelihood_ratios"].iloc[0]
    print(f"  class_likelihood_ratios: {clr}")

    # --- Section 2: (tuple, str) — clustering metrics ---
    print("\n=== Section 2: (tuple, str) — Clustering metrics ===")
    clu_result = cluster_metrics.execute()
    print(f"  silhouette:        {clu_result['silhouette'].iloc[0]:.4f}")
    print(f"  calinski_harabasz: {clu_result['calinski_harabasz'].iloc[0]:.1f}")

    # Unpack per-sample array
    sil = np.array(clu_result["silhouette_samples"].iloc[0])
    print(f"  silhouette_samples: shape={sil.shape}, mean={sil.mean():.4f}")

    # Unpack Struct
    hcv = clu_result["hcv"].iloc[0]
    print(f"  hcv: {hcv}")

    # --- Section 3: (tuple, str) — multilabel array pred ---
    print("\n=== Section 3: (tuple, str) — Multilabel (array pred) ===")
    print(ml_array_metrics.execute().to_string(index=False))

    # --- Section 4: (tuple, tuple) — multilabel separate columns ---
    print("\n=== Section 4: (tuple, tuple) — Multilabel (tuple pred) ===")
    print(ml_cols_metrics.execute().to_string(index=False))

    pytest_examples_passed = True
