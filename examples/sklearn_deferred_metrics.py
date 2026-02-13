"""Sklearn Deferred Metrics

Demonstrates deferred_sklearn_metric — the low-level API for computing
sklearn metrics on xorq expressions. Shows classifier, regressor, and
clusterer pipelines each scored with three metrics per pipeline:

1. Two scalar float64 metrics composed into a single deferred table.
2. A third metric composed via .agg() showcasing non-float64 or non-scorer
   capabilities — all three land in a single table per pipeline:
   - Classifier: confusion_matrix -> Array(Array(int64))
   - Regressor: d2_tweedie_score -> float64 via the non-scorer callable path
   - Clusterer: homogeneity_completeness_v_measure -> Struct(h, c, v)

All metrics on the same predictions compose naturally: predict once,
build multiple metric expressions, then combine them into one table with
a single execute() call.

deferred_sklearn_metric accepts scorer name strings, bare callables
(known sklearn metric functions), or make_scorer objects. Sign is
auto-detected from the scorer (e.g. "neg_mean_squared_error" -> sign=-1).
Non-scalar return types (Struct, Array) are auto-detected from the registry.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    d2_tweedie_score,
    f1_score,
    homogeneity_completeness_v_measure,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline


# --- Shared data ---

con = xo.connect()
target_name, pred_name = "target", "my_predicted"
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

X_reg, y_reg = make_regression(
    n_samples=500, n_features=10, n_informative=5, noise=10.0, random_state=42
)
train_reg, test_reg = train_test_split(
    pd.DataFrame(X_reg, columns=feature_names).assign(target=y_reg),
    test_size=0.3,
    random_state=42,
)
train_reg_expr = con.register(train_reg, "train_reg")
test_reg_expr = con.register(test_reg, "test_reg")

# --- Helper: deferred_sklearn_metric is curried, so partial application works ---
make_metric = deferred_sklearn_metric(target=target_name, pred=pred_name)

# --- Classifier: predict once, score thrice ---

fitted_clf = Pipeline.from_instance(
    SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )
).fit(train_cls_expr, features=tuple(feature_names), target=target_name)

clf_preds = fitted_clf.predict(test_cls_expr, name=pred_name)

clf_metrics = clf_preds.agg(
    accuracy=make_metric(metric=accuracy_score),
    f1=make_metric(metric=f1_score),
    # Non-scalar metric: confusion_matrix -> Array(Array(int64))
    confusion_matrix=make_metric(metric=confusion_matrix),
)

# --- Regressor: predict once, score thrice ---

fitted_reg = Pipeline.from_instance(
    SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
    )
).fit(train_reg_expr, features=tuple(feature_names), target=target_name)

reg_preds = fitted_reg.predict(test_reg_expr, name=pred_name)

reg_metrics = reg_preds.agg(
    r2=make_metric(metric=r2_score),
    mse=make_metric(metric=mean_squared_error),
    # Non-scorer metric: d2_tweedie_score dispatched via callable path
    d2_tweedie=make_metric(metric=d2_tweedie_score),
)

# --- Clusterer: predict once, score thrice ---

# Reuse classification data — target serves as ground-truth labels
fitted_clu = Pipeline.from_instance(
    SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("clu", KMeans(n_clusters=2, random_state=42, n_init=10)),
        ]
    )
).fit(train_cls_expr, features=tuple(feature_names), target=target_name)

clu_preds = fitted_clu.predict(test_cls_expr, name=pred_name)

clu_metrics = clu_preds.agg(
    adj_rand=make_metric(metric=adjusted_rand_score),
    # String scorer — sign auto-detected (neg_* -> sign=-1)
    neg_mse=make_metric(metric="neg_mean_squared_error"),
    # Non-scalar metric: Struct(homogeneity, completeness, v_measure)
    hcv=make_metric(metric=homogeneity_completeness_v_measure),
)


if __name__ == "__pytest_main__":
    # One execute per pipeline — all metrics (scalar and non-scalar) in a single table
    print("=== Classifier (RandomForestClassifier) ===")
    print(clf_metrics.execute().to_string(index=False))

    print("\n=== Regressor (RandomForestRegressor) ===")
    print(reg_metrics.execute().to_string(index=False))

    print("\n=== Clusterer (KMeans) ===")
    print(clu_metrics.execute().to_string(index=False))

    pytest_examples_passed = True
