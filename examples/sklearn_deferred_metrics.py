"""Sklearn Deferred Metrics

Demonstrates deferred_sklearn_metric — the low-level API for computing
sklearn metrics on xorq expressions. Shows classifier, regressor, and
clusterer pipelines each scored with three metrics per pipeline:

1. Two scalar float64 metrics composed into a single deferred table.
2. A third metric composed via .mutate() showcasing non-float64 or non-scorer
   capabilities — all three land in a single table per pipeline:
   - Classifier: confusion_matrix → Array(Array(int64))
   - Regressor: d2_tweedie_score → float64 via the non-scorer callable path
   - Clusterer: homogeneity_completeness_v_measure → Struct(h, c, v)

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

# --- Classifier: predict once, score twice ---

fitted_clf = Pipeline.from_instance(
    SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )
).fit(train_cls_expr, features=tuple(feature_names), target="target")

clf_preds = fitted_clf.predict(test_cls_expr, name="my_predicted")

deferred_accuracy = deferred_sklearn_metric(
    expr=clf_preds,
    target="target",
    pred="my_predicted",
    metric=accuracy_score,
)
deferred_f1 = deferred_sklearn_metric(
    expr=clf_preds,
    target="target",
    pred="my_predicted",
    metric=f1_score,
)
# Non-scalar metric: confusion_matrix -> Array(Array(int64))
# return_type is auto-detected from the registry.
deferred_confusion_matrix = deferred_sklearn_metric(
    expr=clf_preds,
    target="target",
    pred="my_predicted",
    metric=confusion_matrix,
)

clf_metrics = (
    deferred_accuracy.as_scalar()
    .name("accuracy")
    .as_table()
    .mutate(f1=deferred_f1.as_scalar())
    .mutate(confusion_matrix=deferred_confusion_matrix.as_scalar())
)

# --- Regressor: predict once, score twice ---

fitted_reg = Pipeline.from_instance(
    SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
    )
).fit(train_reg_expr, features=tuple(feature_names), target="target")

reg_preds = fitted_reg.predict(test_reg_expr, name="my_predicted")

deferred_r2 = deferred_sklearn_metric(
    expr=reg_preds,
    target="target",
    pred="my_predicted",
    metric=r2_score,
)
deferred_mse = deferred_sklearn_metric(
    expr=reg_preds,
    target="target",
    pred="my_predicted",
    metric=mean_squared_error,
)
# Non-scorer metric: d2_tweedie_score is not available as a scorer string —
# it is dispatched via the non-scorer callable path (sign=None, no negation).
deferred_d2_tweedie = deferred_sklearn_metric(
    expr=reg_preds,
    target="target",
    pred="my_predicted",
    metric=d2_tweedie_score,
)

reg_metrics = (
    deferred_r2.as_scalar()
    .name("r2")
    .as_table()
    .mutate(mse=deferred_mse.as_scalar())
    .mutate(d2_tweedie=deferred_d2_tweedie.as_scalar())
)

# --- Clusterer: predict once, score twice ---

# Reuse classification data — target serves as ground-truth labels
fitted_clu = Pipeline.from_instance(
    SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("clu", KMeans(n_clusters=2, random_state=42, n_init=10)),
        ]
    )
).fit(train_cls_expr, features=tuple(feature_names), target="target")

clu_preds = fitted_clu.predict(test_cls_expr, name="my_predicted")

deferred_adj_rand = deferred_sklearn_metric(
    expr=clu_preds,
    target="target",
    pred="my_predicted",
    metric=adjusted_rand_score,
)
# String scorer — sign auto-detected (neg_* -> sign=-1)
deferred_neg_mse = deferred_sklearn_metric(
    expr=clu_preds,
    target="target",
    pred="my_predicted",
    metric="neg_mean_squared_error",
)
# Non-scalar metric: Struct(homogeneity, completeness, v_measure)
# return_type is auto-detected from the registry.
deferred_hcv = deferred_sklearn_metric(
    expr=clu_preds,
    target="target",
    pred="my_predicted",
    metric=homogeneity_completeness_v_measure,
)

clu_metrics = (
    deferred_adj_rand.as_scalar()
    .name("adj_rand")
    .as_table()
    .mutate(neg_mse=deferred_neg_mse.as_scalar())
    .mutate(hcv=deferred_hcv.as_scalar())
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
