"""Sklearn Deferred Metrics

Demonstrates deferred_sklearn_metric — the low-level API for computing
sklearn metrics on xorq expressions. Shows classifier, regressor, and
clusterer pipelines each scored with two metrics composed into a single
deferred table per pipeline.

Multiple metrics on the same predictions compose naturally: predict once,
build multiple metric expressions, then combine them into one table with
a single execute() call.

deferred_sklearn_metric accepts scorer name strings, bare callables
(known sklearn metric functions), or make_scorer objects. Sign is
auto-detected from the scorer (e.g. "neg_mean_squared_error" -> sign=-1).
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
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

clf_preds = fitted_clf.predict(test_cls_expr)

clf_metrics = (
    deferred_sklearn_metric(
        expr=clf_preds,
        target="target",
        pred_col="predicted",
        scorer=accuracy_score,
    )
    .as_scalar()
    .name("accuracy")
    .as_table()
    .mutate(
        f1=deferred_sklearn_metric(
            expr=clf_preds,
            target="target",
            pred_col="predicted",
            scorer=f1_score,
        ).as_scalar()
    )
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

reg_preds = fitted_reg.predict(test_reg_expr)

reg_metrics = (
    deferred_sklearn_metric(
        expr=reg_preds,
        target="target",
        pred_col="predicted",
        scorer=r2_score,
    )
    .as_scalar()
    .name("r2")
    .as_table()
    .mutate(
        mse=deferred_sklearn_metric(
            expr=reg_preds,
            target="target",
            pred_col="predicted",
            scorer=mean_squared_error,
        ).as_scalar()
    )
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

clu_preds = fitted_clu.predict(test_cls_expr)

clu_metrics = (
    deferred_sklearn_metric(
        expr=clu_preds,
        target="target",
        pred_col="predicted",
        scorer=adjusted_rand_score,
    )
    .as_scalar()
    .name("adj_rand")
    .as_table()
    .mutate(
        # String scorer — sign auto-detected (neg_* -> sign=-1)
        neg_mse=deferred_sklearn_metric(
            expr=clu_preds,
            target="target",
            pred_col="predicted",
            scorer="neg_mean_squared_error",
        ).as_scalar()
    )
)


if __name__ in ("__main__", "__pytest_main__"):
    # One execute per pipeline — each returns both metrics
    print("=== Classifier (RandomForestClassifier) ===")
    print(clf_metrics.execute().to_string(index=False))

    print("\n=== Regressor (RandomForestRegressor) ===")
    print(reg_metrics.execute().to_string(index=False))

    print("\n=== Clusterer (KMeans) ===")
    print(clu_metrics.execute().to_string(index=False))

    pytest_examples_passed = True
