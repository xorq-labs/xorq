"""Classification on the penguins dataset using StandardScaler and RandomForestClassifier,
with deferred metric computation.

Traditional approach: You would load penguin data with seaborn or pandas, filter nulls,
split with sklearn's train_test_split, build an sklearn Pipeline, fit and predict
in-memory, then manually compute each metric (accuracy, precision, recall, F1, ROC AUC).

With xorq: Data is loaded through Ibis expressions and split with expression-level
train_test_splits. The sklearn pipeline is wrapped with deferred execution, and metrics
are computed as lazy expressions via deferred_sklearn_metric, so they are only
materialized when .execute() is called. This enables automatic caching and composability
of metric results.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.api import _
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline


penguins = xo.examples.penguins.fetch()

clean_data = penguins.filter(
    _.bill_length_mm.notnull()
    & _.bill_depth_mm.notnull()
    & _.flipper_length_mm.notnull()
    & _.body_mass_g.notnull()
    & _.species.notnull()
)

train_data, test_data = xo.train_test_splits(
    clean_data,
    test_sizes=0.2,  # 80/20 split
    num_buckets=1000,
    random_seed=42,
)

feature_cols = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

sklearn_pipeline = SkPipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

# Wrap with xorq Pipeline
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

fitted_pipeline = xorq_pipeline.fit(train_data, features=feature_cols, target="species")

predictions = fitted_pipeline.predict(test_data, name="my_predicted")
predictions_proba = fitted_pipeline.predict_proba(test_data, name="my_predicted_proba")

accuracy = deferred_sklearn_metric(
    expr=predictions,
    target="species",
    pred="my_predicted",
    metric=accuracy_score,
)

precision = deferred_sklearn_metric(
    expr=predictions,
    target="species",
    pred="my_predicted",
    metric=precision_score,
    metric_kwargs={"average": "weighted", "zero_division": 0},
)

recall = deferred_sklearn_metric(
    expr=predictions,
    target="species",
    pred="my_predicted",
    metric=recall_score,
    metric_kwargs={"average": "weighted", "zero_division": 0},
)

f1 = deferred_sklearn_metric(
    expr=predictions,
    target="species",
    pred="my_predicted",
    metric=f1_score,
    metric_kwargs={"average": "weighted", "zero_division": 0},
)

roc_auc = deferred_sklearn_metric(
    expr=predictions_proba,
    target="species",
    pred="my_predicted_proba",
    metric=roc_auc_score,
    metric_kwargs={"multi_class": "ovr", "average": "weighted"},
)

feature_importances_expr = fitted_pipeline.feature_importances(test_data)


if __name__ in ("__pytest_main__"):
    print(f"  Accuracy:  {accuracy.execute():.4f}")
    print(f"  Precision: {precision.execute():.4f}")
    print(f"  Recall:    {recall.execute():.4f}")
    print(f"  F1 Score:  {f1.execute():.4f}")
    print(f"  ROC AUC:   {roc_auc.execute():.4f}")

    # Extract and display feature importances (deferred execution)
    print("\n  Feature Importances:")
    feature_importances = feature_importances_expr.execute()[
        "feature_importances"
    ].iloc[0]
    for feature, importance in zip(feature_cols, feature_importances):
        print(f"    {feature:20s}: {importance:.4f}")

    pytest_examples_passed = True
