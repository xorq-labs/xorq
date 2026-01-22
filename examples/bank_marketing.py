"""
Bank Marketing Example

Demonstrates using a standard sklearn Pipeline with xorq for deferred execution.
"""

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xorq.api as xo
import xorq.expr.selectors as s
from xorq.caching import ParquetCache
from xorq.common.utils.defer_utils import deferred_read_csv
from xorq.expr.ml import train_test_splits


con = xo.connect()
cache = ParquetCache.from_kwargs(
    source=con,
    relative_path="./tmp-cache",
    base_path=Path(".").absolute(),
)

target_column = "deposit"
predicted_col = "predicted"

expr = deferred_read_csv(
    path=xo.options.pins.get_path("bank-marketing"),
    con=con,
).mutate(**{target_column: (xo._[target_column] == "yes").cast("int")})

train_table, test_table = expr.pipe(
    train_test_splits,
    test_sizes=[0.5, 0.5],
    num_buckets=2,
    random_seed=42,
)

string_features = list(train_table.select(s.of_type(str)).columns)
numeric_features = list(
    train_table.select(
        s.all_of(s.numeric(), s.matches(f"^(?!{target_column}$)"))
    ).columns
)

sklearn_pipeline = SklearnPipeline([
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), string_features),
    ])),
    ("classifier", GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)),
])

pipeline = xo.Pipeline.from_instance(sklearn_pipeline)
fitted_pipeline = pipeline.fit(
    train_table,
    features=tuple(numeric_features + string_features),
    target=target_column,
    cache=cache,
)

predicted_test = fitted_pipeline.predict(test_table)


if __name__ == "__pytest_main__":
    predictions_df = predicted_test.execute()
    binary_predictions = predictions_df[predicted_col]

    cm = confusion_matrix(predictions_df[target_column], binary_predictions)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    proba_df = fitted_pipeline.predict_proba(test_table).execute()
    proba_class_1 = [row[1]["value"] for row in proba_df["proba"]]
    auc = roc_auc_score(predictions_df[target_column], proba_class_1)
    print(f"\nAUC Score: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(predictions_df[target_column], binary_predictions))
    pytest_examples_passed = True
