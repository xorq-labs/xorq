"""Classification pipeline on the bank marketing dataset with preprocessing, caching,
confusion matrix, and ROC-AUC evaluation.

Traditional approach: You would load a CSV with pandas, manually preprocess numeric and
categorical columns, build an sklearn Pipeline with ColumnTransformer and
GradientBoostingClassifier, fit it, predict, and compute metrics -- all tightly coupled
in a single script with no caching between steps.

With xorq: CSV reading is deferred, preprocessing and model training are wrapped in a
cached pipeline, and each step's output is automatically persisted to a ParquetCache.
The pipeline is backend-agnostic, so the same code works whether the data originates
from a local file or a remote database.
"""

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.common.utils.defer_utils import deferred_read_csv
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.pipeline_lib import Pipeline


target_column = "deposit"
numeric_features = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]
categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
all_features = numeric_features + categorical_features
# Set up cache & input expression, train test split
con = xo.connect()
cache = ParquetCache.from_kwargs(
    source=con,
    relative_path="./tmp-cache",
    base_path=Path(".").absolute(),
)

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
# Define the preprocessing and modeling pipeline
preprocessor = ColumnTransformer(
    [
        (
            "num",
            SklearnPipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            SklearnPipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            categorical_features,
        ),
    ]
)

sklearn_pipeline = SklearnPipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ]
)

xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
# This pipeline is now deferred & has has caching for all registered sklearn pipeline operations
fitted_pipeline = xorq_pipeline.fit(
    train_table,
    features=tuple(all_features),
    target=target_column,
    cache=cache,
)

encoded_test = fitted_pipeline.transform(test_table)
predicted_test = fitted_pipeline.predict(test_table)

if __name__ in ("__main__", "__pytest_main__"):
    predictions_df = predicted_test.execute()
    binary_predictions = predictions_df["predicted"]

    cm = confusion_matrix(predictions_df[target_column], binary_predictions)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    auc = roc_auc_score(predictions_df[target_column], binary_predictions)
    print(f"\nAUC Score: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(predictions_df[target_column], binary_predictions))
    pytest_examples_passed = True
