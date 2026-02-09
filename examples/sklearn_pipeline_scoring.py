"""Sklearn Pipeline Scoring

This example demonstrates how to score an sklearn pipeline wrapped in xorq
using score_expr() — the high-level API that auto-detects the scorer's
response method (predict, predict_proba, decision_function) and sign.

xorq supports all scorers from sklearn.metrics.get_scorer_names() —
including predict-based scorers (accuracy, f1, neg_mean_squared_error, ...),
predict_proba-based scorers (roc_auc, log_loss, ...), and
decision_function-based scorers.

Uses the bank-marketing dataset with a GradientBoostingClassifier.
"""

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.common.utils.defer_utils import deferred_read_csv
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.pipeline_lib import Pipeline


# --- Data setup ---

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

# --- Pipeline definition ---

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
fitted_pipeline = xorq_pipeline.fit(
    train_table,
    features=tuple(all_features),
    target=target_column,
    cache=cache,
)

# --- score_expr(): high-level API ---
#
# score_expr() returns a deferred expression; score() executes immediately.
# The scorer auto-detects sign and response_method.

# By scorer name string
accuracy_expr = fitted_pipeline.score_expr(test_table, scorer="accuracy")
f1_expr = fitted_pipeline.score_expr(test_table, scorer="f1")

# By scorer name string with automatic sign (neg_* convention)
neg_mse_expr = fitted_pipeline.score_expr(test_table, scorer="neg_mean_squared_error")

# By bare callable (known sklearn scorer function)
precision_expr = fitted_pipeline.score_expr(test_table, scorer=precision_score)


if __name__ in ("__main__", "__pytest_main__"):
    print("=== score_expr (deferred) ===")
    print(f"  Accuracy:  {accuracy_expr.execute():.4f}")
    print(f"  F1:        {f1_expr.execute():.4f}")
    print(f"  Neg MSE:   {neg_mse_expr.execute():.4f}")
    print(f"  Precision: {precision_expr.execute():.4f}")

    # score() is the eager wrapper — takes numpy arrays, executes immediately
    print("\n=== score (eager) ===")
    X = test_table.select(all_features).execute().values
    y = test_table.select(target_column).execute().values.ravel()
    print(f"  Accuracy:  {fitted_pipeline.score(X, y, scorer='accuracy'):.4f}")

    pytest_examples_passed = True
