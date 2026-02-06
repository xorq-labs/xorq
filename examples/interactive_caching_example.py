"""Interactive caching demo: shows how xorq caching accelerates iterative data science.

Run this script multiple times to see caching in action:

    # Run 1: cold cache — everything computed from scratch
    python interactive_caching_example.py --clear

    # Run 2: warm cache — cached results loaded instantly
    python interactive_caching_example.py

    # Run 3: compare against no caching
    python interactive_caching_example.py --no-cache

Then try editing the CONFIGURATION section below (e.g. swap the classifier)
and re-run — data loading stays cached, only the model retrains.

See interactive-xorq.md for the full guided walkthrough.
"""

import shutil
import time
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.common.utils.defer_utils import deferred_read_csv
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.pipeline_lib import Pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — change these between runs to see partial cache reuse!
# ═══════════════════════════════════════════════════════════════════════════════

# --- Classifier: uncomment ONE of these ---
classifier = GradientBoostingClassifier(n_estimators=200, random_state=42)
# classifier = RandomForestClassifier(n_estimators=200, random_state=42)
# classifier = KNeighborsClassifier(n_neighbors=5)

# --- Feature set: uncomment ONE of these ---
numeric_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
# numeric_features = ["age", "balance", "duration"]  # smaller set — try it!

categorical_features = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
# categorical_features = ["job", "marital", "education"]  # smaller set — try it!

# ═══════════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path(__file__).parent / "interactive-cache"
TARGET = "deposit"


def run(use_cache=True):
    con = xo.connect()

    if use_cache:
        cache = ParquetCache.from_kwargs(
            source=con,
            relative_path=str(CACHE_DIR),
            base_path=Path(__file__).parent.absolute(),
        )
    else:
        cache = None

    all_features = numeric_features + categorical_features

    # Stage 1: Load and encode data (cached separately so changes to the model
    # below don't force a re-read of the CSV)
    expr = deferred_read_csv(
        path=xo.options.pins.get_path("bank-marketing"),
        con=con,
    ).mutate(**{TARGET: (xo._[TARGET] == "yes").cast("int")})

    if cache:
        expr = expr.cache(cache=cache)

    # Stage 2: Train/test split
    train_table, test_table = expr.pipe(
        train_test_splits,
        test_sizes=[0.7, 0.3],
        random_seed=42,
    )

    # Stage 3: Preprocessing + classifier
    preprocessor = ColumnTransformer([
        ("num", SklearnPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", SklearnPipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_features),
    ])

    sklearn_pipeline = SklearnPipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    fitted = xorq_pipeline.fit(
        train_table,
        features=tuple(all_features),
        target=TARGET,
        cache=cache,
    )

    predicted = fitted.predict(test_table)

    # Execute — this is where caching makes a difference
    t0 = time.perf_counter()
    df = predicted.execute()
    elapsed = time.perf_counter() - t0

    # Metrics
    y_true, y_pred = df[TARGET], df["predicted"]
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return elapsed, acc, auc, len(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive caching demo")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--clear", action="store_true", help="Clear cache before running")
    args = parser.parse_args()

    if args.clear and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"  Cleared cache at {CACHE_DIR}/\n")

    use_cache = not args.no_cache
    mode = "WITH caching" if use_cache else "WITHOUT caching"

    print(f"  Running {mode}")
    print(f"  Classifier: {classifier.__class__.__name__}")
    print(f"  Features:   {len(numeric_features)} numeric + {len(categorical_features)} categorical")
    print()

    elapsed, acc, auc, n_rows = run(use_cache=use_cache)

    print(f"  Pipeline executed in {elapsed:.2f}s  ({n_rows} test rows)")
    print(f"  Accuracy: {acc:.4f}   ROC AUC: {auc:.4f}")

    cache_exists = CACHE_DIR.exists() and any(CACHE_DIR.iterdir())
    print(f"\n  Cache: {'warm' if cache_exists else 'empty'} ({CACHE_DIR.name}/)")
