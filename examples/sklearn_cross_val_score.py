"""Cross-Validation with deferred_cross_val_score

Demonstrates deferred_cross_val_score — the xorq equivalent of sklearn's
cross_val_score. Shows three cv strategies:

1. int cv — uses train_test_splits internally, which partitions rows via
   deterministic hashing (controlled by random_seed).  This produces different
   folds than sklearn's index-based KFold, so per-fold scores differ.  The
   mean converges to the same value (typically within ~1%).

2. sklearn splitter — passes the splitter object (e.g. StratifiedKFold)
   directly.  Rows are sorted by a deterministic hash (controlled by
   random_seed) so the UDWF always sees rows in the same order.  To get
   identical results from standalone sklearn, sort the pandas DataFrame
   with apply_deterministic_sort using the same random_seed.

3. TimeSeriesSplit — requires order_by to specify the temporal column
   so that expanding-window semantics are respected.

deferred_cross_val_score returns a CrossValScore object — nothing is
materialized until .execute() is called.  The .fold_expr attribute
exposes the deferred fold assignments (0=unused, 1=train, 2=test)
for inspection before execution.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from toolz import curry

import xorq.api as xo
from xorq.expr.ml.cross_validation import (
    apply_deterministic_sort,
    deferred_cross_val_score,
)
from xorq.expr.ml.pipeline_lib import Pipeline


con = xo.connect()

RANDOM_STATE = 42

# --- Splitters ---
sklearn_stratified_k_fold = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE
)
sklearn_time_series_split = TimeSeriesSplit(n_splits=5)

# --- Data setup ---
feature_names = tuple(f"f{i}" for i in range(10))
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=5, random_state=RANDOM_STATE
)
df = pd.DataFrame(X, columns=list(feature_names)).assign(target=y, t=range(500))
data = con.register(df, "data")

# --- Pipeline (unfitted — deferred_cross_val_score fits per fold) ---
sk_pipeline = SklearnPipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ]
)
pipeline = Pipeline.from_instance(sk_pipeline)


make_cv = curry(
    deferred_cross_val_score,
    pipeline,
    data,
    feature_names,
    "target",
    random_seed=RANDOM_STATE,
)

# --- int cv: 5-fold (hash-based splitting) ---
# random_seed controls the deterministic hash partitioning
cv_int = make_cv(cv=5)

# --- sklearn splitter: StratifiedKFold (index-based splitting) ---
# random_seed controls the deterministic row ordering; the splitter's
# random_state controls shuffle order within the splitter.
cv_stratified = make_cv(cv=sklearn_stratified_k_fold)

# --- TimeSeriesSplit: expanding window (order_by specifies temporal column) ---
cv_timeseries = make_cv(cv=sklearn_time_series_split, order_by="t")


if __name__ == "__pytest_main__":
    scores_int = cv_int.execute()
    scores_stratified = cv_stratified.execute()
    scores_timeseries = cv_timeseries.execute()

    # --- Reproduce sklearn splitter results with standalone sklearn ---
    # Sort the DataFrame by the same deterministic hash so sklearn sees
    # the same row order as the UDWF.
    df_sorted = apply_deterministic_sort(data, random_seed=RANDOM_STATE).execute()
    sklearn_scores = cross_val_score(
        sk_pipeline,
        df_sorted[list(feature_names)].values,
        df_sorted["target"].values,
        cv=sklearn_stratified_k_fold,
        scoring="accuracy",
    )

    # Reproduce TimeSeriesSplit — sort by the same temporal column.
    df_sorted_by_t = data.execute().sort_values("t")
    sklearn_ts_scores = cross_val_score(
        sk_pipeline,
        df_sorted_by_t[list(feature_names)].values,
        df_sorted_by_t["target"].values,
        cv=sklearn_time_series_split,
        scoring="accuracy",
    )

    print("=== int cv (hash-based folds, random_seed=42) ===")
    print(f"  scores: {scores_int}")
    print(f"  mean:   {scores_int.mean():.4f} +/- {scores_int.std():.4f}")
    print()

    print("=== StratifiedKFold (deterministic sort + index-based folds) ===")
    print(f"  xorq scores:    {scores_stratified}")
    print(f"  sklearn scores: {sklearn_scores}")
    print(f"  match: {np.allclose(scores_stratified, sklearn_scores)}")
    print()

    print("=== TimeSeriesSplit (natural row order, expanding window) ===")
    print(f"  xorq scores:    {scores_timeseries}")
    print(f"  sklearn scores: {sklearn_ts_scores}")
    print(f"  match: {np.allclose(scores_timeseries, sklearn_ts_scores)}")
    print()

    # --- Inspecting fold assignments before execution ---
    # fold_expr is a deferred ibis expression with fold_0..fold_k columns
    # (0=unused, 1=train, 2=test).  You can peek at it without running
    # the full cross-validation.
    print("=== Inspecting deferred fold assignments (TimeSeriesSplit) ===")
    fold_df = cv_timeseries.fold_expr.execute()
    fold_cols = [c for c in fold_df.columns if c.startswith("fold_")]
    print(f"  fold columns: {fold_cols}")
    print("  encoding: 0=unused, 1=train, 2=test")
    print("  first 10 rows:")
    print(fold_df[fold_cols].head(10).to_string(index=False))
    print()

    # Show train/test sizes per fold (TimeSeriesSplit uses expanding windows)
    fold_summary = "\n".join(
        f"  {col}: train={(fold_df[col] == 1).sum()}, test={(fold_df[col] == 2).sum()}, unused={(fold_df[col] == 0).sum()}"
        for col in fold_cols
    )
    print(fold_summary)

    pytest_examples_passed = True
