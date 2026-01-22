"""
Mixed Pipeline Materialization Example
======================================

This example demonstrates how xorq handles pipelines with mixed transformer types,
focusing on materialization points and format conversion between steps.

KEY CONCEPTS
============

1. MATERIALIZATION POINTS
   - Each top-level Step boundary triggers `into_backend()`
   - Creates a temp table in the backend (DuckDB/DataFusion)
   - Enables lazy evaluation - nothing runs until `.execute()`

2. FORMAT CONVERSION
   - KV-ENCODED output: Array[Struct{key: string, value: float64}]
   - KNOWN output: Struct{col1: float64, col2: float64, ...}
   - xorq automatically decodes KV format for next step

3. STEP TYPES
   - KNOWN: Transformers with predictable output schema (StandardScaler, PCA)
   - KV-ENCODED: Transformers with dynamic output (OneHotEncoder, PolynomialFeatures)
   - PREDICTOR: Classifiers/Regressors (RandomForest, LogisticRegression)
   - CLUSTERER: Clustering algorithms (KMeans, DBSCAN)
   - UNREGISTERED: Custom transformers without Structer registration

4. ALLOW_UNREGISTERED FLAG
   - Default: False - raises ValueError for unregistered transformers
   - True: Allows custom transformers (wraps them for execution)
   - Predictors, clusterers, and transductive estimators are always allowed

PIPELINE EXAMPLES
=================

Example 1: All Registered (3 materializations)
----------------------------------------------
```
[StandardScaler] -> mat -> [SelectKBest] -> mat -> [RandomForest]
     KNOWN                    KNOWN               PREDICTOR
```

Example 2: Mixed with ColumnTransformer (2 materializations)
------------------------------------------------------------
```
[ColumnTransformer] -> mat -> [SelectKBest] -> mat -> [RandomForest]
    KV-ENCODED                   KNOWN              PREDICTOR
```
Note: CT children are NOT materialized separately

Example 3: Unregistered Transformer
-----------------------------------
```
[CustomTransformer] -> raises ValueError (by default)
[CustomTransformer] -> works with allow_unregistered=True
```
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import xorq.api as xo
from xorq.expr.ml.pipeline_lib import (
    Pipeline,
    is_fit_predict_only,
    is_fit_transform_only,
)
from xorq.expr.ml.structer import structer_from_instance


def check_step_type(instance, expr, features):
    """
    Check the step type for a transformer/estimator.

    Returns one of:
      - KNOWN: Registered transformer with predictable schema
      - KV-ENCODED: Registered transformer with dynamic schema
      - PREDICTOR: Classifier or regressor (has predict, no transform)
      - CLUSTERER: Clustering algorithm (ClusterMixin)
      - TRANSDUCTIVE: fit_transform only (TSNE) or fit_predict only (DBSCAN)
      - UNREGISTERED: No Structer registration
    """
    typ = type(instance)

    # Check if it's a predictor (has predict but not transform)
    is_predictor = hasattr(instance, "predict") and not hasattr(instance, "transform")
    if is_predictor:
        # Distinguish classifier vs regressor
        from sklearn.base import ClassifierMixin, RegressorMixin

        if isinstance(instance, ClassifierMixin):
            return "PREDICTOR (Classifier)"
        elif isinstance(instance, RegressorMixin):
            return "PREDICTOR (Regressor)"
        return "PREDICTOR"

    # Check if it's a clusterer
    from sklearn.base import ClusterMixin

    if isinstance(instance, ClusterMixin):
        return "CLUSTERER"

    # Check if it's transductive (fit_transform only or fit_predict only)
    if is_fit_transform_only(typ):
        return "TRANSDUCTIVE (fit_transform)"
    if is_fit_predict_only(typ):
        return "TRANSDUCTIVE (fit_predict)"

    # Check Structer registration
    try:
        structer = structer_from_instance(instance, expr, features=features)
        if structer.is_kv_encoded:
            return "KV-ENCODED"
        return "KNOWN"
    except ValueError:
        return "UNREGISTERED"


# =============================================================================
# Create sample data
# =============================================================================
np.random.seed(42)
n_samples = 100

data = pd.DataFrame(
    {
        "age": np.random.randint(25, 70, n_samples).astype(float),
        "income": np.random.randint(30000, 120000, n_samples).astype(float),
        "credit_score": np.random.randint(600, 850, n_samples).astype(float),
        "education": np.random.choice(
            ["high_school", "bachelor", "master", "phd"], n_samples
        ),
        "employment": np.random.choice(["part_time", "full_time"], n_samples),
        "target": np.random.randint(0, 2, n_samples),
    }
)

expr = xo.memtable(data)

numeric_features = ["age", "income", "credit_score"]
categorical_features = ["education", "employment"]
all_features = tuple(numeric_features + categorical_features)

# =============================================================================
# Example 1: All Known Schema (StandardScaler -> SelectKBest -> RFC)
# =============================================================================
print("=" * 70)
print("EXAMPLE 1: ALL KNOWN SCHEMA")
print("=" * 70)

print("""
Pipeline: StandardScaler -> SelectKBest -> RandomForestClassifier

All transformers have KNOWN schema:
  - StandardScaler: preserves column names
  - SelectKBest: outputs k columns with known names
  - RandomForestClassifier: PREDICTOR (always allowed)

Materialization flow:
```
[StandardScaler] ──────> [SelectKBest] ──────> [RFC]
     KNOWN                   KNOWN           PREDICTOR
       │                       │
       ▼                       ▼
  into_backend()          into_backend()
  (mat point 1)           (mat point 2)
```
""")

sklearn_pipe_1 = SklearnPipeline(
    [
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=2)),
        ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
    ]
)

xorq_pipe_1 = Pipeline.from_instance(sklearn_pipe_1)
print(f"xorq Steps: {len(xorq_pipe_1.steps)}")
for i, step in enumerate(xorq_pipe_1.steps):
    stype = check_step_type(step.instance, expr, tuple(numeric_features))
    print(f"  {i + 1}. {step.name}: {step.typ.__name__} [{stype}]")

fitted_1 = xorq_pipe_1.fit(expr, features=tuple(numeric_features), target="target")
preds_1 = fitted_1.predict(expr).execute()
print(f"\nPredictions: {preds_1['predicted'].value_counts().to_dict()}")

# =============================================================================
# Example 2: Mixed with ColumnTransformer (KV-encoded)
# =============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: COLUMNTRANSFORMER (KV-ENCODED)")
print("=" * 70)

print("""
Pipeline: ColumnTransformer -> SelectKBest -> RandomForestClassifier

ColumnTransformer is KV-ENCODED (contains OneHotEncoder):
  - Output: Array[Struct{key, value}]
  - SelectKBest decodes KV input automatically

Materialization flow:
```
[ColumnTransformer] ──────> [SelectKBest] ──────> [RFC]
  (KV-ENCODED)                 (KNOWN)          PREDICTOR
       │                          │
       ▼                          ▼
  into_backend()             into_backend()
  (mat point 1)              (mat point 2)

  sklearn handles             xorq decodes
  all CT children             KV format
  internally                  automatically
```
""")

sklearn_pipe_2 = SklearnPipeline(
    [
        (
            "preprocessor",
            ColumnTransformer(
                [
                    ("num", StandardScaler(), numeric_features),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        categorical_features,
                    ),
                ]
            ),
        ),
        ("selector", SelectKBest(f_classif, k=5)),
        ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
    ]
)

xorq_pipe_2 = Pipeline.from_instance(sklearn_pipe_2)
print(f"xorq Steps: {len(xorq_pipe_2.steps)}")
for i, step in enumerate(xorq_pipe_2.steps):
    # Check step type - use appropriate features for each step
    if step.typ.__name__ == "ColumnTransformer":
        stype = "KV-ENCODED"  # CT with OneHotEncoder
    elif step.typ.__name__ == "SelectKBest":
        stype = "KNOWN"  # SelectKBest has known output schema
    else:
        stype = check_step_type(step.instance, expr, all_features)
    print(f"  {i + 1}. {step.name}: {step.typ.__name__} [{stype}]")

fitted_2 = xorq_pipe_2.fit(expr, features=all_features, target="target")
preds_2 = fitted_2.predict(expr).execute()
print(f"\nPredictions: {preds_2['predicted'].value_counts().to_dict()}")

# =============================================================================
# Example 3: Structer Registration Status
# =============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: STEP TYPE CLASSIFICATION")
print("=" * 70)

print("""
xorq classifies steps into different types:
  - KNOWN: Predictable schema (StandardScaler, PCA, SelectKBest)
  - KV-ENCODED: Dynamic schema (OneHotEncoder, PolynomialFeatures)
  - PREDICTOR: Classifiers/Regressors (always allowed)
  - CLUSTERER: Clustering algorithms (always allowed)
  - TRANSDUCTIVE: fit_transform or fit_predict only (always allowed)
  - UNREGISTERED: Custom transformers (require allow_unregistered=True)
""")

test_instances = [
    ("StandardScaler", StandardScaler(), tuple(numeric_features)),
    ("SelectKBest(k=2)", SelectKBest(k=2), tuple(numeric_features)),
    ("OneHotEncoder", OneHotEncoder(), tuple(categorical_features)),
    ("RandomForestClassifier", RandomForestClassifier(), tuple(numeric_features)),
    ("LogisticRegression", LogisticRegression(), tuple(numeric_features)),
    ("KMeans", KMeans(n_clusters=3), tuple(numeric_features)),
]

for name, instance, features in test_instances:
    status = check_step_type(instance, expr, features)
    print(f"  {name}: {status}")

# =============================================================================
# Example 4: Unregistered Transformer - Failure Case
# =============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: UNREGISTERED TRANSFORMER (FAILURE)")
print("=" * 70)

print("""
Some sklearn transformers don't have Structer registration and will FAIL
by default. This protects against accidentally using unsupported transformers.

FunctionTransformer is a real sklearn transformer that is NOT registered.
""")

# Show that FunctionTransformer is unregistered
# Note: feature_names_out="one-to-one" enables get_feature_names_out()
func_transformer = FunctionTransformer(func=np.log1p, feature_names_out="one-to-one")
status = check_step_type(func_transformer, expr, tuple(numeric_features))
print(f"FunctionTransformer step type: {status}")

# Try to create a pipeline with unregistered transformer (should fail)
sklearn_pipe_unregistered = SklearnPipeline(
    [
        (
            "log_transform",
            FunctionTransformer(func=np.log1p, feature_names_out="one-to-one"),
        ),
        ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
    ]
)

print("\nAttempting Pipeline.from_instance() with default allow_unregistered=False:")
try:
    xorq_pipe_unregistered = Pipeline.from_instance(sklearn_pipe_unregistered)
    print("  ERROR: Should have raised ValueError!")
except ValueError as e:
    print("  EXPECTED: ValueError raised!")
    print(f"  Message: {str(e)[:80]}...")

# =============================================================================
# Example 5: Unregistered Transformer - Success with Flag
# =============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 5: UNREGISTERED TRANSFORMER (SUCCESS)")
print("=" * 70)

print("""
With allow_unregistered=True, unregistered transformers are allowed.
xorq wraps them and uses KV-encoding (schema resolved at runtime).
""")

print("Creating Pipeline with allow_unregistered=True:")
xorq_pipe_unregistered = Pipeline.from_instance(
    sklearn_pipe_unregistered, allow_unregistered=True
)
print(f"  Success! xorq Steps: {len(xorq_pipe_unregistered.steps)}")
for i, step in enumerate(xorq_pipe_unregistered.steps):
    stype = check_step_type(step.instance, expr, tuple(numeric_features))
    print(f"    {i + 1}. {step.name}: {step.typ.__name__} [{stype}]")

print("""
NOTE: Execution of unregistered transformers requires:
  - get_feature_names_out() implementation
  - Numeric-only data for transformers like FunctionTransformer(np.log1p)

For production use, consider registering your transformer with Structer.
""")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: MATERIALIZATION & STEP TYPES")
print("=" * 70)
print(
    """
MATERIALIZATION RULES:
  1. Occurs at TOP-LEVEL Step boundaries only
  2. Each into_backend() creates a temp table
  3. Container children (CT, FU, Pipeline) are NOT materialized separately

STEP TYPES:
  - KNOWN: Predictable schema at build time
  - KV-ENCODED: Dynamic schema resolved at runtime
  - PREDICTOR: Classifiers/Regressors (always allowed)
  - CLUSTERER: Clustering algorithms (always allowed)
  - TRANSDUCTIVE: fit_transform/fit_predict only (always allowed)
  - UNREGISTERED: Requires allow_unregistered=True

FORMAT CONVERSION:
  - KV-ENCODED -> KNOWN: automatic decoding
  - KNOWN -> KV-ENCODED: automatic encoding
  - xorq handles this transparently between Steps

ALLOW_UNREGISTERED FLAG:
  - Default False: Raises ValueError for unregistered transformers
  - True: Allows custom transformers to be wrapped and executed
  - Predictors, clusterers, transductive estimators always allowed
"""
)
