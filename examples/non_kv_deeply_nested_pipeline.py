"""
Non-KV Deeply Nested sklearn Pipeline with xorq
================================================

This example demonstrates depth-4 nesting using ONLY transformers with KNOWN
schemas (no KV-encoded fallback). It showcases how xorq handles complex nested
structures when all transformers have predictable output schemas.

KEY CONCEPTS
============

1. KNOWN vs KV-ENCODED SCHEMAS
   - KNOWN: Schema computed from transformer params at build time
     Output: Struct{col1: float64, col2: float64, ...}
   - KV-ENCODED: Schema resolved at runtime via get_feature_names_out()
     Output: Array[Struct{key: string, value: float64}]

   This example uses ONLY known-schema transformers.

2. TRANSFORMERS WITH KNOWN SCHEMAS
   - Column-preserving: StandardScaler, RobustScaler, MinMaxScaler,
     MaxAbsScaler, PowerTransformer, QuantileTransformer, Normalizer
   - SimpleImputer: preserves input columns
   - PCA(n_components=int): known output size (n_components columns)

3. MATERIALIZATION POINTS
   - Occur at TOP-LEVEL Step boundaries only
   - Container children (CT, Pipeline) are NOT materialized separately
   - sklearn handles all children internally as one unit

PIPELINE STRUCTURE (Depth 4)
============================

```
SklearnPipeline (top-level)
|-- [Step 1] ColumnTransformer ---------------------- MATERIALIZATION POINT 1
|   |        (KNOWN schema - all children are known)
|   |
|   |   sklearn handles ALL children internally:
|   |   +------------------------------------------------------------+
|   |   | ("numeric_a", SklearnPipeline)                             |
|   |   |   |-- SimpleImputer(strategy="median")     [KNOWN]         |
|   |   |   |-- StandardScaler                       [KNOWN]         |
|   |   |   `-- SklearnPipeline (depth 4)                            |
|   |   |         |-- RobustScaler                   [KNOWN]         |
|   |   |         `-- MinMaxScaler                   [KNOWN]         |
|   |   |                                                            |
|   |   | ("numeric_b", SklearnPipeline)                             |
|   |   |   |-- SimpleImputer(strategy="median")     [KNOWN]         |
|   |   |   `-- PowerTransformer                     [KNOWN]         |
|   |   +------------------------------------------------------------+
|   |
|   `-- Output: Struct{numeric_a__..., numeric_b__...}
|
|-- [Step 2] PCA(n_components=5) -------------------- MATERIALIZATION POINT 2
|            (KNOWN schema: 5 output columns)
|            Reads struct input, outputs 5 principal components
|
`-- [Step 3] RandomForestClassifier
             (PREDICTOR - always separate Step)
             Reads struct input, produces predictions
```

DATA FLOW
=========

```
+------------------+
|   Input Data     |
| (6 features +    |
|  1 target)       |
+--------+---------+
         |
         v
+------------------------------------------------------------------+
|  ColumnTransformer.fit_transform()                               |
|  =====================================                           |
|  sklearn handles internally:                                     |
|  * Routes numeric_a cols to numeric_a_pipeline                   |
|  * Routes numeric_b cols to numeric_b_pipeline                   |
|  * Concatenates all outputs with known names                     |
|                                                                  |
|  xorq outputs as: Struct{..., ..., ...}                         |
|  (KNOWN schema - all column names predictable)                   |
+------------------------------------------------------------------+
         |
         v MATERIALIZATION (into_backend)
+------------------------------------------------------------------+
|  PCA.fit_transform()                                             |
|  ===================                                             |
|  * Reads struct columns as features                              |
|  * Fits PCA on training data                                     |
|  * Outputs n_components columns with known names                 |
|                                                                  |
|  xorq outputs as: Struct{pca0, pca1, pca2, pca3, pca4}          |
+------------------------------------------------------------------+
         |
         v MATERIALIZATION (into_backend)
+------------------------------------------------------------------+
|  RandomForestClassifier.fit() / .predict()                       |
|  =========================================                       |
|  * Reads struct columns as features                              |
|  * Fits classifier on training data                              |
|  * Produces predictions                                          |
+------------------------------------------------------------------+
         |
         v
+------------------+
|   Predictions    |
+------------------+
```
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

import xorq.api as xo
from xorq.expr.ml.pipeline_lib import Pipeline
from xorq.expr.ml.structer import structer_from_instance


def check_structer_type(instance, expr, features):
    """Check the Structer type for a transformer."""
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
n_samples = 200

data = pd.DataFrame(
    {
        # Numeric features - group A
        "age": np.random.randint(18, 80, n_samples).astype(float),
        "income": np.random.randint(20000, 150000, n_samples).astype(float),
        "credit_score": np.random.randint(300, 850, n_samples).astype(float),
        # Numeric features - group B
        "years_employed": np.random.randint(0, 40, n_samples).astype(float),
        "debt_ratio": np.random.uniform(0, 1, n_samples),
        "savings": np.random.randint(0, 100000, n_samples).astype(float),
        # Target
        "approved": np.random.randint(0, 2, n_samples),
    }
)

# Add some missing values to test imputation
data.loc[np.random.choice(n_samples, 20), "age"] = np.nan
data.loc[np.random.choice(n_samples, 15), "income"] = np.nan
data.loc[np.random.choice(n_samples, 10), "years_employed"] = np.nan

expr = xo.memtable(data)

numeric_features_a = ["age", "income", "credit_score"]
numeric_features_b = ["years_employed", "debt_ratio", "savings"]
all_features = tuple(numeric_features_a + numeric_features_b)

# =============================================================================
# Build the deeply nested sklearn pipeline (all known-schema transformers)
# =============================================================================
print("=" * 70)
print("NON-KV DEEPLY NESTED PIPELINE (ALL KNOWN SCHEMAS)")
print("=" * 70)

print("""
This example uses ONLY transformers with KNOWN schemas:
  - StandardScaler, RobustScaler, MinMaxScaler: preserve columns
  - PowerTransformer: preserves columns
  - SimpleImputer: preserves columns
  - PCA(n_components=5): known output size (5 columns)

Benefits of known schemas:
  - Schema computed at build time (before fitting)
  - Column names predictable without running transform
  - Better column lineage tracking
""")

# Depth 4: Innermost pipeline (all column-preserving)
inner_pipeline = SklearnPipeline(
    [
        ("robust", RobustScaler()),
        ("minmax", MinMaxScaler()),
    ]
)

# Depth 3: Pipeline for numeric features group A
numeric_a_pipeline = SklearnPipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("inner", inner_pipeline),  # depth 4
    ]
)

# Depth 3: Pipeline for numeric features group B
numeric_b_pipeline = SklearnPipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson")),
    ]
)

# Depth 2: ColumnTransformer routing to different pipelines
preprocessor = ColumnTransformer(
    [
        ("numeric_a", numeric_a_pipeline, numeric_features_a),
        ("numeric_b", numeric_b_pipeline, numeric_features_b),
    ]
)

# Depth 1: Top-level pipeline with PCA (known n_components)
sklearn_pipe = SklearnPipeline(
    [
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=5)),  # Known schema: 5 components
        ("classifier", RandomForestClassifier(n_estimators=50, random_state=42)),
    ]
)

# =============================================================================
# Verify Structer types
# =============================================================================
print("=" * 70)
print("STRUCTER REGISTRATION STATUS")
print("=" * 70)

print("""
All transformers in this pipeline have KNOWN schema registrations:
""")

test_transformers = [
    ("SimpleImputer", SimpleImputer(), numeric_features_a),
    ("StandardScaler", StandardScaler(), numeric_features_a),
    ("RobustScaler", RobustScaler(), numeric_features_a),
    ("MinMaxScaler", MinMaxScaler(), numeric_features_a),
    ("PowerTransformer", PowerTransformer(), numeric_features_b),
    ("PCA(n_components=5)", PCA(n_components=5), all_features),
]

for name, transformer, features in test_transformers:
    status = check_structer_type(transformer, expr, tuple(features))
    print(f"  {name}: {status}")

print("""
Compare to KV-ENCODED transformers (from other examples):
  - OneHotEncoder: KV-ENCODED (dynamic column count)
  - PolynomialFeatures: KV-ENCODED (interaction terms)
""")

# =============================================================================
# Convert to xorq Pipeline
# =============================================================================
print("=" * 70)
print("XORQ PIPELINE CONVERSION")
print("=" * 70)

xorq_pipeline = Pipeline.from_instance(sklearn_pipe)

print(f"""
Top-level Steps: {len(xorq_pipeline.steps)}

Materialization will occur at:
  1. After ColumnTransformer -> into_backend()
  2. After PCA -> into_backend()
  (Predictor does not materialize transform output)
""")

for i, step in enumerate(xorq_pipeline.steps):
    print(f"Step {i + 1}: {step.name}")
    print(f"  Type: {step.typ.__name__}")

# =============================================================================
# Fit the pipeline
# =============================================================================
print("\n" + "=" * 70)
print("FITTING PIPELINE")
print("=" * 70)

fitted_pipeline = xorq_pipeline.fit(expr, features=all_features, target="approved")
print("Pipeline fitted successfully!")

# =============================================================================
# KNOWN SCHEMA: Ibis Expression Has All Column Names
# =============================================================================
print("\n" + "=" * 70)
print("KNOWN SCHEMA: IBIS EXPRESSION COLUMN NAMES")
print("=" * 70)

print("""
For KNOWN schema transformers, the Ibis expression contains all column names
BEFORE calling .execute(). This is because the schema is computed at build time
from transformer parameters, not resolved at runtime.
""")

# Get the prediction expression (NOT executed yet)
predictions_expr = fitted_pipeline.predict(expr)

print(f"predictions_expr type: {type(predictions_expr).__name__}")
print("\nIbis expression schema (columns known at build time):")
print(f"  {predictions_expr.columns}")

# Show transform expression for a transformer step
print("\n--- Transform Step Schema Example ---")
# Get first transform step (ColumnTransformer)
ct_step = fitted_pipeline.transform_steps[0]
transform_expr = ct_step.transform(expr)
print("\nColumnTransformer transform expression:")
print(f"  Type: {type(transform_expr).__name__}")
print(f"  Columns: {transform_expr.columns}")

# Show PCA transform (known n_components=5)
pca_step = fitted_pipeline.transform_steps[1]
# Chain: CT output -> PCA input
ct_output = ct_step.transform(expr)
pca_transform_expr = pca_step.transform(ct_output)
print("\nPCA(n_components=5) transform expression:")
print(f"  Type: {type(pca_transform_expr).__name__}")
print(f"  Columns: {pca_transform_expr.columns}")
print("  (5 columns known at build time from n_components param)")

# =============================================================================
# Predict
# =============================================================================
print("\n" + "=" * 70)
print("PREDICTIONS")
print("=" * 70)

# Now execute the expression
predictions = predictions_expr.execute()
print("\nPrediction distribution:")
print(predictions["predicted"].value_counts())

# =============================================================================
# Verify against sklearn
# =============================================================================
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

X = data[list(all_features)]
y = data["approved"]
sklearn_pipe.fit(X, y)
sklearn_preds = sklearn_pipe.predict(X)

match = np.array_equal(predictions["predicted"].values, sklearn_preds)
print(f"\nPredictions match sklearn: {match}")
print(f"Accuracy: {(predictions['predicted'].values == y.values).mean():.2%}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(
    """
KEY TAKEAWAYS:

1. ALL KNOWN SCHEMAS
   - Every transformer has predictable output schema
   - No runtime schema resolution needed
   - Schema computed from transformer params at build time

2. DEPTH 4 NESTING
   - Pipeline -> ColumnTransformer -> Pipeline -> Pipeline
   - sklearn handles all children internally as one unit
   - xorq materializes at top-level Step boundaries only

3. MATERIALIZATION POINTS
   - 2 materializations: after ColumnTransformer, after PCA
   - Container children are NOT materialized separately
   - Predictor (RFC) does not materialize transform output

4. COMPARISON TO KV-ENCODED EXAMPLES
   - KV-encoded: Array[Struct{key, value}] (runtime names)
   - Known: Struct{col1, col2, ...} (build-time names)
   - This example shows the "all known" case for comparison
"""
)
