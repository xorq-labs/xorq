"""
Deeply Nested sklearn Pipeline with xorq
=========================================

This example demonstrates depth-4 nesting with ColumnTransformer, FeatureUnion,
and sklearn Pipeline. It explains the key concepts of how xorq handles these
complex structures.

KEY CONCEPTS
============

1. MATERIALIZATION POINTS
   - Materialization occurs at TOP-LEVEL Step boundaries only
   - Container children are handled by sklearn as a single unit
   - Each `into_backend()` call creates a temp table

   For this pipeline:
   ```
   [ColumnTransformer] -> materialize -> [SelectKBest] -> materialize -> [RFC]
   ```
   = 2 materialization points (not counting final prediction)

2. KV-ENCODED vs KNOWN SCHEMA
   - KV-ENCODED: Schema resolved at runtime (OneHotEncoder, PolynomialFeatures)
     Output: Array[Struct{key: string, value: float64}]
   - KNOWN: Schema computed from params (StandardScaler, SimpleImputer)
     Output: Struct{col1: float64, col2: float64, ...}
   - Container is KV-ENCODED if ANY child is KV-ENCODED

3. REGISTERED vs UNREGISTERED
   - Registered: Has Structer (StandardScaler, OneHotEncoder, ColumnTransformer)
   - Unregistered: No Structer, requires allow_unregistered=True
   - Predictors (RandomForest) and clusterers are always allowed

PIPELINE STRUCTURE (Depth 4)
============================

```
SklearnPipeline (top-level)
├── [Step 1] ColumnTransformer ─────────────────── MATERIALIZATION POINT 1
│   │        (KV-ENCODED because contains OneHotEncoder)
│   │
│   │   sklearn handles ALL children internally:
│   │   ┌─────────────────────────────────────────────────────────┐
│   │   │ ("numeric", FeatureUnion)                               │
│   │   │   ├── ("scaled", Pipeline)                              │
│   │   │   │     ├── SimpleImputer [KNOWN]                       │
│   │   │   │     └── StandardScaler [KNOWN]                      │
│   │   │   └── ("poly", Pipeline)                                │
│   │   │         ├── SimpleImputer [KNOWN]                       │
│   │   │         └── PolynomialFeatures [KV-ENCODED]             │
│   │   │                                                         │
│   │   │ ("categorical", Pipeline)                               │
│   │   │   ├── SimpleImputer [KNOWN]                             │
│   │   │   └── OneHotEncoder [KV-ENCODED]                        │
│   │   └─────────────────────────────────────────────────────────┘
│   │
│   └── Output: KV-encoded array (single "transformed" column)
│
├── [Step 2] SelectKBest ───────────────────────── MATERIALIZATION POINT 2
│            (KNOWN schema: k output columns)
│            Decodes KV input, selects top k features
│
└── [Step 3] RandomForestClassifier
             (PREDICTOR - always separate Step)
             Reads struct input, produces predictions
```

DATA FLOW
=========

```
┌──────────────────┐
│   Input Data     │
│ (7 features +    │
│  1 target)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│  ColumnTransformer.fit_transform()                               │
│  ════════════════════════════════                                │
│  sklearn handles internally:                                     │
│  • Routes columns to children                                    │
│  • Fits each child transformer                                   │
│  • Concatenates all outputs                                      │
│                                                                  │
│  xorq wraps output as: Array[Struct{key, value}]                │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼ MATERIALIZATION (into_backend)
┌──────────────────────────────────────────────────────────────────┐
│  SelectKBest.fit_transform()                                     │
│  ═══════════════════════════                                     │
│  • Decodes KV input to DataFrame                                 │
│  • Fits selector (needs target)                                  │
│  • Selects top k features                                        │
│                                                                  │
│  xorq outputs as: Struct{transformed_0, ..., transformed_9}     │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼ MATERIALIZATION (into_backend)
┌──────────────────────────────────────────────────────────────────┐
│  RandomForestClassifier.fit() / .predict()                       │
│  ═════════════════════════════════════════                       │
│  • Reads struct columns as features                              │
│  • Fits classifier on training data                              │
│  • Produces predictions                                          │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│   Predictions    │
└──────────────────┘
```
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

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
        "age": np.random.randint(18, 80, n_samples).astype(float),
        "income": np.random.randint(20000, 150000, n_samples).astype(float),
        "credit_score": np.random.randint(300, 850, n_samples).astype(float),
        "years_employed": np.random.randint(0, 40, n_samples).astype(float),
        "education": np.random.choice(
            ["high_school", "bachelor", "master", "phd"], n_samples
        ),
        "employment_type": np.random.choice(
            ["full_time", "part_time", "contract", "self_employed"], n_samples
        ),
        "region": np.random.choice(["north", "south", "east", "west"], n_samples),
        "approved": np.random.randint(0, 2, n_samples),
    }
)

# Add missing values
data.loc[np.random.choice(n_samples, 20), "age"] = np.nan
data.loc[np.random.choice(n_samples, 15), "income"] = np.nan

expr = xo.memtable(data)

numeric_features = ["age", "income", "credit_score", "years_employed"]
categorical_features = ["education", "employment_type", "region"]
all_features = tuple(numeric_features + categorical_features)

# =============================================================================
# Build the deeply nested sklearn pipeline
# =============================================================================
print("=" * 70)
print("BUILDING DEEPLY NESTED PIPELINE")
print("=" * 70)

# Depth 4: Pipelines inside FeatureUnion
scaled_pipeline = SklearnPipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

poly_pipeline = SklearnPipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ]
)

# Depth 3: FeatureUnion
numeric_union = FeatureUnion(
    [
        ("scaled", scaled_pipeline),
        ("poly", poly_pipeline),
    ]
)

# Depth 3: Categorical pipeline
categorical_pipeline = SklearnPipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

# Depth 2: ColumnTransformer
preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_union, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

# Depth 1: Top-level pipeline
sklearn_pipe = SklearnPipeline(
    [
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(f_classif, k=10)),
        ("classifier", RandomForestClassifier(n_estimators=50, random_state=42)),
    ]
)

# =============================================================================
# Verify Structer types
# =============================================================================
print("\n" + "=" * 70)
print("STRUCTER TYPES")
print("=" * 70)

print("\nIndividual transformer types:")
transformers = [
    ("SimpleImputer", SimpleImputer(), tuple(numeric_features)),
    ("StandardScaler", StandardScaler(), tuple(numeric_features)),
    ("PolynomialFeatures", PolynomialFeatures(), tuple(numeric_features)),
    ("OneHotEncoder", OneHotEncoder(), tuple(categorical_features)),
    ("SelectKBest(k=10)", SelectKBest(k=10), tuple(numeric_features)),
]
for name, inst, feats in transformers:
    print(f"  {name}: {check_structer_type(inst, expr, feats)}")

print("\nContainer types (inherit from children):")
print("  ColumnTransformer (has OneHotEncoder): KV-ENCODED")
print("  FeatureUnion (has PolynomialFeatures): KV-ENCODED")

# =============================================================================
# Convert to xorq Pipeline
# =============================================================================
print("\n" + "=" * 70)
print("XORQ PIPELINE CONVERSION")
print("=" * 70)

xorq_pipeline = Pipeline.from_instance(sklearn_pipe)

print(f"\nTop-level Steps: {len(xorq_pipeline.steps)}")
for i, step in enumerate(xorq_pipeline.steps):
    print(f"\n  Step {i + 1}: {step.name} ({step.typ.__name__})")

# =============================================================================
# Fit the pipeline
# =============================================================================
print("\n" + "=" * 70)
print("FITTING PIPELINE")
print("=" * 70)

print("\nMaterialization points during fit:")
print("  1. After ColumnTransformer -> into_backend()")
print("  2. After SelectKBest -> into_backend()")
print("  (Predictor does not materialize transform output)\n")

fitted_pipeline = xorq_pipeline.fit(expr, features=all_features, target="approved")
print("Pipeline fitted successfully!")

# =============================================================================
# KV-ENCODED vs KNOWN: Ibis Expression Schema
# =============================================================================
print("\n" + "=" * 70)
print("KV-ENCODED vs KNOWN: IBIS EXPRESSION SCHEMA")
print("=" * 70)

print("""
The Ibis expression reveals the schema difference:
  - KV-ENCODED: Single 'transformed' column (Array[Struct{key, value}])
  - KNOWN: Multiple named columns (Struct{col1, col2, ...})
""")

# Get the prediction expression (NOT executed yet)
predictions_expr = fitted_pipeline.predict(expr)

print(f"predictions_expr type: {type(predictions_expr).__name__}")
print("\nFinal prediction expression columns:")
print(f"  {predictions_expr.columns}")

# Show transform expressions at each step
print("\n--- Transform Step Schemas ---")

# Step 1: ColumnTransformer (KV-ENCODED because contains OneHotEncoder)
ct_step = fitted_pipeline.transform_steps[0]
ct_transform_expr = ct_step.transform(expr)
print("\nStep 1: ColumnTransformer (KV-ENCODED)")
print(f"  Type: {type(ct_transform_expr).__name__}")
print(f"  Columns: {ct_transform_expr.columns}")
print("  NOTE: Single 'transformed' column = KV-encoded array")
print("        Column names resolved at RUNTIME via get_feature_names_out()")

# Step 2: SelectKBest (KNOWN - k output columns)
selector_step = fitted_pipeline.transform_steps[1]
# SelectKBest receives KV-encoded input, decodes it, outputs KNOWN schema
selector_transform_expr = selector_step.transform(ct_transform_expr)
print("\nStep 2: SelectKBest(k=10) (KNOWN)")
print(f"  Type: {type(selector_transform_expr).__name__}")
print(f"  Columns: {selector_transform_expr.columns}")
print(
    f"  NOTE: {len(selector_transform_expr.columns)} columns known at BUILD TIME from k param"
)

print("""
CONTRAST WITH non_kv_deeply_nested_pipeline.py:
  - That example uses ONLY known-schema transformers
  - ColumnTransformer output has named columns (not KV-encoded)
  - All column names predictable without running transform
""")

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

1. MATERIALIZATION
   - Occurs at TOP-LEVEL Step boundaries only
   - ColumnTransformer children are NOT materialized separately
   - sklearn handles all internal transforms as one unit

2. KV-ENCODED CONTAINERS
   - ColumnTransformer contains OneHotEncoder -> KV-ENCODED
   - Output is Array[Struct{key, value}]
   - Next step (SelectKBest) decodes automatically

3. REGISTERED TYPES
   - All containers (ColumnTransformer, FeatureUnion, Pipeline) are registered
   - No allow_unregistered needed for this example
"""
)
