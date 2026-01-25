"""
Example demonstrating the as_struct pattern for preserving columns during ML predictions.

This pattern is useful when you want to:
1. Make predictions with an ML model
2. Keep all original columns alongside predictions for downstream analysis
3. Avoid manually dropping/selecting columns

The pattern involves 3 steps:
1. Bundle all columns into a struct before prediction
2. Make predictions (features + struct are passed through)
3. Unpack the struct to restore original columns

Why this matters:
- Without as_struct: Only feature columns survive prediction
- With as_struct: ALL original columns preserved (IDs, timestamps, metadata, labels)
- Enables: Comparison, analysis, joining without manual reconstruction
"""

import toolz
import xorq.api as xo
from xorq.expr.ml.pipeline_lib import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler


# Create as_struct helper
@toolz.curry
def as_struct(expr, name=None):
    """Convert all columns into a struct column.

    Args:
        expr: Expression to convert
        name: Optional name for the struct column

    Returns:
        A struct containing all columns from expr
    """
    struct = xo.struct({c: expr[c] for c in expr.columns})
    return struct.name(name) if name else struct


# Load penguins dataset
penguins = xo.examples.penguins.fetch(backend=xo.connect())

# Prepare data: drop nulls and create train/test split
clean_data = penguins.dropna()

# Feature columns
features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

# Simple train/test split by island (for demonstration)
train = clean_data.filter(xo._.island != "Biscoe")
test = clean_data.filter(xo._.island == "Biscoe")

# Build and fit pipeline
sklearn_pipeline = SkPipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

fitted_pipeline = xorq_pipeline.fit(
    train,
    features=features,
    target="species"
)


# ============================================================================
# APPROACH 1: Standard prediction (loses non-feature columns)
# ============================================================================
print("\n" + "="*70)
print("APPROACH 1: Standard Prediction")
print("="*70)

test_predicted_simple = fitted_pipeline.predict(test)
print("\nColumns available after standard prediction:")
print(test_predicted_simple.columns)
print("\nResult (limited columns):")
print(test_predicted_simple.limit(5).execute())


# ============================================================================
# APPROACH 2: Using as_struct pattern (preserves ALL columns)
# ============================================================================
print("\n" + "="*70)
print("APPROACH 2: as_struct Pattern")
print("="*70)

# Step 1: Bundle all columns into a struct
test_with_struct = test.mutate(as_struct(name="original_data"))
print("\nStep 1 - After adding struct column:")
print(test_with_struct.columns)

# Step 2: Make predictions (pass features + struct)
test_predicted_with_struct = fitted_pipeline.predict(test_with_struct)
print("\nStep 2 - After prediction:")
print(test_predicted_with_struct.columns)

# Step 3: Unpack struct and organize columns
test_predicted_full = (
    test_predicted_with_struct
    .select("original_data", "predicted")  # Keep only struct and prediction
    .unpack("original_data")  # Restore all original columns
)
print("\nStep 3 - After unpacking struct:")
print(test_predicted_full.columns)

print("\nResult (ALL columns preserved):")
print("Available columns:", test_predicted_full.columns)
result = test_predicted_full.limit(5).execute()
print(result)


# ============================================================================
# USE CASE: Compare predictions to actual labels with context
# ============================================================================
print("\n" + "="*70)
print("USE CASE: Comparing Predictions with Full Context")
print("="*70)

comparison = (
    test_predicted_full
    .mutate(correct=xo._.species == xo._.predicted)
    .select(
        "species",
        "predicted",
        "correct",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
    )
    .limit(10)
)
print("\nPredictions with context:")
print(comparison.execute())

accuracy = (
    test_predicted_full
    .mutate(correct=(xo._.species == xo._.predicted).cast("int"))
    .aggregate(accuracy=xo._.correct.mean())
)
print("\nModel accuracy:")
print(accuracy.execute())


# ============================================================================
# ADVANCED: Selective unpacking
# ============================================================================
print("\n" + "="*70)
print("ADVANCED: Selective Column Restoration")
print("="*70)

# You can also selectively unpack only specific columns from the struct
selective_unpack = (
    test_predicted_with_struct
    .select(
        xo._.original_data["species"].name("species"),
        xo._.original_data["island"].name("island"),
        "predicted"
    )
)
print("\nSelective unpack (species, island, predicted only):")
print(selective_unpack.limit(5).execute())


print("\n" + "="*70)
print("Summary")
print("="*70)
print("""
The as_struct pattern is essential when you need to:
✓ Preserve non-feature columns (e.g., IDs, timestamps, metadata)
✓ Compare predictions with original labels
✓ Perform downstream analysis with full context
✓ Avoid manually tracking which columns to keep

Without as_struct:
- Only get predictions for feature columns
- Lose valuable context (species, island, other measurements)
- Must manually reconstruct full dataset by joining

With as_struct:
- Bundle everything into a struct before prediction
- Pass struct through prediction step
- Unpack to restore ALL original columns + predictions
- Clean, maintainable pattern

Pattern template:
    1. test_with_struct = test.mutate(as_struct(name="original_data"))
    2. predicted = fitted_pipeline.predict(test_with_struct)
    3. result = predicted.select("original_data", "predicted").unpack("original_data")
""")


# Export for use in other examples
expr = test_predicted_full

if __name__ == "__main__":
    print("\n✓ Example completed successfully!")
