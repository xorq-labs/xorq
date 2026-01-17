# ML Complete Working Examples

## COMPLETE WORKING EXAMPLE

```python
import xorq.api as xo
from xorq.expr.ml.pipeline_lib import Pipeline
from xorq.caching import ParquetCache
import xorq.expr.ml as ml
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score
import toolz

# Constants for struct pattern
ORIGINAL_ROW = "original_row"
PREDICTED = "predicted"

@toolz.curry
def as_struct(expr, name=None):
    struct = xo.struct({c: expr[c] for c in expr.columns})
    if name:
        struct = struct.name(name)
    return struct

# 1. Load and engineer features (deferred)
data = xo.deferred_read_parquet(
    con=xo.duckdb.connect(),
    path="https://storage.googleapis.com/letsql-pins/penguins/20250703T145709Z-c3cde/penguins.parquet",
    table_name="penguins"
)

# ✅ Feature engineering (ENCOURAGED!)
engineered_data = data.mutate(
    # Log transforms
    log_bill_length=_.bill_length_mm.ln(),
    log_bill_depth=_.bill_depth_mm.ln(),

    # Powers
    bill_length_squared=_.bill_length_mm ** 2,

    # Interactions
    bill_ratio=_.bill_length_mm / _.bill_depth_mm,
)

# Setup
features = ("log_bill_length", "log_bill_depth", "bill_length_squared", "bill_ratio")
target = "species"

# Select and clean
clean_data = engineered_data.select(features + (target,)).drop_null()

# 2. Split and cache
train, test = [
    expr.cache(ParquetCache.from_kwargs())
    for expr in xo.train_test_splits(clean_data, test_sizes=0.2, random_seed=42)
]

# 3. Create and wrap pipeline
sklearn_pipeline = SkPipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(C=1E-4))
])
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

# 4. Fit pipeline
fitted_pipeline = xorq_pipeline.fit(train, features=features, target=target)

# 5. Predictions (using struct pattern - ALWAYS use this!)
test_with_predictions = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted_pipeline.predict)
    .drop(target)
    .unpack(ORIGINAL_ROW)
    .cache(ParquetCache.from_kwargs())
)

# 6. Evaluate using the ONE function for ALL metrics!
deferred_model = fitted_pipeline.predict_step.deferred_model  # Key pattern!

metrics = {
    'accuracy': ml.deferred_sklearn_metric(
        test, target, features,
        deferred_model=deferred_model,
        metric_fn=accuracy_score
    ),
    'precision_macro': ml.deferred_sklearn_metric(
        test, target, features,
        deferred_model=deferred_model,
        metric_fn=precision_score,
        metric_kwargs={'average': 'macro', 'zero_division': 0}
    ),
    'f1_weighted': ml.deferred_sklearn_metric(
        test, target, features,
        deferred_model=deferred_model,
        metric_fn=f1_score,
        metric_kwargs={'average': 'weighted', 'zero_division': 0}
    )
}

# 7. Execute everything (computation happens here!)
results = {name: metric.execute() for name, metric in metrics.items()}
final_predictions = test_with_predictions.execute()

print(f"Model Performance: {results}")
print(f"Predictions shape: {final_predictions.shape}")
```

---

## Struct Pattern Examples

### When to Use the Struct Pattern (ALWAYS for ML Predictions!)

#### ✅ ALWAYS Use Struct Pattern When:

**1. Making predictions on a test/validation set**
```python
# You want ALL columns (IDs, metadata, etc.) alongside predictions
test_with_predictions = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted_pipeline.predict)
    .unpack(ORIGINAL_ROW)
)
```

**2. Applying trained models to new data**
```python
# Scoring new data while keeping original columns
scored_data = (
    new_data
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted_pipeline.predict)
    .unpack(ORIGINAL_ROW)
)
```

**3. Working with data that has non-feature columns you need to keep**
```python
# Example: Keep customer_id, transaction_date while predicting fraud
predictions = (
    transactions
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fraud_model.predict)
    .unpack(ORIGINAL_ROW)
)
# Result has: customer_id, transaction_date, amount, ..., predicted_fraud
```

### Why the Struct Pattern is Superior

**Problem with old approach:**
```python
# ❌ OLD WAY: Direct predict loses columns
test_predictions = fitted_pipeline.predict(test[features])
# Result: ONLY predictions, all other columns lost!

# ❌ OLD WAY: Mutate risks column conflicts
test_with_pred = test.mutate(
    predictions=fitted_pipeline.predict(test[features])
)
# Problems:
# - Risk of column name conflicts
# - Relation integrity issues
# - Can't easily compose with other operations
```

**Solution with struct pattern:**
```python
# ✅ NEW WAY: Struct pattern preserves everything
test_predictions = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))  # 1. Store all columns as struct
    .pipe(fitted_pipeline.predict)          # 2. Pipeline adds predictions
    .drop(target)                           # 3. Optional: remove target
    .unpack(ORIGINAL_ROW)                   # 4. Restore original columns
)
# Result: ALL original columns + predictions, no conflicts!
```

### How the Struct Pattern Works

```python
# Step-by-step breakdown:

# 1. Original test table has: [id, feature1, feature2, target, metadata]
test = ...

# 2. Store ALL columns as a single struct column
test_with_struct = test.mutate(as_struct(name=ORIGINAL_ROW))
# Now has: [id, feature1, feature2, target, metadata, original_row (struct)]

# 3. Pipeline predict operates on the table
# The pipeline internally knows to use only 'features' columns
test_predicted = test_with_struct.pipe(fitted_pipeline.predict)
# Now has: [predicted, original_row (struct)]

# 4. Drop target if present (optional)
test_predicted = test_predicted.drop(target)
# Now has: [predicted, original_row (struct)]

# 5. Unpack the struct to restore original columns
test_final = test_predicted.unpack(ORIGINAL_ROW)
# Now has: [id, feature1, feature2, metadata, predicted]
# Perfect! All original columns + predictions
```

### Real-World Example

```python
# Scenario: Predict house prices, keep address and listing_id

# Your data has:
# - listing_id (need to keep for joining)
# - address (need for display)
# - sqft, bedrooms, bathrooms (features)
# - price (target, only in training)
# - photos_url (need for web display)

# Training
train, test = xo.train_test_splits(data, test_sizes=0.2)
features = ('sqft', 'bedrooms', 'bathrooms')
target = 'price'

fitted = pipeline.fit(train, features=features, target=target)

# Predictions with struct pattern
predictions = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted.predict)
    .drop(target)  # Remove actual price from test set
    .unpack(ORIGINAL_ROW)
)

# Result has ALL columns:
# listing_id, address, sqft, bedrooms, bathrooms, photos_url, predicted_price
# Ready to display in web app or join with other data!
```

### Common Mistakes Avoided

```python
# ❌ MISTAKE 1: Losing identifier columns
predictions = fitted.predict(test[features])
# Lost: listing_id, address, photos_url - can't join or display!

# ❌ MISTAKE 2: Column name conflicts in mutate
test.mutate(predictions=fitted.predict(test[features]))
# If 'predictions' already exists, conflict!
# If target is in both, confusion!

# ❌ MISTAKE 3: Relation mixing
test.mutate(predictions=fitted.predict(test.select(features)))
# Error: Cannot add Field, they belong to another relation

# ✅ SOLUTION: Struct pattern handles all of these elegantly
```

---
*Remember: The power of XorQ is deferred execution. Build your entire pipeline first, execute once at the end!*
