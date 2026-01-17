# ML Prediction Struct Pattern

**THIS IS THE ONLY SUPPORTED PATTERN FOR ML PREDICTIONS IN XORQ.**

## Why Use This Pattern?

1. **Preserves all table columns** through the prediction pipeline
2. **Avoids manual joins** (which often fail with deferred execution)
3. **Works seamlessly** with xorq's deferred execution model
4. **Prevents duplicate column errors**

## The Pattern

```python
from xorq.api import _
import toolz

@toolz.curry
def as_struct(expr, name=None):
    """Pack all columns into a struct."""
    struct = xo.struct({column: expr[column] for column in expr.columns})
    if name:
        struct = struct.name(name)
    return struct

# Apply the pattern
predictions = (
    table
    .mutate(as_struct(name="original_row"))  # 1. Pack ENTIRE table into struct
    .pipe(fitted_pipeline.predict)            # 2. Predict (adds 'predicted' column)
    .drop("target_column")                    # 3. Remove target column if present
    .unpack("original_row")                   # 4. Unpack ALL original columns
    .mutate(predicted_price=_.predicted)      # 5. Rename prediction column
    .drop("predicted")                         # 6. Clean up temporary column
)
```

## What Columns End Up in the Result?

- **All original columns** from the input table
- **Renamed prediction**: `predicted_price` (or your chosen name)
- **Any new columns** added in the final `.mutate()`

## Step-by-Step Breakdown

### Step 1: Pack Into Struct
```python
.mutate(as_struct(name="original_row"))
```
Creates a struct column containing ALL table columns. This preserves everything while allowing the pipeline to add new columns.

### Step 2: Predict
```python
.pipe(fitted_pipeline.predict)
```
Runs prediction. The xorq Pipeline adds a column named `predicted` with the model output.

### Step 3: Drop Target (if needed)
```python
.drop("target_column")
```
Remove the target column before unpacking (it may have been duplicated).

### Step 4: Unpack Original Columns
```python
.unpack("original_row")
```
Extracts all columns from the struct back into regular table columns.

### Step 5: Rename Prediction
```python
.mutate(predicted_price=_.predicted)
```
Give the prediction column a meaningful name.

### Step 6: Clean Up
```python
.drop("predicted")
```
Remove the temporary `predicted` column.

## Common Mistakes

### ❌ WRONG: Trying to Select Specific Columns Before Struct
```python
# This breaks - loses columns you didn't select!
predictions = (
    table.select("feature1", "feature2", "price")  # ❌ Other columns lost!
    .mutate(as_struct(name="original_row"))
    .pipe(fitted_pipeline.predict)
    .unpack("original_row")
)
```

### ❌ WRONG: Manually Joining Predictions Back
```python
# This is complicated and error-prone!
predictions_only = fitted.predict(table[features])  # ❌ Breaks deferred execution
result = table.join(predictions_only, ...)          # ❌ Risky with deferred tables
```

### ❌ WRONG: Including Target in feature_columns
```python
feature_columns = ["carat", "cut", "price"]  # ❌ 'price' is target, not feature!
```

### ✅ RIGHT: Full Pattern
```python
# Keep all columns in table, specify only features for training
feature_columns = ["carat", "cut", "color", "clarity"]  # Numeric features only
full_table = table  # Has features + target + any categorical columns

fitted = pipeline.fit(full_table, features=feature_columns, target="price")

predictions = (
    full_table
    .mutate(as_struct(name="original_row"))
    .pipe(fitted.predict)
    .drop("price")  # Remove target
    .unpack("original_row")
    .mutate(predicted_price=_.predicted)
    .drop("predicted")
)
```

## Categorical Columns

**Q: What about categorical columns I want in the output?**

**A:** Keep them in the table throughout! They flow through the struct/unpack automatically.

```python
# Table has: numeric features + categorical columns + target
table = engineer_features(raw_data)  # Returns: carat, cut_score, cut, color, clarity, price

# Features for training (numeric only)
feature_columns = ["carat", "cut_score", "color_score", "clarity_score"]

# The categorical columns (cut, color, clarity) stay in the table
# They're packed into the struct and unpacked automatically
# They DON'T go in feature_columns

fitted = pipeline.fit(table, features=feature_columns, target="price")

predictions = (
    table
    .mutate(as_struct(name="original_row"))  # Packs carat, cut_score, cut, color, clarity, price
    .pipe(fitted.predict)
    .drop("price")
    .unpack("original_row")                  # Unpacks ALL of them back
    .mutate(predicted_price=_.predicted)
    .drop("predicted")
)
# Result has: carat, cut_score, cut, color, clarity, predicted_price
```

## Reference Implementation

See working example: `examples/diamonds_price_prediction.py:172-185`

## Summary Checklist

When building ML pipelines with xorq:

- [ ] Create `as_struct` helper function
- [ ] Pack ENTIRE table with `.mutate(as_struct(name="original_row"))`
- [ ] Use `.pipe(fitted.predict)` for prediction
- [ ] Drop target column if present
- [ ] Unpack with `.unpack("original_row")`
- [ ] Rename prediction column
- [ ] Drop temporary `predicted` column
- [ ] Keep categorical columns in table (don't put in feature_columns)

**If you deviate from this pattern, you WILL get errors.**
