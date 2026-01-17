# sklearn Type Requirements in xorq

**CRITICAL:** sklearn models in xorq require `float64` for all numeric features.

## The Rule

When encoding categorical variables or creating numeric features for sklearn models:
- **ALWAYS cast to float64**
- **Use 0.0 instead of 0** in literals
- **Never use int8, int16, int32, or int64**

## Why?

xorq's sklearn integration uses DataFusion UDFs that perform strict type checking. When you fit a model with int64 features, but the UDF signature expects float64, you'll get:

```
ValueError: Failed to coerce arguments to satisfy a call to 'dumps_of_inner_fit_0' function:
coercion from Float64, Float64, Float64, Int64 to the signature
Exact([Float64, Float64, Float64, Float64]) failed
```

This error means: "I expected all float64, but you gave me an int64."

## Categorical Encoding Pattern

### ❌ WRONG:
```python
cut_score = (
    _.cut.case()
    .when("Fair", 0)      # ❌ int literal
    .when("Good", 1)
    .when("Premium", 3)
    .else_(2)
    .end()
    .cast("int8")         # ❌ int8 will cause errors!
)
```

### ✅ RIGHT:
```python
cut_score = (
    _.cut.case()
    .when("Fair", 0.0)    # ✅ float literal
    .when("Good", 1.0)
    .when("Premium", 3.0)
    .else_(2.0)
    .end()
    .cast("float64")      # ✅ float64 required!
)
```

### ✅ ALSO RIGHT (explicit cast):
```python
cut_score = (
    _.cut.case()
    .when("Fair", 0)
    .when("Good", 1)
    .when("Premium", 3)
    .else_(2)
    .end()
    .cast("float64")      # ✅ Cast at the end
)
```

## Complete Example

```python
def encode_quality_columns(table):
    """Map categorical columns to numeric scores (float64)."""
    return table.mutate(
        cut_score=(
            _.cut.case()
            .when("Fair", 0.0)
            .when("Good", 1.0)
            .when("Very Good", 2.0)
            .when("Premium", 3.0)
            .when("Ideal", 4.0)
            .else_(2.0)
            .end()
            .cast("float64")  # REQUIRED
        ),
        color_score=(
            _.color.case()
            .when("J", 0.0)
            .when("I", 1.0)
            .when("H", 2.0)
            .when("G", 3.0)
            .when("F", 4.0)
            .when("E", 5.0)
            .when("D", 6.0)
            .else_(3.0)
            .end()
            .cast("float64")  # REQUIRED
        ),
        clarity_score=(
            _.clarity.case()
            .when("I1", 0.0)
            .when("SI2", 1.0)
            .when("SI1", 2.0)
            .when("VS2", 3.0)
            .when("VS1", 4.0)
            .when("VVS2", 5.0)
            .when("VVS1", 6.0)
            .when("IF", 7.0)
            .else_(3.0)
            .end()
            .cast("float64")  # REQUIRED
        ),
    )
```

## What About Target Columns?

The **target column** can be any numeric type (int, float, etc.). xorq handles targets separately from features:

```python
# This is fine - target doesn't need to be float64
feature_view = table.select(
    "carat",           # float64 ✅
    "cut_score",       # float64 ✅
    "price",           # int64 is OK for target ✅
)

fitted = pipeline.fit(feature_view, features=["carat", "cut_score"], target="price")
# xorq handles the target type conversion internally
```

## Engineered Features

All engineered numeric features should also be float64:

```python
engineered = table.mutate(
    log_carat=(_.carat + 1e-6).ln(),              # Already float64 ✅
    carat_squared=_.carat**2,                     # Already float64 ✅
    volume=_.x * _.y * _.z,                       # Already float64 ✅
    # If you create from integers, cast:
    count_features=_.some_int_column.cast("float64"),
)
```

## Performance Note

**Q:** "Won't float64 use more memory than int8?"

**A:** Yes, but:
1. The performance difference is negligible for typical ML datasets
2. xorq uses Arrow/Parquet which compresses efficiently
3. The alternative is build errors and debugging time
4. **Correctness > premature optimization**

## Summary Checklist

For sklearn models in xorq:

- [ ] All categorical encodings cast to `float64`
- [ ] Use `0.0`, `1.0`, `2.0` instead of `0`, `1`, `2` in case statements
- [ ] Explicitly `.cast("float64")` after case statements
- [ ] All numeric features in `feature_columns` are float64
- [ ] Target column can remain any numeric type
- [ ] Engineered features are float64 (usually automatic)

**If you see type coercion errors, check that ALL features are float64!**

## Reference

See working example: `examples/diamonds_price_prediction.py:44-82`
