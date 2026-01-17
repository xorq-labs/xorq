# ML Task Planning Phase

**Before building any ML pipeline, complete this planning checklist.**

## Pre-Build Checklist

### 1. Check for Similar Examples

**Search the project for existing ML examples:**
```bash
find examples/ -name "*prediction*" -o -name "*ml*" -o -name "*model*"
```

**If found:** Read the example and adapt the pattern instead of building from scratch.

**Common examples:**
- `examples/diamonds_price_prediction.py` - Regression with sklearn
- Other examples in `examples/` directory

### 2. Select Template (if no example found)

**List available templates:**
```bash
xorq agent templates list
```

**For ML tasks, use:**
- `sklearn_pipeline` - For scikit-learn models (regression, classification)
- `cached_fetcher` - If you need to hydrate and cache data first

**Scaffold the template:**
```bash
xorq agent templates scaffold sklearn_pipeline
```

### 3. Define Your Feature Set

**Document your features BEFORE coding:**

**Numeric features** (go in `feature_columns`):
- List all numeric columns: `carat`, `depth`, `table`, `x`, `y`, `z`
- List engineered features: `log_carat`, `volume`, `surface_area`
- List encoded categoricals: `cut_score`, `color_score`, `clarity_score`

**Categorical features** (for reference/output, NOT in `feature_columns`):
- List raw categorical columns: `cut`, `color`, `clarity`
- These stay in the table but DON'T go in `feature_columns`

**Critical Rule:**
> `feature_columns` should contain ONLY numeric features used for training.
> Raw categoricals should NOT be in `feature_columns` but can stay in the table.

### 4. Define Your Target

**What are you predicting?**
- Column name: `price`, `survived`, `category`, etc.
- Type: continuous (regression) or categorical (classification)?
- Data type: Can be int, float, or string (xorq handles conversion)

### 5. Training Strategy

**Choose your approach:**

**Option A: Train/Test Split** (recommended for evaluation)
```python
from xorq.expr.ml import train_test_splits

train, test = train_test_splits(feature_view, test_fraction=0.2)
fitted = pipeline.fit(train, features=feature_columns, target=TARGET)

# Predict on test set
test_predictions = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted.predict)
    .drop(TARGET)
    .unpack(ORIGINAL_ROW)
    .mutate(predicted=_.predicted)
    .drop("predicted")
)
```

**Option B: Full Dataset** (simpler, for production models)
```python
# Train on everything
fitted = pipeline.fit(full_data, features=feature_columns, target=TARGET)

# Predict on same data
predictions = (
    full_data
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted.predict)
    .drop(TARGET)
    .unpack(ORIGINAL_ROW)
    .mutate(predicted=_.predicted)
    .drop("predicted")
)
```

**Trade-offs:**
- Train/test: Can evaluate model performance, more complex
- Full dataset: Simpler code, no evaluation, better for deployment

### 6. Output Columns

**What columns do you need in the final result?**

**Must have:**
- Prediction column (renamed from `predicted`)
- Original features (for analysis)

**Nice to have:**
- Categorical columns (for segmentation)
- Error metrics (`abs_error`, `pct_error`)
- Confidence scores
- Row identifiers

**Remember:** The struct pattern preserves ALL columns, so don't worry about losing them.

## Documentation Template

**Fill this out before coding:**

```markdown
# ML Pipeline: <Name>

## Task Type
- [ ] Regression
- [ ] Classification
- [ ] Clustering

## Data Source
- Table: `<table_name>`
- Schema: `<uppercase/lowercase>`
- Row count: ~<estimate>

## Features

### Numeric (in feature_columns):
1. `carat` - Diamond weight
2. `cut_score` - Encoded cut quality (0-4)
3. `color_score` - Encoded color (0-6)
4. `clarity_score` - Encoded clarity (0-7)
5. `log_carat` - Log-transformed carat
6. `volume` - x * y * z

### Categorical (NOT in feature_columns):
1. `cut` - Raw cut value (Fair, Good, Premium, Ideal)
2. `color` - Raw color (D-J)
3. `clarity` - Raw clarity (I1-IF)

## Target
- Column: `price`
- Type: continuous (regression)

## Training Strategy
- [x] Full dataset (no split)
- [ ] Train/test split (80/20)

## Output Columns
- `predicted_price` - Model prediction
- `deal_score` - Custom metric
- `price_category` - Under/over/fair priced
- All original features
- All categorical columns

## Reference
- Template: sklearn_pipeline
- Example: examples/diamonds_price_prediction.py
```

## Critical Reminders

Before you start coding, remember:

1. **Schema First**: Run `print(table.schema())` and note column case
2. **Struct Pattern**: Use the standard struct pattern for predictions
3. **Float64**: Cast all sklearn features to float64
4. **No Pandas**: Everything must be deferred ibis expressions
5. **Check Examples**: Adapt existing code instead of reinventing

## Next Steps

Once planning is complete:

1. **Create script** with documented structure
2. **Import required modules**:
   ```python
   import xorq.api as xo
   from xorq.api import _
   from xorq.vendor import ibis
   from xorq.expr.ml.pipeline_lib import Pipeline
   from sklearn.pipeline import Pipeline as SkPipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LinearRegression  # or your model
   import toolz
   ```
3. **Build incrementally**: Test each section before proceeding
4. **Catalog early**: Register builds as you go for debugging

## Common Pitfalls to Avoid

- ❌ Starting without checking examples
- ❌ Not documenting features upfront
- ❌ Including categoricals in feature_columns
- ❌ Using int8/int16 for sklearn features
- ❌ Trying to predict without struct pattern
- ❌ Not checking schema before building
- ❌ Creating new files instead of using inline code

**Planning takes 5 minutes. Debugging takes 30 minutes. Plan first!**
