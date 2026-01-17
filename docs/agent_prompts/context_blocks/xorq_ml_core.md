# Xorq Machine Learning Core Guide

## 🔥 CRITICAL: ALWAYS USE STRUCT PATTERN FOR PREDICTIONS! 🔥

**Every prediction MUST use the struct pattern. No exceptions!**

```python
test_predicted = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted_pipeline.predict)
    .drop(target)
    .unpack(ORIGINAL_ROW)
)

# ❌ NEVER use these old patterns:
fitted_pipeline.predict(test[features])  # Loses columns!
test.mutate(pred=fitted_pipeline.predict(test[features]))  # Conflicts!
```

**Why this matters:**
- Keeps ALL columns (IDs, metadata, etc.)
- Prevents relation conflicts and errors
- Makes code composable and maintainable

---

## ⚠️ REQUIRED IMPORTS - ALWAYS USE THESE!

```python
# Core imports (ALWAYS needed)
import xorq.api as xo
from xorq.api import _
from xorq.vendor import ibis  # ✅ CORRECT way to import ibis
from xorq.caching import ParquetCache
from xorq.expr.ml.pipeline_lib import Pipeline
import xorq.expr.ml as ml

# Common sklearn imports
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score

# For struct pattern
import toolz
```

**❌ COMMON IMPORT MISTAKES TO AVOID:**
```python
# WRONG: These will cause AttributeError
import ibis  # Use: from xorq.vendor import ibis
import xorq.vendor.ibis  # Use: from xorq.vendor import ibis
xo.vendor.ibis.desc('count')  # Use: ibis.desc('count') after importing
```

## ⚠️ CRITICAL LIMITATIONS - CHECK THESE FIRST!

### Sklearn Transformer Support (Based on xorq/expr/ml/structer.py):
```python
# ✅ OFFICIALLY SUPPORTED (registered in structer):
# - StandardScaler (confirmed in examples)
# - SimpleImputer
# - SelectKBest

# ⚠️ REQUIRES CUSTOM WRAPPER (see bank_marketing.py example):
# - OneHotEncoder (needs custom Step class with get_step_f_kwargs)

# ❌ NOT SUPPORTED (will raise "can't handle type" error):
# - ColumnTransformer (no implementation, see pipeline_lib.py:204)
# - MinMaxScaler (not registered)
# - RobustScaler (not registered)
# - LabelEncoder (not registered)
# - Any other transformer not explicitly registered

# ✅ MODELS THAT WORK WITH Pipeline.from_instance():
# - LogisticRegression, LinearRegression (confirmed)
# - KNeighborsClassifier, DecisionTreeClassifier (in examples)
# - RandomForestClassifier, SVC, GaussianNB (in sklearn_classifier_comparison.py)
```

### The Core Problem: Structer Registration
```python
# XorQ uses a registration system (structer_from_instance)
# Only registered transformers can be used with Pipeline.from_instance()
# Unregistered transformers raise: ValueError: can't handle type <class>
```

## ⚠️ CRITICAL RULES - NEVER VIOLATE THESE

### Rule #1: ALWAYS Use Deferred Execution
```python
# ❌ WRONG: Early execution
df = table.execute()  # NO! Too early!
X_train, X_test = train_test_split(df)  # NO! Use xorq

# ✅ CORRECT: Stay deferred
train, test = xo.train_test_splits(table, test_sizes=0.2)
```

### Rule #2: ALWAYS Wrap sklearn with xorq
```python
# ❌ WRONG: Bare sklearn
model.fit(X_train, y_train)  # NO!

# ✅ CORRECT: Wrapped with Pipeline.from_instance
from xorq.expr.ml.pipeline_lib import Pipeline
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
fitted = xorq_pipeline.fit(train, features=features, target=target)
```

### Rule #3: ALWAYS Use Struct Pattern for Predictions (Critical!)
```python
# ❌ WRONG: Direct prediction loses non-feature columns
predictions = fitted_pipeline.predict(test[features])  # Only returns predictions!
test_with_predictions = test.mutate(predictions=fitted_pipeline.predict(test[features]))  # Column conflicts possible

# ✅ CORRECT: Use struct pattern to preserve all columns
# Option A: Standalone predictions with struct (RECOMMENDED)
test_predicted = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted_pipeline.predict)
    .drop(target)  # Optional: drop target if present
    .unpack(ORIGINAL_ROW)
)

# Option B: Predictions with explicit column name
test_with_predictions = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))
    .select(
        fitted_pipeline.predict(test).name(PREDICTED),
        ORIGINAL_ROW
    )
    .unpack(ORIGINAL_ROW)
)

# Option C: Access training predictions (no struct needed, already computed)
train_predicted = fitted_pipeline.fitted_steps[-1].predicted
```

### Rule #4: Handle Mixed Data Types WITHOUT ColumnTransformer (Common Pitfall #2)
```python
# ❌ WRONG: ColumnTransformer is NOT supported
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(...)  # ValueError: can't handle type ColumnTransformer

# ✅ CORRECT Option 1: Preprocess in Snowflake/Ibis
# One-hot encode categoricals BEFORE caching for ML
encoded_data = data.mutate(
    is_category_a=(_.CATEGORY == 'A').cast('int'),
    is_category_b=(_.CATEGORY == 'B').cast('int'),
    LOG_AMOUNT=_.AMOUNT.ln()
).drop('CATEGORY')  # Remove original categorical
# Now all features are numeric, use simple pipeline
sklearn_pipeline = SkPipeline([
    ('scaler', StandardScaler()),  # Works on all numeric features
    ('model', LogisticRegression())
])

# ✅ CORRECT Option 2: Custom OneHotEncoder wrapper (advanced)
# See bank_marketing.py example for OneHotStep implementation
```

### Rule #5: Use Struct Pattern to Avoid Relation Issues (Common Pitfall #3)
```python
# ❌ WRONG: Mixing different relations
test_with_pred = test.mutate(
    predictions=fitted_pipeline.predict(test.select(features))  # Different relation!
)

# ❌ WRONG: Direct mutate can cause column conflicts
test_with_pred = test.mutate(
    predictions=fitted_pipeline.predict(test[features])  # Risk of conflicts!
)

# ✅ CORRECT: Use struct pattern to preserve original and add predictions
test_with_pred = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted_pipeline.predict)
    .drop(target)  # Optional
    .unpack(ORIGINAL_ROW)
)
```

## COMPLETE ML WORKFLOW

### Helper Functions for Struct Pattern

The struct pattern preserves all original columns when making predictions:

```python
import toolz

# Constants for struct pattern
ORIGINAL_ROW = "original_row"
STRUCTED = "structed"
PREDICTED = "predicted"

@toolz.curry
def as_struct(expr, name=None):
    """Convert all columns of an expression into a single struct column.

    This is essential for predictions because it preserves all original
    columns while transforming the data, then allows unpacking them back.
    """
    struct = xo.struct({c: expr[c] for c in expr.columns})
    if name:
        struct = struct.name(name)
    return struct
```

**Why use the struct pattern?**
- **Preserves all columns**: When you call `.predict()`, you don't lose non-feature columns
- **Safe column management**: The original row is stored as a struct, preventing column conflicts
- **Composable**: Works cleanly with `.pipe()` for pipeline operations
- **Idempotent**: You can rerun predictions without worrying about losing data

### 1. Data Loading and Feature Engineering (HIGHLY ENCOURAGED!)

**Feature engineering is CRITICAL for model performance!** Create transformations liberally.

```python
import xorq.api as xo
from xorq.caching import ParquetCache

# Load data
data = xo.deferred_read_parquet(
    con=xo.duckdb.connect(),
    path="data.parquet",
    table_name="data"
)

# ✅ ENGINEER FEATURES - This improves model performance!
# Mathematical transformations
engineered_data = data.mutate(
    # Log transformations (use .ln() NOT .log()!)
    log_price=_.price.ln(),
    log_carat=_.carat.ln(),

    # Power transformations
    price_squared=_.price ** 2,
    carat_squared=_.carat ** 2,
    sqrt_depth=_.depth.sqrt(),

    # Interactions
    carat_x_depth=_.carat * _.depth,
    price_per_carat=_.price / _.carat,

    # Polynomial features
    carat_cubed=_.carat ** 3,
    depth_squared=_.depth ** 2,

    # Standardization (optional - sklearn can do this too)
    price_zscore=(_.price - _.price.mean()) / _.price.std(),

    # Binning
    price_category=ibis.case()
        .when(_.price < 1000, 'low')
        .when(_.price < 5000, 'mid')
        .else_('high')
        .end()
)

# ✅ IMPORTANT: Use .ln() for log, NOT .log()!
# In Snowflake/Ibis: .ln() is natural log
# Also available: .log10(), .log2()
```

**Why feature engineering matters:**
- Models can't learn transformations they need
- Log transforms handle skewness
- Polynomials capture non-linear relationships
- Interactions reveal combined effects
- Proper scaling improves convergence

**See transformation_patterns.md for comprehensive list of all available transformations!**

### 2. Train/Test Splitting
```python
# After feature engineering, select features and split

# Define features and target upfront
features = ("log_price", "carat_squared", "carat_x_depth", "sqrt_depth")  # Engineered features!
target = "price_category"

# Select and clean
clean_data = engineered_data.select(features + (target,)).drop_null()

# Split and cache (still deferred!)
train, test = [
    expr.cache(ParquetCache.from_kwargs())
    for expr in xo.train_test_splits(clean_data, test_sizes=0.2, random_seed=42)
]
```

### 3. Model Training
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from xorq.expr.ml.pipeline_lib import Pipeline

# Create sklearn pipeline
# ⚠️ ColumnTransformer is NOT supported by xorq!
# Use simple pipelines that apply transformations to all features
sklearn_pipeline = SkPipeline([
    ('scaler', StandardScaler()),  # Applies to all numeric features
    ('model', LogisticRegression(C=1E-4))
])

# For mixed data types (numeric + categorical):
# 1. Handle categorical encoding in Snowflake/Ibis BEFORE ML:
#    - Use ibis .case() for one-hot encoding
#    - Use ibis operations for label encoding
# 2. Ensure all features are numeric before caching for ML
# 3. Use simple sklearn pipeline with StandardScaler + model

# Convert to XorQ pipeline (REQUIRED!)
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

# Fit the pipeline (returns deferred expression)
fitted_pipeline = xorq_pipeline.fit(train, features=features, target=target)
```

### 4. Predictions - ALWAYS Use Struct Pattern!

#### Pattern A: Standalone Predictions with Struct (RECOMMENDED)
```python
# Generate predictions preserving all original columns
test_predicted = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted_pipeline.predict)
    .drop(target)  # Optional: drop target column if present
    .unpack(ORIGINAL_ROW)
)
test_predicted_cached = test_predicted.cache(ParquetCache.from_kwargs())

# Execute when ready
predictions = test_predicted_cached.execute()
```

#### Pattern B: Manual Struct Construction (More Control)
```python
# Manually construct the prediction pipeline
test_with_predictions = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))
    .select(
        fitted_pipeline.predict(test).name(PREDICTED),
        ORIGINAL_ROW
    )
    .unpack(ORIGINAL_ROW)
    .cache(ParquetCache.from_kwargs())
)

# Now predictions are a column alongside all original columns
results = test_with_predictions.execute()
```

### 5. Model Evaluation - ONE Function for ALL Metrics!
```python
import xorq.expr.ml as ml
import xorq.expr.datatypes as dt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 🎯 KEY INSIGHT: ml.deferred_sklearn_metric() works with ANY sklearn metric!
# Get the deferred model from the fitted pipeline
deferred_model = fitted_pipeline.predict_step.deferred_model

# Basic accuracy (simplest case)
accuracy = ml.deferred_sklearn_metric(
    expr=test,
    target=target,
    features=features,
    deferred_model=deferred_model,  # Use this pattern!
    metric_fn=accuracy_score
)

# Multiclass metrics with averaging strategies
precision_macro = ml.deferred_sklearn_metric(
    expr=test,
    target=target,
    features=features,
    deferred_model=deferred_model,
    metric_fn=precision_score,
    metric_kwargs={'average': 'macro', 'zero_division': 0}
)

# Weighted metrics (for imbalanced classes)
f1_weighted = ml.deferred_sklearn_metric(
    expr=test,
    target=target,
    features=features,
    deferred_model=deferred_model,
    metric_fn=f1_score,
    metric_kwargs={'average': 'weighted', 'zero_division': 0}
)

# Execute all metrics at once (deferred execution!)
results = {
    'accuracy': accuracy.execute(),
    'precision_macro': precision_macro.execute(),
    'f1_weighted': f1_weighted.execute()
}
```

## QUICK DECISION GUIDE

```
Q: Remote data (Snowflake/BigQuery)?
├─ YES → Cache locally first: data.cache()
└─ NO → Continue

Q: Standard sklearn model?
├─ YES → Use Pipeline.from_instance()
└─ NO → Use UDF pattern (see ml_advanced.md)

Q: Need predictions multiple times?
├─ YES → Use table.mutate(predictions=...)
└─ NO → Use standalone fitted.predict()

Q: Need probability scores?
├─ YES → Set use_probabilities=True
└─ NO → Use regular predictions
```

## MODEL VALIDATION

### Sanity Checks for Model Performance:
```python
# Always verify your model makes sense
if r2_score < 0:
    print("WARNING: Negative R² means model is worse than mean baseline!")
    if r2_score < -100:  # Extremely negative R² (e.g., -5086)
        print("CRITICAL: Extremely negative R² indicates severe data issues!")
        print("Action required:")
        print("  1. Apply LOG TRANSFORMS to skewed features (use _.feature.ln())")
        print("  2. Check target variable - may need transformation too")
        print("  3. Verify training/test split - ensure no data mismatch")
        print("  4. Review feature engineering - create interactions, polynomials")
        print("  5. Check for outliers - filter extreme values")
        print("Example fix:")
        print("  data = data.mutate(")
        print("      log_price=_.price.ln(),")
        print("      log_carat=_.carat.ln(),")
        print("      price_per_unit=_.price / _.quantity")
        print("  )")
    else:
        print("Likely causes: wrong features, data leakage, or model misconfiguration")
        print("Try: Better feature engineering with transformations and interactions")

# Check for overfitting
if train_score > 0.95 and test_score < 0.5:
    print("WARNING: Possible overfitting - model performs well on training but poor on test")

# Check for data leakage
if test_score > 0.99:
    print("WARNING: Suspiciously high test score - check for data leakage")
```

## THE GOLDEN RULES SUMMARY

1. **Deferred everything** - Only `.execute()` at the very end
2. **Always wrap sklearn** - Use `Pipeline.from_instance()`
3. **ALWAYS use struct pattern for predictions** - Preserve all columns, avoid conflicts
4. **Engineer features liberally** - Use `.ln()` (NOT `.log()`!), powers, interactions, etc.
5. **Use xorq's train_test_splits** - Never sklearn's version
6. **Check transformer support** - Most sklearn transformers need registration
7. **Cast regression targets** - Use float64 for regression targets
8. **Use `from xorq.vendor import ibis`** - For desc(), case(), etc.
9. **Sanity check metrics** - Negative R² or suspiciously high scores indicate issues

---

**For advanced topics, see:**
- `resources/ml_examples.md` - Complete working examples
- `resources/ml_troubleshooting.md` - Debugging common errors
- `resources/ml_advanced.md` - Remote data patterns, UDF patterns, supported components
