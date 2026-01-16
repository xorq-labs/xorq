# Xorq Machine Learning Complete Guide

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

**See Rule #3 below for complete details.**

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

## COMPLETE ML WORKFLOW (Reference: ../xorq/xorq-template/expr.py)

### Helper Functions for Struct Pattern

The struct pattern preserves all original columns when making predictions, which is crucial for maintaining data integrity and avoiding column loss:

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

### When to Use the Struct Pattern (ALWAYS for ML Predictions!)

#### ✅ ALWAYS Use Struct Pattern When:

1. **Making predictions on a test/validation set**
   ```python
   # You want ALL columns (IDs, metadata, etc.) alongside predictions
   test_with_predictions = (
       test
       .mutate(as_struct(name=ORIGINAL_ROW))
       .pipe(fitted_pipeline.predict)
       .unpack(ORIGINAL_ROW)
   )
   ```

2. **Applying trained models to new data**
   ```python
   # Scoring new data while keeping original columns
   scored_data = (
       new_data
       .mutate(as_struct(name=ORIGINAL_ROW))
       .pipe(fitted_pipeline.predict)
       .unpack(ORIGINAL_ROW)
   )
   ```

3. **Working with data that has non-feature columns you need to keep**
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

#### Why the Struct Pattern is Superior:

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

#### How the Struct Pattern Works:

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

#### Real-World Example:

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

#### Common Mistakes Avoided:

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

# Per-class metrics (returns array)
per_class_precision = ml.deferred_sklearn_metric(
    expr=test,
    target=target,
    features=features,
    deferred_model=deferred_model,
    metric_fn=precision_score,
    metric_kwargs={'average': None, 'zero_division': 0},
    return_type=dt.Array(dt.float64)  # Required for array returns!
)

# Probability-based metrics (ROC-AUC, log loss)
roc_auc = ml.deferred_sklearn_metric(
    expr=test,
    target=target,
    features=features,
    deferred_model=deferred_model,
    metric_fn=roc_auc_score,
    use_probabilities=True  # Uses predict_proba instead of predict
)

# If using predictions column (from Pattern B)
accuracy_from_col = ml.deferred_sklearn_metric(
    expr=test_with_predictions,
    target=target,
    predictions_col="predictions",  # Use column instead of deferred_model
    metric_fn=accuracy_score
)

# Custom metrics - just pass your function!
def custom_weighted_accuracy(y_true, y_pred, weight_factor=2.0):
    """Custom metric example"""
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred) * weight_factor

custom_metric = ml.deferred_sklearn_metric(
    expr=test,
    target=target,
    features=features,
    deferred_model=deferred_model,
    metric_fn=custom_weighted_accuracy,
    metric_kwargs={'weight_factor': 1.5},
    return_type=dt.float64  # Specify return type for custom metrics
)

# Execute all metrics at once (deferred execution!)
results = {
    'accuracy': accuracy.execute(),
    'precision_macro': precision_macro.execute(),
    'f1_weighted': f1_weighted.execute(),
    'per_class': per_class_precision.execute(),
    'roc_auc': roc_auc.execute(),
    'custom': custom_metric.execute()
}
```

## REMOTE DATA PATTERNS (Snowflake/BigQuery)

### Principle: Push computation to backend, cache locally only for ML

```python
# 1. Feature engineering on REMOTE (efficient!)
remote_features = remote_table.mutate(
    log_amount=_.amount.ln(),
    amount_squared=_.amount ** 2,
    days_since=_.date.delta('2024-01-01', 'day')
).filter(
    (_.target.notnull()) &
    (_.amount > 0)
).select(feature_cols + [target_col])

# 2. Cache locally ONLY for ML (minimizes transfer)
local_data = remote_features.cache(ParquetCache.from_kwargs())
# OR
local_data = remote_features.into_backend(xo.connect())

# 3. Now proceed with standard ML workflow
train, test = xo.train_test_splits(local_data, test_sizes=0.2)
# ... continue as above
```

## ADVANCED: ML UDF PATTERNS (For Unsupported Models)

### Train Once, Predict Many Pattern
```python
from xorq.expr.udf import agg, make_pandas_expr_udf
import pickle

# Step 1: Train model via UDAF
def train_model(df):
    from sklearn.ensemble import GradientBoostingClassifier
    X = df[features].values
    y = df['label'].values

    model = GradientBoostingClassifier(n_estimators=50)
    model.fit(X, y)

    return pickle.dumps({'model': model, 'score': model.score(X, y)})

train_udf = agg.pandas_df(
    fn=train_model,
    schema=train_t.select(features + ['label']).schema(),
    return_type=dt.binary,
    name="train_model"
)

# Step 2: Apply model via UDF
def predict_with_model(model_data, df):
    model = model_data['model']
    X = df[features].values
    predictions = model.predict(X)
    return pd.Series(predictions, index=df.index)

# Step 3: Connect train → predict
predict_udf = make_pandas_expr_udf(
    computed_kwargs_expr=train_udf.on_expr(train_t),  # Train ONCE
    fn=predict_with_model,                            # Predict MANY
    schema=test_t.select(features).schema(),
    return_type=dt.string,  # or appropriate type
    name="predict"
)

# Execute complete pipeline
predictions = test_t.mutate(
    prediction=predict_udf.on_expr(test_t)
).execute()  # Training happens here, then predictions!
```

## THE MINIMAL API ADVANTAGE

### 🎯 ONE Function for ALL Metrics: `ml.deferred_sklearn_metric()`

**Why this is powerful:**
- Works with ANY sklearn metric function (accuracy, precision, recall, f1, roc_auc, mse, mae, r2, etc.)
- Works with custom metric functions
- Handles scalar and array returns
- No need to wait for wrapper functions to be created
- Clean, maintainable, infinitely extensible

### Key Patterns:
```python
# Pattern 1: Get deferred_model from fitted pipeline
deferred_model = fitted_pipeline.predict_step.deferred_model

# Pattern 2: Use any sklearn metric
from sklearn.metrics import any_metric_you_want
result = ml.deferred_sklearn_metric(
    expr=test,
    target=target,
    features=features,
    deferred_model=deferred_model,
    metric_fn=any_metric_you_want,
    metric_kwargs={...}  # Pass any parameters the metric needs
)

# Pattern 3: For array returns (per-class metrics)
import xorq.expr.datatypes as dt
per_class = ml.deferred_sklearn_metric(
    ...
    metric_kwargs={'average': None},
    return_type=dt.Array(dt.float64)  # Required for array returns!
)

# Pattern 4: For custom metrics
def my_custom_metric(y_true, y_pred, **kwargs):
    # Your custom logic
    return result

custom = ml.deferred_sklearn_metric(
    ...
    metric_fn=my_custom_metric,
    metric_kwargs={'my_param': value},
    return_type=dt.float64  # Specify return type
)
```

## SUPPORTED SKLEARN COMPONENTS (Verified from xorq source)

### Models (Use with Pipeline.from_instance())
**These work based on examples:**
- **Classification**: LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, SVC, GaussianNB, QuadraticDiscriminantAnalysis, AdaBoostClassifier, MLPClassifier
- **Regression**: LinearRegression, Lasso, ElasticNet (others may work but unverified)

### Preprocessing (LIMITED - Most require registration!)
**Officially registered in structer.py:**
- **Scalers**: StandardScaler (✅ confirmed)
- **Imputation**: SimpleImputer
- **Feature Selection**: SelectKBest

**NOT registered (will fail with "can't handle type"):**
- MinMaxScaler, RobustScaler, Normalizer
- PCA, TruncatedSVD
- Most other sklearn transformers

**Workaround available:**
- OneHotEncoder (requires custom Step class - see bank_marketing.py)

### Metrics (Use with ml.deferred_sklearn_metric())
**ANY sklearn metric!** Including but not limited to:
- **Classification**: accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix, classification_report
- **Regression**: mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
- **Clustering**: silhouette_score, calinski_harabasz_score, davies_bouldin_score
- **Custom**: Any function that takes (y_true, y_pred) and returns a scalar or array

### For Unsupported Models
Use the UDF pattern shown above for models like XGBoost, LightGBM, custom models, etc.

## DEBUGGING COMMON ERRORS

### Error: "AttributeError: module 'ibis' has no attribute 'vendor'"
**Problem**: Trying to access `xo.vendor.ibis.desc()` which doesn't exist.

**Why it happens**:
```python
# ❌ WRONG: Trying to access through xo.vendor
xo.vendor.ibis.desc('count')  # AttributeError: no attribute 'vendor'

# ❌ WRONG: Importing ibis directly
import ibis  # May not work in xorq environment
```

**Solution**: Import ibis from xorq.vendor
```python
# ✅ CORRECT import
from xorq.vendor import ibis

# Then use normally
grouped = table.group_by(_.COLOR).aggregate([
    _.count().name('count')
])
ordered = grouped.order_by(ibis.desc('count'))  # ✅ Works!
```

**Complete pattern for ordering after aggregation**:
```python
from xorq.vendor import ibis  # Do this ONCE at top of file

# After aggregation, MUST use string names for ordering
grouped = table.group_by(_.CATEGORY).aggregate([
    _.PRICE.mean().name('avg_price'),
    _.count().name('count')
])

# ✅ CORRECT: Use ibis.desc() with string column name
ordered = grouped.order_by(ibis.desc('count'))

# ❌ WRONG: Using _.count.desc() fails
# ordered = grouped.order_by(_.count.desc())  # AttributeError!

# ❌ WRONG: Accessing through xo.vendor
# ordered = grouped.order_by(xo.vendor.ibis.desc('count'))  # AttributeError!
```

### Error: "AttributeError: 'DataFrame' object has no attribute 'fit'"
**Fix**: You're using bare sklearn. Wrap with `Pipeline.from_instance()`

### Error: "Cannot execute on remote backend"
**Fix**: Cache to local first: `local_data = remote_table.cache()`

### Error: "train_test_split not found"
**Fix**: Use `xo.train_test_splits()` not sklearn's version

### Error: "Predictions don't match expected shape"
**Fix**: You're mixing prediction patterns. Pick one and stick with it!

### Error: "ArrowInvalid: Float value was truncated converting to int64"
**Problem**: Regression models return float64 predictions, but your target column is int64.

**Why it happens**:
```python
# Your target is int64 in the schema
# LinearRegression.predict() returns float64
# Arrow can't convert negative floats or decimals to int64
```

**Solution**: Cast target to float64 BEFORE training
```python
# ✅ CORRECT: Cast target to float for regression
data = data.mutate(TARGET=_.TARGET.cast('float64'))
train, test = xo.train_test_splits(data, test_sizes=0.2)

# Now train your regression model
fitted = xorq_pipeline.fit(train, features=features, target='TARGET')
```

### Error: "unhashable type: 'list'" (ColumnTransformer Issue)
**Problem**: XorQ needs to hash sklearn objects for caching/deterministic execution, but `ColumnTransformer` contains lists in its configuration that make it unhashable.

**Why it happens**:
```python
# ❌ This causes hashing errors
ct = ColumnTransformer(
    transformers=[  # List causes problems!
        ('num', StandardScaler(), numeric_features),  # List features also problematic
        ('cat', OneHotEncoder(), categorical_features)
    ]
)
```

**Solutions**:

**Option 1 (RECOMMENDED): Use tuples everywhere with ColumnTransformer**
```python
# ✅ Use tuples for transformers AND feature lists
ct = ColumnTransformer(
    transformers=(  # Tuple instead of list
        ('num', StandardScaler(), tuple(numeric_features)),  # Tuple features
        ('cat', OneHotEncoder(sparse_output=False), tuple(categorical_features)),
    )
)
sklearn_pipeline = SkPipeline([
    ('preprocessor', ct),
    ('model', LogisticRegression())
])
# This pattern works with xorq's hashing!
```

**Option 2: Simple pipeline (if you don't need mixed preprocessing)**
```python
# ✅ Simple pipeline for numeric-only features
sklearn_pipeline = SkPipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
# Always works, but limited to single preprocessing type
```

### Error: "IntegrityError: Cannot add Field ... they belong to another relation"
**Problem**: When using `mutate()` to add predictions, the expression must be anchored to the same relation as the table being mutated.

**Why it happens**:
```python
# ❌ WRONG: Mixing relations
test_with_predictions = test.mutate(
    predictions=fitted_pipeline.predict(test.select(features))  # Different relation!
)
```
The `test.select(features)` creates a new relation, and you can't mutate the original `test` table with expressions from this derived relation.

**Solution: ALWAYS Use Struct Pattern**

The struct pattern elegantly solves all relation issues by preserving the original table structure:

```python
# ✅ RECOMMENDED: Struct pattern handles all relations correctly
test_with_predictions = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))  # Store original row
    .pipe(fitted_pipeline.predict)          # Pipeline handles features
    .drop(target)                           # Optional: remove target
    .unpack(ORIGINAL_ROW)                   # Restore original columns
)
```

**Why this works**:
- The struct preserves all columns, avoiding relation conflicts
- `.pipe()` lets the pipeline handle feature selection internally
- `.unpack()` restores the original columns alongside predictions
- No need to worry about which relation you're referencing

## QUICK DECISION GUIDE

```
Q: Remote data (Snowflake/BigQuery)?
├─ YES → Cache locally first: data.cache()
└─ NO → Continue

Q: Standard sklearn model?
├─ YES → Use Pipeline.from_instance()
└─ NO → Use UDF pattern

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

## COMPLETE WORKING EXAMPLE

```python
import xorq.api as xo
from xorq.expr.ml.pipeline_lib import Pipeline
from xorq.caching import ParquetCache
import xorq.expr.ml as ml
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
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
    for expr in xo.train_test_splits(data, test_sizes=0.2, random_seed=42)
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
*Remember: The power of XorQ is deferred execution. Build your entire pipeline first, execute once at the end!*
