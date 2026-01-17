# ML Troubleshooting Guide

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
Use the UDF pattern shown in ml_advanced.md for models like XGBoost, LightGBM, custom models, etc.
