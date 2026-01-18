# ML Pipelines

## Overview

Xorq provides first-class integration with scikit-learn for building ML pipelines with deferred execution, enabling reproducible and composable machine learning workflows.

---

## Core ML Patterns

### 1. Deferred ML Operations

**Key Concept**: ML operations (fit, transform, predict) are deferred until execution, enabling:
- Expression-based ML workflows
- Lazy evaluation of entire pipelines
- Optimization opportunities
- Reproducible training

**Pattern**: Use deferred fit/transform/predict functions

```python
from xorq.expr.ml import (
    deferred_fit_transform_sklearn_struct,
    deferred_fit_predict_sklearn,
    train_test_splits
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Split data
train, test = train_test_splits(data, test_size=0.2)

# Deferred scaler fitting and transformation
*_, deferred_transform = deferred_fit_transform_sklearn_struct(
    train,
    features=["sepal_length", "sepal_width"],
    cls=StandardScaler,
    params=()
)

# Build expression (no execution yet)
transformed = (
    train
    .select(
        deferred_transform.on_expr(train).name("scaled"),
        "species"
    )
    .unpack("scaled")
)

# Execute when ready
result = transformed.execute()
```

---

## Train/Test Splitting

### Pattern: Deterministic Splits

**Problem**: Need reproducible train/test splits

**Solution**: Use `train_test_splits()` for deterministic splitting

```python
from xorq.expr.ml import train_test_splits

# Simple split
train, test = train_test_splits(data, test_size=0.2)

# With stratification
train, test = train_test_splits(
    data,
    test_size=0.2,
    stratify="target_column"
)

# Multiple splits
train, val, test = train_test_splits(
    data,
    test_size=0.2,
    val_size=0.1
)
```

**Features:**
- Deterministic (same split every time)
- Maintains expression-based workflow
- Supports stratification
- Multiple split support

---

## Deferred Fitting and Transformation

### Pattern: Fit Transformers on Training Data

```python
from xorq.expr.ml import deferred_fit_transform_sklearn_struct
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Deferred scaler
*_, deferred_scaler = deferred_fit_transform_sklearn_struct(
    train,
    features=["age", "income", "score"],
    cls=StandardScaler,
    params=()
)

# Apply to training data
train_scaled = (
    train
    .select(
        deferred_scaler.on_expr(train).name("scaled_features"),
        "target"
    )
    .unpack("scaled_features")
)

# Apply same transformation to test data
test_scaled = (
    test
    .select(
        deferred_scaler.on_expr(test).name("scaled_features"),
        "target"
    )
    .unpack("scaled_features")
)

# Execute both
train_result = train_scaled.execute()
test_result = test_scaled.execute()
```

**Key points:**
- Scaler fits on training data during execution
- Same fitted scaler applies to test data
- No data leakage (test data doesn't influence fit)
- `.on_expr()` applies deferred operation to expression

---

## Deferred Prediction

### Pattern: Fit and Predict with Sklearn Models

```python
from xorq.expr.ml import deferred_fit_predict_sklearn
from sklearn.ensemble import RandomForestClassifier

# Define features and target
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target = "species"

# Deferred model fitting and prediction
*_, deferred_predict = deferred_fit_predict_sklearn(
    train,
    target=target,
    features=features,
    cls=RandomForestClassifier,
    return_type=train[target].type(),  # Match target type
    params=(
        ("n_estimators", 100),
        ("max_depth", 5),
        ("random_state", 42)
    )
)

# Apply predictions to test set
predictions = test.mutate(
    prediction=deferred_predict.on_expr
)

# Execute and evaluate
result = predictions.execute()
```

**Features:**
- Model fits on training data during execution
- `return_type` ensures type safety
- `params` passed as tuple of (name, value) pairs
- `.on_expr` applies prediction to current expression

---

## Pipeline Integration

### Pattern: Convert Sklearn Pipelines to Xorq

```python
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from xorq.expr.ml.pipeline_lib import Pipeline

# Create sklearn pipeline
sk_pipeline = SkPipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("classifier", KNeighborsClassifier(n_neighbors=11))
])

# Convert to xorq pipeline
xorq_pipeline = Pipeline.from_instance(sk_pipeline)

# Fit pipeline
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target = "species"
fitted_pipeline = xorq_pipeline.fit(train, features=features, target=target)

# Apply to test data
predictions = test.pipe(fitted_pipeline.predict)

# Execute
result = predictions.execute()
```

**Benefits:**
- Reuse existing sklearn pipelines
- Expression-based composition
- Fitted pipelines are reusable
- Can serialize fitted pipelines

---

## Complete ML Workflow Example

### Pattern: Full Training Pipeline

```python
import xorq.api as xo
from xorq.expr.ml import train_test_splits
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xorq.expr.ml.pipeline_lib import Pipeline

# Load data
data = xo.examples.iris.fetch()

# Define features and target
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target = "species"

# Split data
train, test = train_test_splits(data, test_size=0.2)

# Create sklearn pipeline
sk_pipeline = SkPipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=11))
])

# Convert to xorq pipeline
xorq_pipeline = Pipeline.from_instance(sk_pipeline)

# Fit on training data
fitted = xorq_pipeline.fit(train, features=features, target=target)

# Predict on test data with original data preserved
import toolz

@toolz.curry
def as_struct(expr, name=None):
    struct = xo.struct({c: expr[c] for c in expr.columns})
    if name:
        struct = struct.name(name)
    return struct

predictions = (
    test
    .mutate(original=as_struct(name="original_row"))  # Preserve original
    .pipe(fitted.predict)                              # Apply predictions
    .unpack("original_row")                            # Restore original columns
)

# Execute and evaluate
result = predictions.execute()

# Calculate accuracy
accuracy = (result["species"] == result["prediction"]).mean()
print(f"Accuracy: {accuracy:.2%}")
```

---

## Model Composition

### Pattern: Compose Multiple Models

```python
from xorq.expr.ml import deferred_fit_predict_sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Fit multiple models
*_, rf_predict = deferred_fit_predict_sklearn(
    train,
    target=target,
    features=features,
    cls=RandomForestClassifier,
    return_type=train[target].type(),
    params=(("n_estimators", 100),)
)

*_, gb_predict = deferred_fit_predict_sklearn(
    train,
    target=target,
    features=features,
    cls=GradientBoostingClassifier,
    return_type=train[target].type(),
    params=(("n_estimators", 100),)
)

# Create ensemble predictions
ensemble = test.mutate(
    rf_prediction=rf_predict.on_expr,
    gb_prediction=gb_predict.on_expr,
    # Voting ensemble
    final_prediction=xo.case()
        .when(xo._.rf_prediction == xo._.gb_prediction, xo._.rf_prediction)
        .else_(xo._.rf_prediction)  # Break ties with RF
        .end()
)

result = ensemble.execute()
```

---

## Feature Engineering in ML Pipelines

### Pattern: Combine Feature Engineering with ML

```python
# Feature engineering pipeline
engineered = (
    train
    .mutate(
        age_squared=xo._.age ** 2,
        age_income=xo._.age * xo._.income,
        log_income=xo._.income.log(),
        age_group=xo._.age // 10
    )
    .select(
        "target",
        "age", "age_squared",
        "income", "log_income", "age_income",
        "age_group"
    )
)

# Fit model on engineered features
features_eng = [
    "age", "age_squared", "income", "log_income",
    "age_income", "age_group"
]

*_, deferred_predict = deferred_fit_predict_sklearn(
    engineered,
    target="target",
    features=features_eng,
    cls=RandomForestClassifier,
    return_type=train["target"].type(),
    params=(("n_estimators", 100),)
)

# Apply same engineering to test
test_engineered = (
    test
    .mutate(
        age_squared=xo._.age ** 2,
        age_income=xo._.age * xo._.income,
        log_income=xo._.income.log(),
        age_group=xo._.age // 10
    )
)

predictions = test_engineered.mutate(prediction=deferred_predict.on_expr)
result = predictions.execute()
```

---

## Cross-Validation

### Pattern: K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold

# Manual k-fold with xorq
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Get indices from pandas DataFrame
data_pd = data.execute()
scores = []

for train_idx, val_idx in kf.split(data_pd):
    # Create train/val splits
    train_fold = con.create_table(
        f"train_fold",
        data_pd.iloc[train_idx]
    )
    val_fold = con.create_table(
        f"val_fold",
        data_pd.iloc[val_idx]
    )

    # Train model
    *_, deferred_predict = deferred_fit_predict_sklearn(
        train_fold,
        target=target,
        features=features,
        cls=KNeighborsClassifier,
        return_type=train_fold[target].type(),
        params=(("n_neighbors", 11),)
    )

    # Validate
    predictions = val_fold.mutate(prediction=deferred_predict.on_expr)
    result = predictions.execute()

    # Calculate score
    score = (result[target] == result["prediction"]).mean()
    scores.append(score)

print(f"CV Scores: {scores}")
print(f"Mean CV Score: {sum(scores) / len(scores):.2%}")
```

---

## Hyperparameter Tuning

### Pattern: Grid Search with Xorq

```python
from itertools import product

# Define parameter grid
param_grid = {
    "n_neighbors": [3, 5, 7, 11],
    "weights": ["uniform", "distance"]
}

# Generate all combinations
param_combinations = [
    dict(zip(param_grid.keys(), values))
    for values in product(*param_grid.values())
]

# Evaluate each combination
results = []
for params in param_combinations:
    *_, deferred_predict = deferred_fit_predict_sklearn(
        train,
        target=target,
        features=features,
        cls=KNeighborsClassifier,
        return_type=train[target].type(),
        params=tuple(params.items())
    )

    predictions = test.mutate(prediction=deferred_predict.on_expr)
    result = predictions.execute()

    score = (result[target] == result["prediction"]).mean()
    results.append({"params": params, "score": score})

# Find best parameters
best = max(results, key=lambda x: x["score"])
print(f"Best params: {best['params']}")
print(f"Best score: {best['score']:.2%}")
```

---

## Model Persistence

### Pattern: Save and Load Fitted Pipelines

```python
import pickle

# Fit pipeline
fitted_pipeline = xorq_pipeline.fit(train, features=features, target=target)

# Save to disk
with open("fitted_pipeline.pkl", "wb") as f:
    pickle.dump(fitted_pipeline, f)

# Load from disk
with open("fitted_pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)

# Use loaded pipeline
predictions = test.pipe(loaded_pipeline.predict)
result = predictions.execute()
```

---

## Best Practices

### 1. Use Deferred Fitting for Reproducibility

```python
# Good: deferred fitting
*_, deferred_fit = deferred_fit_transform_sklearn_struct(
    train, features=features, cls=StandardScaler, params=()
)

# Avoid: eager fitting
scaler = StandardScaler()
scaler.fit(train[features].execute())  # Executes immediately
```

### 2. Preserve Original Data with Structs

```python
# Good: preserve original columns
predictions = (
    test
    .mutate(original=as_struct(name="original_row"))
    .pipe(fitted.predict)
    .unpack("original_row")
)

# Avoid: losing original data
predictions = test.pipe(fitted.predict)  # Original columns may be lost
```

### 3. Use Pipeline for Composition

```python
# Good: composable pipeline
pipeline = Pipeline.from_instance(sklearn_pipeline)
fitted = pipeline.fit(train, features=features, target=target)

# Avoid: manual step-by-step
scaled = apply_scaler(train)
reduced = apply_pca(scaled)
predictions = apply_model(reduced)
```

### 4. Specify Return Types

```python
# Good: explicit return type
*_, deferred_predict = deferred_fit_predict_sklearn(
    train,
    target=target,
    features=features,
    cls=KNeighborsClassifier,
    return_type=train[target].type(),  # Explicit
    params=params
)

# Avoid: missing return type (may cause type errors)
```

### 5. Use Train/Test Splits Consistently

```python
# Good: deterministic splits
train, test = train_test_splits(data, test_size=0.2)

# Avoid: random splits that change
train = data.sample(frac=0.8)
test = data.anti_join(train)  # Not reproducible
```

---

## Troubleshooting

### Issue: Model Not Fitting

**Symptom**: Predictions fail or produce incorrect results

**Check:**
1. Are features defined correctly?
2. Is target column present in training data?
3. Are parameter types correct?

**Solution:**
```python
# Verify features exist
print(train.columns)
assert all(f in train.columns for f in features)

# Verify target exists
assert target in train.columns

# Check parameter format
params = (("n_neighbors", 11),)  # Tuple of (name, value) pairs
```

### Issue: Type Errors in Predictions

**Symptom**: `TypeError` or type mismatch errors

**Solution:**
```python
# Ensure return_type matches target type
target_type = train[target].type()

*_, deferred_predict = deferred_fit_predict_sklearn(
    train,
    target=target,
    features=features,
    cls=KNeighborsClassifier,
    return_type=target_type,  # Must match
    params=params
)
```

### Issue: Data Leakage

**Symptom**: Unrealistically high accuracy

**Check:**
- Is scaler fitted on training data only?
- Are test features computed independently?

**Solution:**
```python
# Fit scaler on training data
*_, deferred_scaler = deferred_fit_transform_sklearn_struct(
    train,  # Only training data
    features=features,
    cls=StandardScaler,
    params=()
)

# Apply same fitted scaler to test
test_scaled = test.select(
    deferred_scaler.on_expr(test).name("scaled"),  # Uses training fit
    target
)
```

---

## Summary

**Key ML patterns:**
1. **Deferred fitting**: Models train during execution, not expression building
2. **Pipeline integration**: Convert sklearn pipelines to xorq
3. **Struct preservation**: Keep original data during transformations
4. **Type safety**: Explicit return types for predictions
5. **Reproducibility**: Deterministic splits and deferred operations

**Common operations:**
- `train_test_splits()` - Deterministic splitting
- `deferred_fit_transform_sklearn_struct()` - Deferred transformers
- `deferred_fit_predict_sklearn()` - Deferred models
- `Pipeline.from_instance()` - Convert sklearn pipelines
- `.pipe()` - Apply fitted transformations
