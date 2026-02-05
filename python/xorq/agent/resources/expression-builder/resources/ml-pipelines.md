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

## ColumnTransformer

### Pattern: Mixed Preprocessing for Numeric and Categorical Features

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from xorq.expr.ml.pipeline_lib import Pipeline

# Define feature groups
numeric_features = ["age", "balance", "duration"]
categorical_features = ["job", "marital", "education"]

# Build ColumnTransformer with separate preprocessing
preprocessor = ColumnTransformer([
    ("num", SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", SkPipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_features)
])

# Create sklearn pipeline with preprocessor
sklearn_pipeline = SkPipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(n_estimators=50, random_state=42))
])

# Convert to xorq and fit
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
fitted = xorq_pipeline.fit(
    train,
    features=numeric_features + categorical_features,
    target="deposit"
)

# Predict
predictions = fitted.predict(test)
```

### Pattern: ColumnTransformer with Feature Selection

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

# Hybrid ColumnTransformer: known-schema + KV-encoded outputs
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), ["age", "income"]),        # Known-schema
    ("tfidf", TfidfVectorizer(), "text"),                    # KV-encoded
    ("cat", OneHotEncoder(sparse_output=False), ["category"]) # KV-encoded
])

sklearn_pipeline = SkPipeline([
    ("preprocessor", preprocessor),
    ("selector", SelectKBest(f_classif, k=5)),  # Feature selection after transform
    ("classifier", RandomForestClassifier(n_estimators=10, random_state=42))
])

xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
fitted = xorq_pipeline.fit(expr, features=all_features, target="target")
```

**Key points:**
- ColumnTransformer handles heterogeneous preprocessing
- Supports nested pipelines for each feature type
- Automatically concatenates transformed features
- Works with text features (TfidfVectorizer) and encoders (OneHotEncoder)
- Can chain with feature selection (SelectKBest, etc.)

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

## Real-World Examples

### Example 1: Multi-Classifier Comparison

**Use case**: Compare multiple sklearn classifiers on different datasets

**Pattern**: Use `Pipeline.from_instance()` to convert sklearn pipelines, then fit and evaluate in xorq

```python
import xorq.api as xo
from xorq.expr.ml import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Define features and target naming convention
target = "target"
feature_prefix = "feature_"

# Helper to create xorq expressions from numpy arrays
def make_exprs(X_train, y_train, X_test, y_test):
    con = xo.connect()
    train, test = (
        con.register(
            pd.DataFrame(X)
            .rename(columns=(feature_prefix + "{}").format)
            .assign(**{target: y}),
            name,
        )
        for (X, y, name) in (
            (X_train, y_train, "train"),
            (X_test, y_test, "test"),
        )
    )
    features = train.select(xo.selectors.startswith(feature_prefix)).columns
    return (train, test, features)

# Train and compare
def train_and_score(clf, X_train, X_test, y_train, y_test):
    # Create sklearn pipeline with preprocessing
    sklearn_pipeline = make_pipeline(StandardScaler(), clf).fit(X_train, y_train)
    sklearn_score = sklearn_pipeline.score(X_test, y_test)

    # Convert to xorq pipeline
    train, test, features = make_exprs(X_train, y_train, X_test, y_test)
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline).fit(
        train, features=features, target=target
    )
    xorq_score = xorq_pipeline.score_expr(test).execute()

    return {
        "sklearn_score": sklearn_score,
        "xorq_score": xorq_score,
    }

# Compare classifiers
classifiers = [
    ("KNN", KNeighborsClassifier(3)),
    ("Linear SVM", SVC(kernel="linear", C=0.025, random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("Random Forest", RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)),
]

# Test on dataset
from sklearn.datasets import make_moons
X, y = make_moons(noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

for name, clf in classifiers:
    scores = train_and_score(clf, X_train, X_test, y_train, y_test)
    print(f"{name}: sklearn={scores['sklearn_score']:.3f}, xorq={scores['xorq_score']:.3f}")
    # Scores should match exactly!
```

**Key patterns:**
- Use `make_pipeline()` for sklearn preprocessing + model
- Convert with `Pipeline.from_instance()`
- Use `.score_expr()` for deferred evaluation
- Scores match sklearn exactly (validates correctness)

---

### Example 2: Model Evaluation with Multiple Metrics

**Use case**: Evaluate models with precision, recall, F1, etc.

**Pattern**: Use `deferred_sklearn_metric()` for lazy metric computation

```python
import xorq.api as xo
from xorq.expr.ml import Pipeline
from xorq.expr.ml.metrics import deferred_sklearn_metric
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Define metric configurations
metric_configs = [
    ("accuracy", accuracy_score, {}),
    ("precision_macro", precision_score, {"average": "macro", "zero_division": 0}),
    ("precision_weighted", precision_score, {"average": "weighted", "zero_division": 0}),
    ("recall_macro", recall_score, {"average": "macro", "zero_division": 0}),
    ("recall_weighted", recall_score, {"average": "weighted", "zero_division": 0}),
    ("f1_macro", f1_score, {"average": "macro", "zero_division": 0}),
    ("f1_weighted", f1_score, {"average": "weighted", "zero_division": 0}),
]

def compute_metrics(clf, X_train, X_test, y_train, y_test):
    # Create expressions
    train, test, features = make_exprs(X_train, y_train, X_test, y_test)

    # Fit sklearn pipeline
    sklearn_pipeline = make_pipeline(clf).fit(X_train, y_train)

    # Convert to xorq and fit
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline).fit(
        train, features=features, target=target
    )

    # Get predictions (deferred)
    expr_with_preds = xorq_pipeline.predict(test)

    # Compute all metrics (deferred)
    xorq_metrics = {
        name: deferred_sklearn_metric(
            expr=expr_with_preds,
            target=target,
            pred_col="predicted",
            metric_fn=metric_fn,
            metric_kwargs=kwargs if kwargs else (),
        ).execute()
        for name, metric_fn, kwargs in metric_configs
    }

    return xorq_metrics

# Evaluate models
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_classes=3, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
]

for model_name, clf in models:
    metrics = compute_metrics(clf, X_train, X_test, y_train, y_test)
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.3f}")
```

**Key patterns:**
- Use `.predict()` to get expression with predictions
- `deferred_sklearn_metric()` computes metrics lazily
- Supports all sklearn metrics with kwargs
- Multiple metrics computed in single pass

---

### Example 3: Probability Predictions and ROC-AUC

**Use case**: Get probability predictions for calibration or ROC-AUC

**Pattern**: Use `.predict_proba()` method

```python
from xorq.expr.ml.metrics import deferred_sklearn_metric
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# Generate binary classification data
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and fit pipeline
train, test, features = make_exprs(X_train, y_train, X_test, y_test)
sklearn_pipeline = make_pipeline(LogisticRegression(max_iter=1000, random_state=42))
sklearn_pipeline.fit(X_train, y_train)

xorq_pipeline = Pipeline.from_instance(sklearn_pipeline).fit(
    train, features=features, target=target
)

# Get probability predictions
expr_with_proba = xorq_pipeline.predict_proba(test)

# Compute ROC-AUC (uses probability of positive class)
xorq_auc = deferred_sklearn_metric(
    expr=expr_with_proba,
    target=target,
    pred_col="predicted_proba",  # Column with probabilities
    metric_fn=roc_auc_score,
).execute()

print(f"ROC-AUC: {xorq_auc:.3f}")

# Compare with sklearn
sklearn_auc = roc_auc_score(y_test, sklearn_pipeline.predict_proba(X_test)[:, 1])
print(f"Sklearn ROC-AUC: {sklearn_auc:.3f}")
# Should match exactly!
```

**Key patterns:**
- `.predict_proba()` returns probabilities in `predicted_proba` column
- Works with any sklearn classifier that has `predict_proba()`
- Use with probability-based metrics like ROC-AUC, log loss

---

### Example 4: Decision Function for SVM

**Use case**: Get decision function scores (e.g., for SVM, linear models)

**Pattern**: Use `.decision_function()` method

```python
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

# Generate data
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_classes=2, random_state=123
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create pipeline with LinearSVC
train, test, features = make_exprs(X_train, y_train, X_test, y_test)
sklearn_pipeline = make_pipeline(LinearSVC(random_state=123, max_iter=5000))
sklearn_pipeline.fit(X_train, y_train)

xorq_pipeline = Pipeline.from_instance(sklearn_pipeline).fit(
    train, features=features, target=target
)

# Get decision function scores
expr_with_scores = xorq_pipeline.decision_function(test)

# Compute ROC-AUC using decision scores
xorq_auc = deferred_sklearn_metric(
    expr=expr_with_scores,
    target=target,
    pred_col="decision_function",  # Column with scores
    metric_fn=roc_auc_score,
).execute()

print(f"ROC-AUC (decision scores): {xorq_auc:.3f}")

# Compare with sklearn
sklearn_auc = roc_auc_score(y_test, sklearn_pipeline.decision_function(X_test))
print(f"Sklearn ROC-AUC: {sklearn_auc:.3f}")
```

**Key patterns:**
- `.decision_function()` for models that don't have `predict_proba()`
- Returns scores in `decision_function` column
- Useful for SVM, LinearSVC, Logistic Regression

---

### Example 5: Feature Importances

**Use case**: Extract feature importances from tree-based models

**Pattern**: Use `.feature_importances()` method

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Generate data
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=6,
    n_redundant=2, n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and fit pipeline
train, test, features = make_exprs(X_train, y_train, X_test, y_test)
sklearn_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42))
sklearn_pipeline.fit(X_train, y_train)

xorq_pipeline = Pipeline.from_instance(sklearn_pipeline).fit(
    train, features=features, target=target
)

# Get feature importances (deferred)
importances_expr = xorq_pipeline.feature_importances(test)
xorq_importances = np.array(
    importances_expr.execute()["feature_importances"].iloc[0]
)

# Compare with sklearn
sklearn_importances = sklearn_pipeline.named_steps["randomforestclassifier"].feature_importances_

print("Feature Importances:")
for i, (sklearn_imp, xorq_imp) in enumerate(zip(sklearn_importances, xorq_importances)):
    print(f"  Feature {i}: sklearn={sklearn_imp:.4f}, xorq={xorq_imp:.4f}")

# Should match exactly
assert np.allclose(sklearn_importances, xorq_importances)
```

**Key patterns:**
- `.feature_importances()` for tree-based models (RF, GBM, etc.)
- Returns array in `feature_importances` column
- Extract with `.iloc[0]` since it's stored as a single row

---

### Example 6: Complete Pipeline with Catalog Integration

**Use case**: Build ML pipeline, catalog it, reuse across sessions

**Pattern**: Build → Catalog → Load → Compose

```python
# Step 1: Build feature engineering pipeline
import xorq.api as xo

# Load data
data = xo.examples.penguins.fetch()

# Engineer features
features_expr = (
    data
    .filter(xo._.bill_length_mm.notnull())
    .mutate(
        bill_ratio=xo._.bill_length_mm / xo._.bill_depth_mm,
        flipper_body_ratio=xo._.flipper_length_mm / xo._.body_mass_g,
        bill_area=xo._.bill_length_mm * xo._.bill_depth_mm,
    )
    .select(
        "species",  # target
        "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g",
        "bill_ratio", "flipper_body_ratio", "bill_area",
    )
)

# Save to build script: features.py
# expr = features_expr

# Step 2: Build and catalog
# $ xorq build features.py -e expr
# $ xorq catalog add builds/<hash> --alias penguin-features

# Step 3: Build model pipeline using catalog
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xorq.expr.ml import train_test_splits, Pipeline

# Option A: Load features from parquet (recommended for ML)
# $ xorq run penguin-features -o features.parquet
features_df = pd.read_parquet("features.parquet")
con = xo.connect()
features = con.register(features_df, "features")

# Option B: Load from catalog and execute
# features = xo.catalog.get("penguin-features")
# features_df = features.execute()  # Execute to get data

# Split data
train, test = train_test_splits(features, test_size=0.2)

# Define model
sklearn_pipeline = SkPipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Convert and fit
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
feature_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm",
                "body_mass_g", "bill_ratio", "flipper_body_ratio", "bill_area"]
fitted = xorq_pipeline.fit(train, features=feature_cols, target="species")

# Predict
predictions = fitted.predict(test)

# Evaluate
score = fitted.score_expr(test).execute()
print(f"Model accuracy: {score:.3f}")

# Save model pipeline: model.py
# expr = predictions

# Step 4: Build and catalog model
# $ xorq build model.py -e expr
# $ xorq catalog add builds/<hash> --alias penguin-model

# Step 5: Compose pipelines via CLI
# $ xorq run penguin-features -o arrow | xorq run-unbound penguin-model --to_unbind_hash <hash>
```

**Key patterns:**
- Separate feature engineering and modeling
- Catalog each stage independently
- Compose via catalog or CLI streaming
- Reuse across sessions and notebooks

---

### Example 7: Advanced - UDAF + ExprScalarUDF for Unsupported Models

**Use case**: Use a model not in the official Pipeline registry (e.g., XGBoost, CatBoost, or specific RandomForest configurations)

**Pattern**: UDAF trains model once, ExprScalarUDF applies predictions to all rows

**When to use this:**
- Model not supported by `Pipeline.from_instance().predict()`
- Need full control over training/prediction logic
- Want to train on full dataset without train/test split
- Building custom ensemble or calibration pipelines

```python
import xorq.api as xo
from xorq.expr.udf import agg, make_pandas_expr_udf
import xorq.expr.datatypes as dt
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Prepare data (features + target)
# Assume we have 'features_list' and 'target' defined
data = xo.examples.diamonds.fetch()
ml_ready = data.select(*features_list, target).cache()

# Step 1: Define UDAF to train model ONCE
def train_rf_model(df):
    """Train RandomForest on full dataset - executes ONCE"""
    # Extract features and target
    X = df[features_list].values
    y = df[target].values

    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_scaled, y)

    # Return serialized model + metadata
    # This gets cached - training happens only once!
    return pickle.dumps({
        'model': model,
        'scaler': scaler,
        'features': features_list,
        'r2_score': model.score(X_scaled, y),
        'n_samples': len(df)
    })

# Create training UDAF
train_udf = agg.pandas_df(
    fn=train_rf_model,
    schema=ml_ready.schema(),
    return_type=dt.binary,  # Binary for pickled object
    name="train_rf_model"
)

# Step 2: Define ExprScalarUDF to apply predictions
def predict_with_model(model_dict, df):
    """Apply trained model to each row

    IMPORTANT: model_dict is ALREADY UNPICKLED by xorq!
    Do NOT call pickle.loads() - it's already a dict
    """
    # Unpack model components
    model = model_dict['model']
    scaler = model_dict['scaler']
    features = model_dict['features']

    # Get features and transform
    X = df[features].values
    X_scaled = scaler.transform(X)

    # Predict
    predictions = model.predict(X_scaled)

    # Return as pandas Series (must match df.index!)
    return pd.Series(predictions, index=df.index)

# Create prediction UDF
predict_udf = make_pandas_expr_udf(
    computed_kwargs_expr=train_udf.on_expr(ml_ready),  # Train ONCE
    fn=predict_with_model,  # Apply to rows
    schema=ml_ready.schema(),
    return_type=dt.float64,
    name="predict_rf"
)

# Step 3: Apply predictions to full dataset
predictions_expr = ml_ready.mutate(
    predicted_price=predict_udf.on_expr(ml_ready)
)

# Step 4: Calculate metrics (e.g., residuals, deal scores)
analysis_expr = predictions_expr.mutate(
    residual=_.price - _.predicted_price,
    pct_error=((_.price - _.predicted_price) / _.predicted_price).abs(),
    deal_score=(_.predicted_price - _.price) / _.predicted_price
)

# Execute (this trains model once, then applies predictions)
results = analysis_expr.execute()

print(f"Model R² score: {train_udf.on_expr(ml_ready).execute()['r2_score']}")
print(f"Mean absolute % error: {results['pct_error'].mean():.2%}")
print(f"\nTop 5 underpriced items (best deals):")
print(results.nlargest(5, 'deal_score')[['price', 'predicted_price', 'deal_score']])
```

**Key patterns:**
- **UDAF trains once**: `agg.pandas_df()` aggregates all data and trains model once
- **ExprScalarUDF predicts**: `make_pandas_expr_udf()` applies trained model to rows
- **Always provide name parameter**: Both UDAF and ExprScalarUDF require explicit `name=` (critical!)
- **Model is auto-unpickled**: Don't call `pickle.loads()` - xorq does it
- **Return binary from UDAF**: Use `dt.binary` for pickled objects
- **Return Series from UDF**: Must have `index=df.index` for proper alignment

**Common pitfalls:**
```python
# ❌ WRONG - missing name parameter (causes unclear errors)
train_udf = agg.pandas_df(
    fn=train_rf_model,
    schema=ml_ready.schema(),
    return_type=dt.binary
    # ERROR: Missing name="train_rf_model"
)

predict_udf = make_pandas_expr_udf(
    computed_kwargs_expr=train_udf.on_expr(ml_ready),
    fn=predict_with_model,
    schema=ml_ready.schema(),
    return_type=dt.float64
    # ERROR: Missing name="predict_rf"
)

# ❌ WRONG - calling pickle.loads() twice
def predict_with_model(model_dict, df):
    model_dict = pickle.loads(model_dict)  # ERROR: already unpickled!

# ❌ WRONG - returning numpy array instead of Series
def predict_with_model(model_dict, df):
    return model.predict(X_scaled)  # ERROR: needs df.index for alignment

# ❌ WRONG - not matching schema
train_udf = agg.pandas_df(
    fn=train_rf_model,
    schema=different_schema,  # ERROR: must match input expr schema
    ...
)

# ✅ CORRECT - always provide name parameter
train_udf = agg.pandas_df(
    fn=train_rf_model,
    schema=ml_ready.schema(),
    return_type=dt.binary,
    name="train_rf_model"  # ← REQUIRED
)

predict_udf = make_pandas_expr_udf(
    computed_kwargs_expr=train_udf.on_expr(ml_ready),
    fn=predict_with_model,
    schema=ml_ready.schema(),
    return_type=dt.float64,
    name="predict_rf"  # ← REQUIRED
)

# ✅ CORRECT - proper Series return
def predict_with_model(model_dict, df):
    predictions = model.predict(X_scaled)
    return pd.Series(predictions, index=df.index)  # Proper alignment
```

**Performance notes:**
- Training happens ONCE during first execution, then cached
- Predictions computed per-row (not vectorized across rows)
- Best for moderate datasets (10K-1M rows)
- For larger datasets, consider materializing to Parquet first

**When NOT to use this pattern:**
- Model is already supported by `Pipeline.from_instance()` - use that instead
- Need distributed training - UDAF runs on single machine
- Need online predictions - this is batch-oriented

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
- `.predict()` - Get class predictions
- `.predict_proba()` - Get probability predictions
- `.decision_function()` - Get decision scores
- `.feature_importances()` - Get feature importances
- `.score_expr()` - Get deferred accuracy score
- `deferred_sklearn_metric()` - Compute any sklearn metric
- `.pipe()` - Apply fitted transformations

**Real-world examples:**
- Multi-classifier comparison (Example 1)
- Multi-metric evaluation (Example 2)
- ROC-AUC with probabilities (Example 3)
- SVM decision functions (Example 4)
- Feature importances (Example 5)
- Catalog-based ML workflows (Example 6)
- UDAF + ExprScalarUDF for unsupported models (Example 7)

**Advanced patterns:**
- `agg.pandas_df()` - Create UDAF for model training (trains once)
- `make_pandas_expr_udf()` - Create ExprScalarUDF for predictions (applies to rows)
- `computed_kwargs_expr` - Pass trained model from UDAF to ExprScalarUDF
- Model serialization via `pickle.dumps()` / auto-unpickling by xorq
