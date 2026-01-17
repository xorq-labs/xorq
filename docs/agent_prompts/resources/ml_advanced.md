# ML Advanced Patterns

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

### Advanced Metric Examples

```python
import xorq.expr.ml as ml
import xorq.expr.datatypes as dt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Get the deferred model from the fitted pipeline
deferred_model = fitted_pipeline.predict_step.deferred_model

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

# If using predictions column
accuracy_from_col = ml.deferred_sklearn_metric(
    expr=test_with_predictions,
    target=target,
    predictions_col="predictions",  # Use column instead of deferred_model
    metric_fn=accuracy_score
)

# Custom metrics
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
    'per_class': per_class_precision.execute(),
    'roc_auc': roc_auc.execute(),
    'custom': custom_metric.execute()
}
```
