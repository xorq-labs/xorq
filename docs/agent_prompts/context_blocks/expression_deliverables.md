EXPRESSION-BASED DELIVERABLES - CORE PHILOSOPHY:

## YOUR DELIVERABLES ARE EXPRESSION SETS & FUNCTION TOOLKITS

### 1. ALWAYS DELIVER MULTIPLE EXPRESSIONS
Never deliver just one expression. Create a suite of related expressions that build on each other:

```python
# ✅ GOOD: Expression suite
# Base expression with initial filtering
clean_data = table.filter(_.status.notnull())

# Feature-engineered expression
featured_data = clean_data.mutate(
    log_value=_.value.ln(),  # Natural log in Snowflake/Ibis
    value_squared=_.value ** 2,
    normalized_value=(_.value - _.value.mean()) / _.value.std()
)

# Multiple aggregation expressions
overall_stats = featured_data.aggregate([
    _.value.mean().name('mean'),
    _.value.std().name('std'),
    _.value.quantile(0.5).name('median')
])

grouped_stats = featured_data.group_by(_.category).aggregate([
    _.value.sum().name('total'),
    _.count().name('count'),
    _.value.mean().name('avg')
])

time_series = featured_data.group_by(_.date.truncate('M')).aggregate([
    _.value.sum().name('monthly_total')
])

# ❌ BAD: Single expression
result = table.filter(_.status.notnull()).execute()  # Too simple, immediately executed
```

### 2. CREATE FUNCTION TOOLKITS
Build collections of functions that work with expressions:

```python
# ✅ GOOD: Function toolkit
def prepare_data(expr, date_col='date', value_col='value'):
    """Prepare and clean data expression"""
    return expr.filter(
        (_[value_col].notnull()) &
        (_[date_col] >= '2020-01-01')
    )

def add_features(expr, value_col='value'):
    """Add engineered features to expression"""
    return expr.mutate(
        log_value=_[value_col].ln(),  # Natural log in Snowflake/Ibis
        value_pct_change=(_[value_col] - _[value_col].shift(1)) / _[value_col].shift(1),
        rolling_mean=_[value_col].mean().over(ibis.window(preceding=7))
    )

def create_aggregations(expr, group_cols, value_col):
    """Generate multiple aggregation expressions"""
    return {
        'summary': expr.aggregate([
            _[value_col].mean().name('mean'),
            _[value_col].std().name('std')
        ]),
        'grouped': expr.group_by(*group_cols).aggregate([
            _[value_col].sum().name('total'),
            _.count().name('count')
        ]),
        'top_n': expr.order_by(_[value_col].desc()).limit(10)
    }

def visualize_trends(expr, date_col, value_col, title="Trend Analysis"):
    """Visualize trends from expression"""
    data = expr.select(date_col, value_col).execute()
    # Plotting logic here

# ❌ BAD: Single function that does everything
def analyze_data(table):
    df = table.execute()
    # All logic crammed into one function
    return df  # Returns DataFrame, not expression
```

### 3. EXPRESSION + FUNCTION COMBINATIONS
Deliver both expressions AND functions that operate on them:

```python
# ✅ GOOD: Complete analytical toolkit

# Core expressions
base_expr = table.filter(_.active == True)
daily_expr = base_expr.group_by(_.date).aggregate([
    _.amount.sum().name('daily_total'),
    _.count().name('daily_count')
])
monthly_expr = base_expr.mutate(
    month=_.date.truncate('M')
).group_by(_.month).aggregate([
    _.amount.sum().name('monthly_total')
])

# Analysis functions that work with the expressions
def compare_periods(expr, period1_start, period1_end, period2_start, period2_end):
    """Compare metrics between two time periods"""
    period1 = expr.filter((_.date >= period1_start) & (_.date <= period1_end))
    period2 = expr.filter((_.date >= period2_start) & (_.date <= period2_end))

    return {
        'period1_stats': period1.aggregate([_.amount.sum(), _.count()]),
        'period2_stats': period2.aggregate([_.amount.sum(), _.count()]),
        'growth_rate': ((period2.amount.sum() - period1.amount.sum()) / period1.amount.sum())
    }

def detect_anomalies(expr, value_col, threshold=2):
    """Find anomalies using statistical methods"""
    stats = expr.aggregate([
        _[value_col].mean().name('mean'),
        _[value_col].std().name('std')
    ])

    return expr.mutate(
        z_score=(_[value_col] - stats['mean']) / stats['std'],
        is_anomaly=(_.z_score.abs() > threshold)
    ).filter(_.is_anomaly == True)
```

### 4. ML PIPELINE DELIVERABLES
For ML tasks, deliver complete pipeline components:

```python
# ✅ GOOD: Complete ML toolkit

# Data preparation expressions
ml_features = base_table.select(feature_cols + [target_col])
ml_clean = ml_features.filter(_.target.notnull())
ml_engineered = add_polynomial_features(ml_clean, degree=2)

# Split generation function
def create_splits(expr, test_size=0.2, val_size=0.1):
    """Create train/val/test splits as expressions"""
    train_val, test = list(xo.train_test_splits(expr, test_sizes=test_size))[0]
    train, val = list(xo.train_test_splits(train_val, test_sizes=val_size/(1-test_size)))[0]
    return {'train': train, 'val': val, 'test': test}

# Model pipeline function
def build_pipeline(features_list):
    """Create reusable model pipeline"""
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from xorq.expr.ml.pipeline_lib import Pipeline

    sk_pipeline = SkPipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100))
    ])

    return Pipeline.from_instance(sk_pipeline)

# Evaluation function
def evaluate_model(pipeline, test_expr, features, target):
    """Generate evaluation metrics as expressions"""
    import xorq.expr.ml as ml
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    metrics = {}
    for name, func in [('accuracy', accuracy_score),
                       ('precision', precision_score),
                       ('recall', recall_score)]:
        metrics[name] = ml.deferred_sklearn_metric(
            expr=test_expr,
            target=target,
            features=features,
            deferred_model=pipeline.predict_step.deferred_model,
            metric_fn=func,
            metric_kwargs={'average': 'weighted'} if name != 'accuracy' else {}
        )
    return metrics
```

### 5. DOCUMENTATION & USAGE EXAMPLES
Always include examples showing how to use your deliverables:

```python
# Usage examples for your deliverables
"""
Usage Examples:
==============

1. Basic Analysis:
   # Use the prepared expressions
   result = overall_stats.execute()
   print(f"Mean value: {result['mean'][0]}")

2. Custom Time Period:
   # Apply functions to expressions
   q1_data = featured_data.filter((_.date >= '2024-01-01') & (_.date < '2024-04-01'))
   q1_stats = create_aggregations(q1_data, ['category'], 'value')

3. Combining Components:
   # Chain expressions and functions
   prepared = prepare_data(base_expr)
   featured = add_features(prepared)
   anomalies = detect_anomalies(featured, 'value')

4. ML Pipeline:
   # Use the complete ML toolkit
   splits = create_splits(ml_engineered)
   pipeline = build_pipeline(feature_cols)
   fitted = pipeline.fit(splits['train'], features=feature_cols, target='target')
   metrics = evaluate_model(fitted, splits['test'], feature_cols, 'target')
"""
```

## KEY PRINCIPLES

### DO:
- ✅ Create multiple related expressions
- ✅ Build function toolkits that work together
- ✅ Keep everything deferred until execution is needed
- ✅ Provide clear usage examples
- ✅ Make components composable and reusable

### DON'T:
- ❌ Deliver single expressions or functions
- ❌ Execute prematurely (.execute() only for visualization/final output)
- ❌ Create monolithic scripts
- ❌ Return pandas DataFrames from functions (return expressions)
- ❌ Forget to show how components work together

### REMEMBER:
Your goal is to deliver a complete, composable toolkit of expressions and functions that users can:
1. Use immediately for their task
2. Modify and extend for related tasks
3. Combine in different ways for new analyses
4. Execute only when they need final results