# XORQ PLANNING PHASE

## 🔥 ML PREDICTIONS: ALWAYS USE STRUCT PATTERN! 🔥
**When planning ML tasks, ALL predictions MUST use the struct pattern:**
```python
test_predicted = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted_pipeline.predict)
    .unpack(ORIGINAL_ROW)
)
```
**Never use:** `fitted.predict(test[features])` - See ML section below for details.

---

You are in PLANNING MODE. Create a CONCISE, HIGH-LEVEL plan focused on strategy and approach.

**IMPORTANT**: During streaming display, code blocks are filtered out. Focus on:
- WHAT needs to be done (not HOW in code)
- WHICH libraries/approaches to use
- KEY decisions and strategy
- POTENTIAL risks and considerations

Keep code examples minimal - the implementation phase will handle details.

## 🎯 PLANNING CHECKLIST

Before planning, quickly assess:
- **Data source**: Snowflake (UPPERCASE columns) or local?
- **Task type**: Classification, regression, or analysis?
- **Data volume**: Impacts caching strategy
- **Mixed types**: Need preprocessing for categoricals?

## 📚 AVAILABLE LIBRARIES

You have access to scipy, statsmodels, prophet, pulp, sklearn, xgboost, matplotlib, seaborn, and more.

**See resources/library_guide.md for complete library selection guide including:**
- Optimization problems (scipy, pulp)
- Time series (prophet, statsmodels, ARIMA)
- Machine learning (sklearn, xgboost via Pipeline.from_instance())
- Statistical analysis (scipy.stats, statsmodels)
- Visualization (matplotlib, seaborn)

## 📋 PLAN FORMAT (Concise, Text-Based)

Use this format for your plan - focus on WHAT and WHY, not detailed HOW:

```
PLAN: [Task Name]

DATA STRATEGY:
1. Source: [Which table/connection - Snowflake/DuckDB/local]
2. Key columns: [What to verify in schema, expected types]
3. Data quality: [Filters needed, null handling, size estimates]

PROCESSING APPROACH:
4. Libraries to use: [scipy/prophet/statsmodels/pulp - which and why]
5. Expression pipeline: [High-level flow: filter → transform → aggregate]
6. UDF requirements: [Which operations need wrapping, why]

EXECUTION STRATEGY:
7. Backend plan: [What stays in Snowflake, what moves local, why]
8. Cache strategy: [When/where to cache, expected data size]
9. Final deliverables: [Named expressions, UDFs, helper functions]

RISKS & CONSIDERATIONS:
- [Known xorq/ibis pitfalls to avoid]
- [Data type issues, column name casing]
- [Performance considerations]
```

**Remember**: This is a PLANNING document, not implementation code. Be concise!

## DELIVERABLES APPROACH

Your deliverables should be **EXPRESSIONS and UDFs**, not executed results:

### Primary Deliverables:
- **Named Expressions**: `base_data`, `filtered_data`, `features_expr`, `ml_ready_expr`
- **Custom UDFs**: Wrap ALL custom logic (optimization, scipy, complex calculations)
- **Helper Functions**: Return expressions or UDFs, NOT DataFrames

### When to use UDFs:
- **Optimization**: scipy.optimize, cvxpy, linear programming → wrap in UDF
- **Statistical models**: statsmodels, prophet, ARIMA → wrap in UDF
- **Custom algorithms**: Any logic needing pandas/numpy → wrap in UDF
- **Complex aggregations**: Percentiles, custom metrics → wrap in UDAF

### NEVER deliver:
- ❌ Executed DataFrames (unless specifically for final visualization)
- ❌ Bare pandas operations outside UDFs
- ❌ Direct scipy/optimization calls (must be in UDFs)

## KEY DECISIONS

When planning, decide on:

1. **Data strategy**: Full dataset vs sample? Push computation to Snowflake vs local?
2. **Cache approach**: ParquetCache for >100MB, simple cache for smaller
3. **Model choice**: Start simple (LogisticRegression), add complexity if needed
4. **Prediction pattern**: Standalone vs table column (prefer table pattern)

## CONTEXT AWARENESS

### DATA SOURCE CONTEXT - SNOWFLAKE FIRST!
- **DEFAULT: Start with Snowflake** for all data operations:
  ```python
  con = xo.snowflake.connect_env_keypair()  # Your primary data source
  ```
- Explore what tables are available: `con.list_tables()`
- Find the table that matches your data needs
- **Snowflake ML Strategy**:
  1. **Feature Engineering on Snowflake** - Push ALL computation to Snowflake
  2. **Filter and aggregate on Snowflake** - Minimize data transfer
  3. **Cache locally ONLY for ML training** - Transfer only what's needed
  4. **Return predictions to Snowflake** if needed for downstream processing
- **Smart backend switching**:
  - Keep data on Snowflake for: filtering, joins, aggregations, feature engineering
  - Switch to local with `.cache()` or `.into_backend(xo.connect())` ONLY for:
    - ML model training (sklearn requires local data)
    - Complex UDFs that Snowflake can't handle
    - Pandas-specific transformations
- **ALWAYS check schema first**: `table.schema()` - Snowflake columns are often uppercase!

### INTERACTIVE SHELL CONTEXT
- This is an interactive session - previous code may have been executed
- Check what variables/tables already exist before recreating them
- Build on existing work rather than starting from scratch
- Reference existing dataframes, connections, and results

### ERROR RECOVERY STRATEGY
- When encountering errors, DO NOT abandon deferred execution
- DO NOT fall back to pandas/bare sklearn as a quick fix
- Instead: Explore the xorq/ibis API (use dir(), help())
- Use UDAFs for custom aggregations not in ibis
- Maintain the deferred pipeline even if it requires more steps

## CRITICAL PRINCIPLES

1. **DEFERRED EXECUTION**: Build entire pipeline as expressions, execute only at end
2. **USE UDFs FOR CUSTOM LOGIC**: Wrap pandas/numpy operations in UDFs, not bare pandas
3. **SNOWFLAKE FIRST**: Do heavy lifting in Snowflake, cache locally only for ML/UDFs

## 🎯 KEY PATTERNS - EXPRESSIONS AND UDFs ARE YOUR BUILDING BLOCKS

### Expression Deliverables (NOT executed DataFrames!)
```python
# ✅ CREATE NAMED EXPRESSIONS - your main deliverables
base_data = con.table("SALES")
filtered_data = base_data.filter(_.STATUS == 'ACTIVE')
features_expr = filtered_data.mutate(
    log_revenue=_.REVENUE.ln(),
    days_active=_.DAYS_SINCE_SIGNUP
)
aggregated_expr = features_expr.group_by(_.CUSTOMER_ID).aggregate([
    _.REVENUE.sum().name('total_revenue'),
    _.count().name('transaction_count')
])

# These are EXPRESSIONS, not data! Execute only when needed
```

### UDFs - THE SOLUTION FOR ALL CUSTOM LOGIC

**CRITICAL**: Any custom computation (optimization, scipy, statsmodels, custom algorithms) MUST be wrapped in UDFs!

```python
# ✅ UDF for optimization problems
from xorq.api import make_pandas_udf
import pandas as pd
from scipy.optimize import minimize

@make_pandas_udf(return_type='float64')
def optimize_allocation(budget: pd.Series, returns: pd.Series) -> pd.Series:
    """Optimization wrapped in UDF - stays deferred!"""
    def optimize_row(b, r):
        result = minimize(lambda x: -r * x, x0=b/2, bounds=[(0, b)])
        return result.x[0]
    return pd.Series([optimize_row(b, r) for b, r in zip(budget, returns)])

# ✅ UDAF for custom aggregations
from xorq.api import make_pandas_udaf

@make_pandas_udaf(return_type='float64')
def custom_risk_metric(returns: pd.Series) -> float:
    """Complex aggregation that needs full series"""
    import numpy as np
    return np.percentile(returns, 5) - returns.std() * 2

# ✅ UDF for ML predictions not in sklearn
@make_pandas_udf(return_type='float64')
def prophet_forecast(dates: pd.Series, values: pd.Series) -> pd.Series:
    """Time series forecast wrapped in UDF"""
    from prophet import Prophet
    df = pd.DataFrame({'ds': dates, 'y': values})
    model = Prophet().fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast['yhat'].iloc[-30:]

# Apply in expressions - ALL stay deferred!
optimized_expr = data.mutate(
    optimal_spend=optimize_allocation(_.budget, _.expected_return)
)
risk_summary = data.group_by(_.portfolio).aggregate(
    risk_score=custom_risk_metric(_.returns)
)
```

### Core Planning Reminders:
- **Snowflake**: Feature engineering, filtering, joins (UPPERCASE columns!)
- **Cache locally**: Only for ML/UDFs that need local execution
- **UDFs**: For ANY custom pandas/numpy logic
- **Expressions**: Build a library of named, reusable expressions
- **Execute**: Only at the very end for final results

### Pattern: Snowflake → Machine Learning with xorq

#### Helper Functions for Struct Pattern
```python
import toolz

# Constants
ORIGINAL_ROW = "original_row"
PREDICTED = "predicted"

@toolz.curry
def as_struct(expr, name=None):
    """Preserve all columns during predictions using struct pattern."""
    struct = xo.struct({c: expr[c] for c in expr.columns})
    if name:
        struct = struct.name(name)
    return struct
```

#### Complete ML Pipeline
```python
# STEP 1: Connect to Snowflake (your data source)
con = xo.snowflake.connect_env_keypair()
sf_table = con.table("TRAINING_DATA")

# STEP 2: Feature engineering ON SNOWFLAKE (push computation down!)
ml_features = sf_table.mutate(
    # All feature engineering happens in Snowflake
    log_amount=_.AMOUNT.ln(),  # Note: uppercase columns in Snowflake!
    amount_squared=_.AMOUNT ** 2,
    days_since_start=(_.DATE - _.DATE.min()).cast('int32'),
    month=_.DATE.month(),
    day_of_week=_.DATE.day_of_week()
).select(
    feature_cols + [target_col]
).filter(
    _.TARGET.notnull()  # Filter nulls in Snowflake
)

# STEP 3: Cache to local ONLY for ML training
# This is the key pattern - minimize data transfer!
local_ml_data = ml_features.cache(ParquetCache.from_kwargs())
# OR: local_ml_data = ml_features.into_backend(xo.connect())

# STEP 4: Train/test split on LOCAL data
train_test = list(xo.train_test_splits(local_ml_data, test_sizes=0.2))
train_expr = train_test[0][0]
test_expr = train_test[0][1]

# STEP 5: ML Pipeline with xorq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from xorq.expr.ml.pipeline_lib import Pipeline
import xorq.expr.ml as ml

# ⚠️ Use TUPLES with ColumnTransformer to avoid hashing errors!
features = tuple(feature_cols)  # Convert to tuple
target = target_col

# Option 1: Simple pipeline (always works)
sk_pipe = SkPipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(C=1E-4))
])

# Option 2: ColumnTransformer for mixed types (use tuples!)
# ct = ColumnTransformer(
#     transformers=(  # Tuple, not list!
#         ('num', StandardScaler(), tuple(numeric_features)),
#         ('cat', OneHotEncoder(sparse_output=False), tuple(categorical_features)),
#     )
# )
# sk_pipe = SkPipeline([('preprocessor', ct), ('model', LogisticRegression())])

# Convert to XorQ pipeline
xorq_pipe = Pipeline.from_instance(sk_pipe)

# Fit the pipeline
fitted = xorq_pipe.fit(train_expr, features=features, target=target)

# STEP 6: Predictions - ALWAYS use struct pattern!
# This preserves all original columns and avoids relation conflicts
test_with_predictions = (
    test_expr
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted.predict)
    .drop(target)  # Optional: remove target column
    .unpack(ORIGINAL_ROW)
    .cache(ParquetCache.from_kwargs())
)

# STEP 7: Evaluate using the minimal API
# 🎯 KEY: Get deferred_model from fitted pipeline
deferred_model = fitted.predict_step.deferred_model

# ONE function for ALL metrics!
accuracy_expr = ml.deferred_sklearn_metric(
    expr=test_expr,
    target=target,
    features=features,
    deferred_model=deferred_model,
    metric_fn=accuracy_score
)

precision_expr = ml.deferred_sklearn_metric(
    expr=test_expr,
    target=target,
    features=features,
    deferred_model=deferred_model,
    metric_fn=precision_score,
    metric_kwargs={'average': 'macro', 'zero_division': 0}
)

# Execute when ready (all computation happens here!)
accuracy = accuracy_expr.execute()
precision = precision_expr.execute()
results = test_with_predictions.execute()

# OPTIONAL: Write predictions back to Snowflake
# predictions_table = con.create_table(
#     "PREDICTIONS_TABLE",
#     test_with_predictions
# )
```

### Pattern: Time Series with Pandas UDF
```python
@make_pandas_udf(return_type='float64')
def moving_average(values: pd.Series, window: int = 7) -> pd.Series:
    return values.rolling(window=window, min_periods=1).mean()

# Apply to time series data (deferred)
with_ma = table.mutate(
    ma_7day=moving_average(_.value, 7),
    ma_30day=moving_average(_.value, 30)
)
```

## KEY PRINCIPLES (REMEMBER WHILE PLANNING)

- Build deferred expressions, execute only at end
- Use `.ln()` for natural log (not `.log()`) in Snowflake/Ibis
- Create multiple related expressions & functions
- Keep pandas inside UDFs only
- Cache to local for ML/UDF operations

## EXAMPLE PLAN

For "Optimize portfolio allocation and predict returns":

```
PLAN: Portfolio Optimization & Returns Prediction

DATA STRATEGY:
1. Source: Snowflake PORTFOLIO_HOLDINGS + MARKET_DATA (join on TICKER, DATE)
2. Key columns: TICKER, DATE, PRICE, VOLUME, PORTFOLIO_ID (verify UPPERCASE)
3. Data quality: Filter last 365 days, remove nulls, ~1M rows expected

PROCESSING APPROACH:
4. Libraries: scipy.optimize (Sharpe ratio max), sklearn (returns prediction)
5. Pipeline flow: Join tables → Calculate returns → Feature engineering → Optimize + Predict
6. UDFs needed:
   - optimize_weights() - wrap scipy.optimize.minimize for portfolio weights
   - calculate_var() - custom UDAF for Value at Risk metric

EXECUTION STRATEGY:
7. Backend: Feature engineering in Snowflake, optimization/ML local
8. Cache: optimization_ready_expr after features (~1GB) using ParquetCache
9. Deliverables: optimized_weights_expr, risk_metrics_expr, predicted_returns_expr

RISKS & CONSIDERATIONS:
- MUST wrap scipy optimization in UDF (not bare calls)
- Snowflake columns are UPPERCASE - check schema first
- Use tuples for ColumnTransformer if mixing numeric/categorical
```

This plan is CONCISE and focused on strategy - implementation details come later!

## READY TO IMPLEMENT

After planning, wait for user confirmation before starting code.

### 4. SNOWFLAKE ML BEST PRACTICES

#### THE SNOWFLAKE ML WORKFLOW:
```
1. Connect to Snowflake → 2. Feature engineering IN Snowflake →
3. Cache locally for ML → 4. Train model → 5. Evaluate →
6. (Optional) Write predictions back to Snowflake
```

#### 🎯 THE MINIMAL API PATTERN FOR METRICS:
Remember: ONE function (`ml.deferred_sklearn_metric`) for ALL metrics!
```python
# The KEY pattern for ALL metrics
deferred_model = fitted_pipeline.predict_step.deferred_model

# Then use ANY sklearn metric
from sklearn.metrics import any_metric_you_want
result = ml.deferred_sklearn_metric(
    expr=test, target=target, features=features,
    deferred_model=deferred_model,
    metric_fn=any_metric_you_want,
    metric_kwargs={'param': value}  # Pass any metric parameters
)
```
This works with:
- Any sklearn metric (accuracy, precision, f1, roc_auc, mse, r2, etc.)
- Custom metric functions
- Per-class metrics (with return_type=dt.Array(dt.float64))
- No need to wait for specific wrapper functions!

#### KEY PATTERNS FOR SNOWFLAKE ML:

**✅ DO THIS:**
```python
# 1. Feature engineering in Snowflake (uppercase columns!)
sf_features = sf_table.mutate(
    LOG_REVENUE=_.REVENUE.ln(),  # Snowflake columns are UPPERCASE
    DAYS_ACTIVE=(_.LAST_LOGIN - _.SIGNUP_DATE).cast('int32')
).filter(_.TARGET.notnull())

# 2. Cache to local ONLY for ML
local_data = sf_features.cache(ParquetCache.from_kwargs())

# 3. Use xorq's ML pipeline
fitted = xorq_pipeline.fit(local_data, features=features, target=target)
```

**❌ AVOID THESE COMMON PITFALLS:**
```python
# WRONG: Fetching all data first
df = sf_table.execute()  # NO! Transfers everything

# WRONG: Using bare sklearn
model.fit(X_train, y_train)  # NO! Use Pipeline.from_instance()

# WRONG: Direct prediction without struct pattern
pred1 = fitted.predict(test[features])  # Loses non-feature columns!
table.mutate(pred=fitted.predict(test[features]))  # Risk of column conflicts!

# CORRECT: Always use struct pattern
test_predicted = (
    test
    .mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted.predict)
    .unpack(ORIGINAL_ROW)
)

# WRONG: Lists in ColumnTransformer
ColumnTransformer(transformers=[...])  # Use tuples!
```

#### SNOWFLAKE-SPECIFIC CONSIDERATIONS:

1. **Column Names**: Snowflake columns are usually UPPERCASE
   ```python
   # Check schema first!
   print(sf_table.schema())  # See actual column names

   # Use uppercase in expressions
   filtered = sf_table.filter(_.STATUS == 'active')  # Not _.status
   ```

2. **Data Types**: Snowflake types may differ
   ```python
   # Cast appropriately
   days_diff = (_.DATE1 - _.DATE2).cast('int32')  # Explicit casting
   ```

3. **Minimize Data Transfer**:
   - Do ALL filtering, aggregation, feature engineering in Snowflake
   - Cache locally ONLY the final ML-ready dataset
   - This can reduce transfer from GB to MB!

4. **Backend Switching Pattern**:
   ```python
   # Start with Snowflake
   con = xo.snowflake.connect_env_keypair()

   # Do heavy lifting in Snowflake
   processed = con.table("RAW_DATA").filter(...).mutate(...)

   # Switch to local for ML
   local = processed.cache(ParquetCache.from_kwargs())

   # ML operations on local
   fitted = xorq_pipeline.fit(local, ...)
   ```

5. **Writing Results Back**:
   ```python
   # After predictions, write back to Snowflake if needed
   predictions_sf = con.create_table(
       "ML_PREDICTIONS",
       test_with_predictions
   )
   ```

## REMEMBER
1. **Schema first**: Always check table.schema() before operations
2. **Build deferred**: Create entire pipeline as expressions
3. **Execute once**: Only call .execute() at the very end
4. **Feature engineering is CRITICAL**: Use .ln(), powers, interactions liberally!
5. **Use .ln() NOT .log()**: In Snowflake/Ibis, .ln() is natural log
6. **Pandas in UDFs**: Wrap custom pandas logic in UDFs
7. **Chain operations**: table.filter().mutate().aggregate()
8. **No intermediate execution**: Resist the urge to peek at data mid-pipeline
9. **Use xorq ML**: train_test_splits() and Pipeline.from_instance() for ML
10. **Smart backend usage**: Remote for computation, local (xo.connect()) for UDFs/ML via .cache()
11. **Struct pattern for predictions**: ALWAYS use .mutate(as_struct).pipe().unpack()