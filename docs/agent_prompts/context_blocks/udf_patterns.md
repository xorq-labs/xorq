# UDF Patterns for Deferred Computation

## ⚠️ CRITICAL: UDFs Only Work on LOCAL Backend

UDFs only work on LOCAL backend (xo.connect())!
For remote backends (Snowflake, BigQuery, etc.), you must:

1. Cache with ParquetCache (PREFERRED): `local_table = remote_table.cache(ParquetCache.from_kwargs())`
2. Simple cache: `local_table = remote_table.cache()`
3. Use into_backend: `local_table = remote_table.into_backend(xo.connect())`

NOTE: If you get "Function already exists" error, just recreate it - backends handle replacement.
NOTE: For operations not available in backend (like .corr() in Snowflake), use cache + pandas UDF!

## Pattern 1: Scalar UDFs (make_pandas_udf) - Row-Level Transformations

Use for: Transforming each row independently

```python
from xorq.expr.udf import make_pandas_udf
import xorq.expr.datatypes as dt

def calculate_features(df):
    '''Transform each row'''
    result = df['price'] * df['quantity'] * (1 - df['discount'])
    return result

schema = table.select(['price', 'quantity', 'discount']).schema()
feature_udf = make_pandas_udf(
    fn=calculate_features,
    schema=schema,
    return_type=dt.float64,
    name="calculate_total"
)

# Apply to table (stays deferred!)
result = table.mutate(total=feature_udf.on_expr(table))
```

## Pattern 2: Aggregate UDFs (agg.pandas_df) - Group Aggregations

Use for: Custom aggregations, statistics, model training per group

```python
from xorq.expr.udf import agg

# Example 1: Statistical aggregations returning structs
def calculate_advanced_stats(df):
    '''Compute multiple statistics at once'''
    return {
        'iqr': df['value'].quantile(0.75) - df['value'].quantile(0.25),
        'trimmed_mean': df['value'].iloc[
            int(len(df)*0.1):int(len(df)*0.9)
        ].mean(),
        'mode': df['value'].mode().iloc[0] if not df['value'].mode().empty else None,
        'cv': df['value'].std() / df['value'].mean(),  # Coefficient of variation
        'correlation': df['value'].corr(df['other_value'])
    }

schema = table.select(['value', 'other_value']).schema()
stats_udf = agg.pandas_df(
    fn=calculate_advanced_stats,
    schema=schema,
    return_type=dt.Struct({
        'iqr': dt.float64,
        'trimmed_mean': dt.float64,
        'mode': dt.float64,
        'cv': dt.float64,
        'correlation': dt.float64
    }),
    name="advanced_stats"
)

# Apply to groups
results = table.group_by(_.category).agg(
    stats=stats_udf.on_expr(table)
)

# Example 2: Model training as aggregation
def train_segment_model(df):
    '''Train a model per group'''
    model = RandomForestRegressor()
    model.fit(df[features], df['target'])
    return pickle.dumps(model)

model_udf = agg.pandas_df(
    fn=train_segment_model,
    schema=table.select(features + ['target']).schema(),
    return_type=dt.binary,
    name="train_model"
)

# Train one model per segment
models = table.group_by("segment").agg(
    model=model_udf.on_expr(table)
)
```

## Pattern 3: ExprScalarUDF - Connect Pipelines (UDAF → Scalar)

Use for: Applying computed values (like fitted models) to each row

```python
from xorq.expr.udf import make_pandas_expr_udf
import pickle

# Step 1: UDAF to compute something expensive ONCE
def fit_normalizer(df):
    '''Fit on all data'''
    scaler = StandardScaler()
    scaler.fit(df[['value']])
    return pickle.dumps({'scaler': scaler})

fit_udf = agg.pandas_df(
    fn=fit_normalizer,
    schema=schema,
    return_type=dt.binary,
    name="fit"
)

# Step 2: Apply computed value to each row
def apply_normalizer(fitted_data, df):
    '''Apply fitted params to rows'''
    # NOTE: fitted_data is ALREADY UNPICKLED by xorq!
    scaler = fitted_data['scaler']
    return scaler.transform(df[['value']])[:, 0]

normalize_udf = make_pandas_expr_udf(
    computed_kwargs_expr=fit_udf.on_expr(table),
    fn=apply_normalizer,
    schema=schema,
    return_type=dt.float64,
    name="normalize"
)

# Apply normalization (deferred!)
normalized = table.mutate(normalized_value=normalize_udf.on_expr(table))
```

**Critical Note:** When using `computed_kwargs_expr`, the argument is **already deserialized**. Don't call `pickle.loads()`.

## UDF Decision Tree

```
What do you need?
├─ Transform each row independently? → Pattern 1: Scalar UDF (make_pandas_udf)
├─ Aggregate groups with custom logic? → Pattern 2: Aggregate UDF (agg.pandas_df)
└─ Compute once, apply to all rows? → Pattern 3: ExprScalarUDF (UDAF + make_pandas_expr_udf)
```

## Key Points

- **UDAFs reduce groups → single values** (stats, models, complex aggregations) while keeping the pipeline deferred!
- **Always cache remote data first** before applying UDFs
- **Prefer ParquetCache.from_kwargs()** for better performance
- **UDAFs let you use ANY pandas aggregation logic** within the deferred execution model!
