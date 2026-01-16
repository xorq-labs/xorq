THREE CORE UDF PATTERNS FOR DEFERRED COMPUTATION:

⚠️ CRITICAL: UDFs only work on LOCAL backend (xo.connect())!
For remote backends (Snowflake, BigQuery, etc.), you must either:
1. Cache with ParquetCache (PREFERRED): `local_table = remote_table.cache(ParquetCache.from_kwargs())`
2. Simple cache: `local_table = remote_table.cache()`
3. Use into_backend: `local_table = remote_table.into_backend(xo.connect())`

NOTE: If you get "Function already exists" error, just recreate it - backends handle replacement.
NOTE: For operations not available in backend (like .corr() in Snowflake), use cache + pandas UDF!

## Pattern 1: Scalar UDFs (make_pandas_udf) - Row-level transformations
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

## Pattern 2: Aggregate UDFs (agg.pandas_df) - Group aggregations
```python
from xorq.expr.udf import agg

def calculate_group_stats(df):
    '''Aggregate entire group'''
    return {
        'mean': df['value'].mean(),
        'std': df['value'].std(),
        'custom_metric': (df['value'] * df['weight']).sum()
    }

schema = table.select(['value', 'weight']).schema()
stats_udf = agg.pandas_df(
    fn=calculate_group_stats,
    schema=schema,
    return_type=dt.Struct({
        'mean': dt.float64,
        'std': dt.float64,
        'custom_metric': dt.float64
    }),
    name="group_statistics"
)

# Aggregate by group (deferred!)
result = table.group_by("category").agg(stats=stats_udf.on_expr(table))
```

## Pattern 3: ExprScalarUDF - Connect pipelines (UDAF → scalar)
```python
from xorq.expr.udf import make_pandas_expr_udf

# Step 1: UDAF to compute something expensive ONCE
def fit_normalizer(df):
    '''Fit on all data'''
    scaler = StandardScaler()
    scaler.fit(df[['value']])
    return pickle.dumps({'scaler': scaler})

fit_udf = agg.pandas_df(fn=fit_normalizer, schema=schema,
                        return_type=dt.binary, name="fit")

# Step 2: Apply computed value to each row
def apply_normalizer(fitted_data, df):
    '''Apply fitted params to rows'''
    scaler = fitted_data['scaler']
    return scaler.transform(df[['value']])[:, 0]

# THE KEY PATTERN: Connect fit → transform
normalize_udf = make_pandas_expr_udf(
    computed_kwargs_expr=fit_udf.on_expr(table),  # Compute once
    fn=apply_normalizer,                          # Apply to each row
    schema=schema,
    return_type=dt.float64,
    name="normalize"
)

# Complete pipeline (ALL deferred until execute!)
result = table.mutate(normalized=normalize_udf.on_expr(table))
```

KEY: ExprScalarUDF makes ALMOST ANYTHING deferred by connecting aggregations to projections!
