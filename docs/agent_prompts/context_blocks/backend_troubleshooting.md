# Backend and UDF Troubleshooting

## UDF Backend Errors

When you see "PyArrow UDFs are not supported in the <backend> backend":

```python
# Solution 1: Cache with ParquetCache for best performance
from xorq.caching import ParquetCache
local_table = remote_table.cache(ParquetCache.from_kwargs())
result = local_table.mutate(new_col=my_udf.on_expr(local_table))

# Solution 2: Use into_backend to switch backends
import xorq.api as xo
local_table = remote_table.into_backend(xo.connect())
result = local_table.mutate(new_col=my_udf.on_expr(local_table))

# Solution 3: For aggregations, cache with ParquetCache
local_data = remote_table.cache(ParquetCache.from_kwargs())
result = local_data.group_by(_.category).agg(
    custom=my_udaf.on_expr(local_data)
)
```

## Backend Operation Workarounds

When operations aren't supported in your backend (e.g., .corr() in Snowflake):

```python
from xorq.caching import ParquetCache
from xorq.expr.udf import make_pandas_expr_udf
import xorq.expr.datatypes as dt

# 1. For correlation not available in Snowflake
cached_table = table.cache(ParquetCache.from_kwargs())

@make_pandas_expr_udf(return_type=dt.float64)
def compute_correlation(df):
    return df['col1'].corr(df['col2'])

result = cached_table.mutate(
    correlation=compute_correlation(cached_table)
)

# 2. For covariance matrix computation
@make_pandas_expr_udf(return_type=dt.Struct({
    'cov_xy': dt.float64,
    'cov_xz': dt.float64,
    'cov_yz': dt.float64
}))
def compute_covariance(df):
    return pd.Series([{
        'cov_xy': df['x'].cov(df['y']),
        'cov_xz': df['x'].cov(df['z']),
        'cov_yz': df['y'].cov(df['z'])
    } for _ in range(len(df))])

cached_data = table.cache(ParquetCache.from_kwargs())
cov_results = cached_data.mutate(
    covariances=compute_covariance(cached_data)
)

# 3. For complex statistical operations
from scipy import stats

@make_pandas_expr_udf(return_type=dt.float64)
def ks_test_statistic(df):
    return stats.ks_2samp(df['sample1'], df['sample2']).statistic

cached = table.cache(ParquetCache.from_kwargs())
result = cached.mutate(ks_stat=ks_test_statistic(cached))
```

## Common UDF Error Patterns

### Error: "Can't handle RandomForestRegressor"

**Problem:** `Pipeline.from_instance()` accepts RandomForest for fitting, but `predict()` relies on a type registry that doesn't have RandomForestRegressor registered.

**Fix:** Use UDAF + ExprScalarUDF pattern instead:

```python
from xorq.expr.udf import agg, make_pandas_expr_udf
import pickle

# Train via UDAF (reduces entire dataset to pickled model)
def train_rf_model(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, df[target])
    return pickle.dumps({'model': model, 'scaler': scaler, 'features': features})

train_udf = agg.pandas_df(
    fn=train_rf_model,
    schema=train_data.select(features + [target]).schema(),
    return_type=dt.binary,
    name="train_rf"
)

# Predict via ExprScalarUDF (applies model to each row)
def predict_with_rf(model_dict, df):
    # NOTE: model_dict is ALREADY UNPICKLED by xorq! Don't call pickle.loads()
    model = model_dict['model']
    scaler = model_dict['scaler']
    X = df[model_dict['features']].values
    X_scaled = scaler.transform(X)
    return pd.Series(model.predict(X_scaled), index=df.index)

predict_udf = make_pandas_expr_udf(
    computed_kwargs_expr=train_udf.on_expr(train_data),
    fn=predict_with_rf,
    schema=train_data.schema(),
    return_type=dt.float64,
    name="predict_rf"
)

# Apply predictions
predictions_expr = table.mutate(
    PREDICTED=predict_udf.on_expr(table)
)
```

### Error: "a bytes-like object is required, not 'dict'"

**Problem:** When using `make_pandas_expr_udf` with `computed_kwargs_expr`, xorq **automatically unpickles** the binary result from the UDAF before passing it to your function.

```python
# ❌ WRONG - Double unpickling
def predict_with_model(model_data, df):
    model_dict = pickle.loads(model_data)  # ❌ ERROR: model_data is already a dict!
    # ...

# ✅ CORRECT - Use directly
def predict_with_model_fixed(model_dict, df):
    # model_dict is ALREADY unpickled by xorq
    model = model_dict['model']
    scaler = model_dict['scaler']
    # ... use model directly
```

**Rule:** When using `computed_kwargs_expr`, the argument passed to your function is **already deserialized**. Don't call `pickle.loads()`.

### Error: "KeyError: 0" during UDF struct unpacking

**Problem:** When a UDF returns a `Struct` type, you must return a `pd.Series` of dictionaries, NOT a `pd.DataFrame`.

```python
# ❌ WRONG - Returns DataFrame for Struct type
markup_udf = make_pandas_udf(
    fn=lambda df: pd.DataFrame({
        'MARKUP_30PCT': df['BASELINE_PRICE'] * 1.30,
        'OPTIMAL_MARKUP_PCT': ((df['PRICE'] / df['BASELINE_PRICE']) - 1) * 100,
    }),
    return_type=dt.Struct({
        'MARKUP_30PCT': dt.float64,
        'OPTIMAL_MARKUP_PCT': dt.float64,
    }),
    ...
)

# ✅ CORRECT - Returns Series of dicts for Struct type
def calc_markup(df):
    return pd.Series([
        {
            'MARKUP_30PCT': row['BASELINE_PRICE'] * 1.30,
            'OPTIMAL_MARKUP_PCT': ((row['PRICE'] / row['BASELINE_PRICE']) - 1) * 100,
            'REVENUE_GAP': row['PRICE'] - (row['BASELINE_PRICE'] * 1.30),
            'REVENUE_GAP_PCT': ((row['PRICE'] - (row['BASELINE_PRICE'] * 1.30)) / (row['BASELINE_PRICE'] * 1.30)) * 100
        }
        for _, row in df.iterrows()
    ])

markup_udf = make_pandas_udf(
    fn=calc_markup,
    return_type=dt.Struct({
        'MARKUP_30PCT': dt.float64,
        'OPTIMAL_MARKUP_PCT': dt.float64,
        'REVENUE_GAP': dt.float64,
        'REVENUE_GAP_PCT': dt.float64,
    }),
    schema=test_data.schema(),
    name="calc_markup"
)
```

**Rule for Struct UDFs:**
- **Scalar types** (int, float, string): Return `pd.Series`
- **Struct types**: Return `pd.Series` of dictionaries (one dict per row)
- **Never return** `pd.DataFrame` for Struct types - it will cause KeyError: 0

## Key Pattern

**Cache with ParquetCache → Apply pandas UDF with complex logic**

This maintains deferred execution while accessing operations not in backend!

REMEMBER: UDFs run locally, so cache remote data first!
PREFER: `.cache(ParquetCache.from_kwargs())` for better performance
