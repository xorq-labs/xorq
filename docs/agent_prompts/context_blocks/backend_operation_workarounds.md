BACKEND OPERATION WORKAROUNDS:

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
    return {
        'cov_xy': df['x'].cov(df['y']),
        'cov_xz': df['x'].cov(df['z']),
        'cov_yz': df['y'].cov(df['z'])
    }

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

KEY PATTERN: Cache with ParquetCache → Apply pandas UDF with complex logic
REMEMBER: This maintains deferred execution while accessing operations not in backend!
