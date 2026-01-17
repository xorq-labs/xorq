# Quick Fixes for Common Errors

## Fixing Schema Errors
**The column names don't match. Check:**
1. Run `print(table.schema())` again
2. Use EXACT case from schema
3. Remember: Snowflake=UPPER, others=lower

## Fixing Import Errors
**Check available imports:**
```python
from xorq.api import _
from xorq.api import make_pandas_udf, make_pandas_expr_udf
from xorq.caching import ParquetCache
from xorq.vendor import ibis
```

## Fixing Data Errors
1. Check table exists: `con.list_tables()`
2. Verify schema: `table.schema()`
3. Check data types match operations

## Fixing Attribute Errors
**When you encounter AttributeError, the operation may not be available in the backend.**

### Solution: Cache and Use UDF
```python
from xorq.caching import ParquetCache
from xorq.expr.udf import make_pandas_expr_udf
import xorq.expr.datatypes as dt

# Cache to local backend
cached = table.cache(ParquetCache.from_kwargs())

# Use pandas operations in UDF
@make_pandas_expr_udf(return_type=dt.float64)
def compute_correlation(df):
    return df['col1'].corr(df['col2'])

result = cached.mutate(correlation=compute_correlation(cached))
```

### Common Missing Operations
- `.corr()` - Not in Snowflake, use UDF
- `.cov()` - Not in most backends, use UDF
- Complex aggregations - Use UDAF pattern (see udaf_aggregation_patterns.md)

**REMEMBER:** Never fall back to `df = table.execute()` + pandas. Always use UDF patterns to maintain deferred execution.

For detailed exploration techniques, see repl_exploration.md
