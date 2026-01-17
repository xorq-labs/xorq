# Quick Reference - Common Fixes and Patterns

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
- Complex aggregations - Use UDAF pattern

**REMEMBER:** Never fall back to `df = table.execute()` + pandas. Always use UDF patterns to maintain deferred execution.

## Optimization Patterns

### Optimization Approach
1. Define constraints clearly
2. Build solution incrementally
3. Track objective function
4. Validate constraints at each step

### For Linear Programming/Complex Algorithms
```python
# 1. Filter and cache
cached = table.filter(conditions).cache(ParquetCache.from_kwargs())

# 2. Execute to pandas
df = cached.execute()

# 3. Use scipy/sklearn/custom algorithms
from scipy.optimize import linprog
result = linprog(c=costs, A_ub=constraints, ...)
```

## Plotting Patterns

- Use matplotlib/seaborn in the notebook; call `plt.show()` once per figure
- Keep datasets small for plotting: `expr.limit()` or summary tables
- Label axes and add titles that explain the insight
- Prefer clear chart types (line, bar, scatter) unless the task demands more
- Mention any important parameters (bins, colors) in markdown to aid interpretation

## Result Summary Patterns

- Recap the objective in one sentence
- Highlight the key quantitative findings or metrics
- Call out limitations, assumptions, or next steps
- Reference the code cells/plots that support each claim
- Close with a clear recommendation or follow-up action

## Quick Checks Before Building

1. ☐ **Schema check first**: `print(table.schema())`
2. ☐ **Test on small sample**: `table.limit(5).execute()`
3. ☐ **Check operation support**: If unsure, cache first
4. ☐ **Verify imports**: Use `from xorq.vendor import ibis`

For detailed exploration techniques, see repl_exploration.md
