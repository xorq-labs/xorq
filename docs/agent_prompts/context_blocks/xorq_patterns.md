# XorQ Patterns - Working Patterns and Common Pitfalls

## ⚠️ CRITICAL: Read Before Writing Expressions

## Imports
```python
# CORRECT
from xorq.vendor import ibis
from xorq.api import _

# WRONG
xo.vendor.ibis  # AttributeError
```

## ❌ NEVER USE THESE (They WILL Fail)

### 1. `.substitute()` for String → Integer Mapping
```python
# ❌ FAILS with XorqTypeError: Cannot compute precedence for 'int8' and 'string'
_.CLARITY.substitute({'SI1': 3, 'VS2': 4})
score_map = {'SI1': 1, 'VS2': 2, 'VS1': 3}
table.mutate(score=_.CLARITY.substitute(score_map))  # WILL FAIL
```

**CRITICAL:** `.substitute()` cannot handle string keys mapping to integer values. NEVER use it for categorical to numeric conversion!

### 2. `ibis.cases()` with Deferred Expressions
```python
# ❌ FAILS with signature validation
ibis.cases((_.CLARITY == 'SI1', 3), ...)
```

### 3. Column `.cases()` Method
```python
# ❌ ALSO FAILS despite documentation
_.CUT.cases(('Ideal', 5), ('Good', 2), else_=0)
```

### 4. `ibis.row_number()` Without ORDER BY
```python
# ❌ FAILS in Snowflake - SQL compilation error
table.mutate(id=ibis.row_number())
```

### 5. Unsupported Aggregations
```python
# ❌ NOT SUPPORTED: .mode(), sometimes .nunique()
df.aggregate(_.CLARITY.mode().name('most_common'))
```

## ✅ WORKING PATTERNS

### Pattern: Categorical to Numeric - `.case().when().else_().end()`

**Use this for:** Converting categorical values to numeric scores

```python
# ✅ THIS IS THE ONLY WORKING PATTERN for categorical → numeric
enriched = filtered.mutate(
    clarity_score=(
        _.CLARITY.case()
        .when('SI1', 1)
        .when('VS2', 2)
        .when('VS1', 3)
        .when('VVS2', 4)
        .when('VVS1', 5)
        .when('IF', 6)
        .when('FL', 7)
        .else_(0)
        .end()
    ).cast('int64'),

    color_score=(
        _.COLOR.case()
        .when('G', 1)
        .when('F', 2)
        .when('E', 3)
        .when('D', 4)
        .else_(0)
        .end()
    ).cast('int64')
)
```

### Pattern: Filter Remote → Cache → Transform Local

**Use this for:** Complex transformations, scoring, feature engineering

```python
# Step 1: Simple filtering in Snowflake (works reliably)
filtered = table.filter([
    _.COLUMN.isin(['value1', 'value2']),
    _.PRICE <= 10000,
    _.AMOUNT > 0
])

# Step 2: Cache locally
from xorq.caching import ParquetCache
cached = filtered.cache(ParquetCache.from_kwargs())

# Step 3: Execute and use pandas for complex ops
df = cached.execute()
df['NEW_COLUMN'] = df['COL1'] / df['COL2']
df['SCORE'] = df['COL1'].map(mapping_dict)
```

### Pattern: Simple Aggregations Stay Remote

**Use this for:** Basic statistics, grouping, counting

```python
# These work well in Snowflake
summary = table.aggregate([
    _.COLUMN.count().name('count'),
    _.PRICE.mean().name('avg_price'),
    _.AMOUNT.sum().name('total')
])

# Group by also works well
by_category = table.group_by(_.CATEGORY).aggregate([
    _.count().name('n'),
    _.PRICE.min().name('min_price')
]).order_by(ibis.desc('n'))
```

### Pattern: Window Functions with Explicit Ordering

```python
# ✅ WORKING: With explicit ordering
df.mutate(id=ibis.row_number().over(ibis.window(order_by=_.PRICE)))

# ✅ ALTERNATIVE: Add IDs post-cache
df = cached.execute()
df['ID'] = range(len(df))
```

### Pattern: UDAFs for Single-Row Outputs

**Use this when:** Aggregating multiple rows into ONE output row

```python
from xorq.expr.udf import agg
from xorq.vendor.ibis import dtypes as dt
import pandas as pd

def optimize_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    # Your optimization logic here
    selected = optimization_algorithm(df)
    return pd.DataFrame([{
        'selected_ids': ','.join(selected['ID'].astype(str)),
        'total_value': selected['PRICE'].sum(),
        'avg_quality': selected['quality_score'].mean()
    }])

optimize_udf = agg.pandas_df(
    fn=optimize_portfolio,
    schema=cached.schema(),
    return_type=dt.Struct({
        'selected_ids': dt.string,
        'total_value': dt.float64,
        'avg_quality': dt.float64
    }),
    name="optimize"
)

result = cached.aggregate(solution=optimize_udf.on_expr(cached))
```

### UDF vs UDAF Decision Guide

```
Need to transform data?
├─ Each row → New row? → Use UDF
│   Example: df['new_col'] = df['old_col'] * 2
└─ Many rows → One row? → Use UDAF
    Example: Find optimal subset, custom aggregation
```

### Pattern: Optimization = Pure Pandas

**Use this for:** Linear programming, complex algorithms

```python
# 1. Filter and cache
cached = table.filter(conditions).cache(ParquetCache.from_kwargs())

# 2. Execute to pandas
df = cached.execute()

# 3. Use scipy/sklearn/custom algorithms
from scipy.optimize import linprog
result = linprog(c=costs, A_ub=constraints, ...)
```

## GOLDEN RULES

1. **Keep Snowflake operations simple**: filter, select, basic aggregates
2. **Cache before complexity**: Always cache before UDFs or complex transforms
3. **Use `.case().when().end()` for categorical scoring**: NEVER use `.substitute()`
4. **Embrace pandas**: Once cached/executed, pandas is your friend
5. **Don't fight the API**: If it doesn't work easily, use the simpler pattern

## Quick Decision Tree

```
What do you need to do?
├─ Simple filter/select? → Use ibis directly
├─ Basic aggregation (sum/mean/count)? → Use ibis.aggregate()
├─ Categorical to numeric? → Use .case().when().end()
├─ Complex transformation?
│   ├─ Row-wise (each row → new row)? → Cache → UDF
│   └─ Aggregate (many rows → one row)? → Cache → UDAF
└─ Optimization/ML algorithm? → Cache → Execute → Pure pandas
```

## DEBUGGING CHECKLIST

Before building complex expressions:

1. ☐ **Schema check first**: `print(table.schema())`
2. ☐ **Test on small sample**: `table.limit(5).execute()`
3. ☐ **Categorical scoring**: Use `.case().when().end().cast()` NOT `.substitute()`
4. ☐ **Window functions**: Add `.over(ibis.window(order_by=...))`
5. ☐ **Avoid in aggregates**: `.mode()`, `.nunique()` → use pandas post-execute
6. ☐ **Complex logic**: Cache first, then use pandas_df UDF or pure pandas

## ERROR PATTERN RECOGNITION

### Type Coercion Issues
```
Error: "Cannot compute precedence for int8 and string types"
Solution: Use .case().when().end().cast('int64') pattern
```

### Deferred Object Issues
```
Error: "Deferred" in error message
Solution: Can't use deferred expressions in that context
```

### Window Function Issues
```
Error: "Window function requires ORDER BY"
Solution: Add .over(ibis.window(order_by=...))
```

### Compilation Not Defined
```
Error: "OperationNotDefinedError: Compilation rule for 'Mode' not defined"
Solution: Operation not supported, use pandas after .execute()
```

### Backend Compatibility
```
Error: "XorqException" or "SnowflakeException"
Solution: Operation not supported on remote backend, cache first
```

## Complete Example Pipeline

```python
# 1. Connect and explore
con = xo.snowflake.connect_env_keypair()
table = con.table("DATA")
print(table.schema())  # ALWAYS check schema first

# 2. Simple operations in Snowflake with categorical scoring
filtered = table.filter([
    _.PRICE.between(1000, 10000),
    _.CLARITY.isin(['VS2', 'VS1', 'VVS2'])
])

# Add numeric scores using .case().when().end() pattern
scored = filtered.mutate(
    category_score=(
        _.CATEGORY.case()
        .when('Premium', 3)
        .when('Standard', 2)
        .when('Basic', 1)
        .else_(0)
        .end()
    ).cast('int64')
)

summary = scored.aggregate([_.count(), _.category_score.mean()])

# 3. Cache for complex work
cached = scored.cache(ParquetCache.from_kwargs())

# 4. Complex operations in pandas
df = cached.execute()
df['complex_score'] = custom_scoring_function(df)
optimal_selection = optimization_algorithm(df)
```

**Remember:** Pragmatism > Purity. Get the job done efficiently!
