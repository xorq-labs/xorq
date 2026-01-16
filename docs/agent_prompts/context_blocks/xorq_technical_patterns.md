# Xorq/Ibis Expression Patterns - Known Issues & Working Patterns

## ⚠️ CRITICAL: Read before writing expressions

## IMPORTS
```python
# CORRECT
from xorq.vendor import ibis
from xorq.api import _

# WRONG
xo.vendor.ibis  # AttributeError
```

## CATEGORICAL → NUMERIC SCORING

### ❌ NEVER USE `.substitute()` for String → Integer Mapping
```python
# ❌ FAILS with XorqTypeError: Cannot compute precedence for 'int8' and 'string'
df.mutate(score=_.CUT.substitute({'Ideal': 5, 'Good': 2}))

# ❌ FAILS - This is a COMMON ERROR - DO NOT USE substitute() for type mixing!
clarity_score_map = {'SI1': 1, 'VS2': 2, 'VS1': 3}
df.mutate(clarity_score=_.CLARITY.substitute(clarity_score_map))
```

### ✅ ALWAYS USE `.case().when().else_().end()` Instead
```python
# ✅ THIS IS THE ONLY PATTERN THAT WORKS for categorical to numeric!
# Use .case().when().else_().end() - NO .cast() needed
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
    ),

    color_score=(
        _.COLOR.case()
        .when('G', 1)
        .when('F', 2)
        .when('E', 3)
        .when('D', 4)
        .else_(0)
        .end()
    ),

    cut_score=(
        _.CUT.case()
        .when('Fair', 1)
        .when('Good', 2)
        .when('Very Good', 3)
        .when('Premium', 4)
        .when('Ideal', 5)
        .else_(0)
        .end()
    )
)
```

**CRITICAL:** Whenever you need to map categorical (string) values to numeric scores:
- ❌ DO NOT use `.substitute()` - it WILL fail with type errors
- ✅ ALWAYS use `.case().when().else_().end()` pattern shown above

### ❌ BROKEN: `ibis.cases()` with Deferred
```python
# Fails: Deferred objects can't be in cases()
ibis.cases((_.CUT == 'Ideal', 5), else_=0)
```

### ❌ BROKEN: Column `.cases()` method
```python
# Despite documentation, this also fails
_.CUT.cases(('Ideal', 5), ('Good', 2), else_=0)
```

## WINDOW FUNCTIONS

### ❌ BROKEN: `row_number()` without ORDER BY (Snowflake)
```python
df.mutate(id=ibis.row_number())  # SQL compilation error
```

### ✅ WORKING: With explicit ordering
```python
df.mutate(id=ibis.row_number().over(ibis.window(order_by=_.PRICE)))
```

### ✅ ALTERNATIVE: Add IDs post-cache
```python
df = cached.execute()
df['ID'] = range(len(df))
```

## AGGREGATION FUNCTIONS

### ❌ NOT SUPPORTED: `.mode()`
```python
# OperationNotDefinedError: Compilation rule for 'Mode' not defined
df.aggregate(_.CLARITY.mode().name('most_common'))
```

### ✅ WORKAROUND: Use pandas after execute
```python
result_df = expr.execute()
most_common = result_df['CLARITY'].mode()[0]
```

### ❌ SOMETIMES PROBLEMATIC: `.nunique()`
```python
# May fail depending on backend
df.aggregate(_.CATEGORY.nunique())
```

### ✅ WORKAROUND: Count distinct
```python
df.aggregate(_.CATEGORY.distinct().count())
# Or post-execution pandas
df = expr.execute()
n_unique = df['CATEGORY'].nunique()
```

## OPTIMIZATION/ML TASKS - PROVEN 5-STEP PATTERN

This pattern has been tested and works reliably:

```python
# Step 1: Filter in Snowflake (push down predicates)
filtered = snow_con.table("DIAMONDS").filter([
    _.PRICE.between(1000, 10000),
    _.CLARITY.isin(['VS2', 'VS1', 'VVS2', 'VVS1']),
    _.COLOR.isin(['D', 'E', 'F', 'G'])
])

# Step 2: Add computed columns with .case().when().end() pattern
enriched = filtered.mutate(
    clarity_score=(
        _.CLARITY.case()
        .when('VS2', 2)
        .when('VS1', 3)
        .when('VVS2', 4)
        .when('VVS1', 5)
        .else_(0)
        .end()
    ).cast('int64'),

    price_per_carat=_.PRICE / _.CARAT
)

# Step 3: Cache locally
from xorq.caching import ParquetCache
cached = enriched.cache(ParquetCache.from_kwargs())

# Step 4: Use pandas_df aggregation UDF for complex logic
from xorq.expr.udf import agg
from xorq.vendor.ibis import dtypes as dt
import pandas as pd

def optimize_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    # Your optimization logic here
    # Return single row DataFrame with results
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

# Step 5: Post-process with pandas for unsupported operations
df = result.execute()
# Add any pandas-only operations here (mode, complex transformations, etc.)
```

## UDF PATTERNS

### ✅ WORKING: pandas_df aggregation UDF
```python
from xorq.expr.udf import agg

# Define function that takes DataFrame, returns DataFrame
def my_agg(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{'result': df['col'].sum()}])

# Create UDF
my_udf = agg.pandas_df(
    fn=my_agg,
    schema=table.schema(),
    return_type=dt.Struct({'result': dt.float64}),
    name="my_agg"
)

# Apply UDF
result = table.aggregate(output=my_udf.on_expr(table))
```

### ❌ COMMON UDF ERROR
```python
# Error: 'Table' has no attribute 'items'
# This happens when passing table instead of columns to regular UDF
@make_pandas_udf(...)
def my_udf(df): ...
my_udf(some_table)  # WRONG - pass columns explicitly
```

## DEPRECATION WARNINGS (safe to ignore but fix when possible)

```
FutureWarning: Selecting/filtering arbitrary expressions in `Table.__getitem__`
is deprecated... Please use `Table.select` or `Table.filter` instead.
```

### Fix:
```python
# OLD (deprecated but still works)
table[table.PRICE > 1000]

# NEW (preferred)
table.filter(_.PRICE > 1000)
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

## QUICK REFERENCE CARD

```python
# Connection
con = xo.snowflake.connect_env_keypair()

# Schema check (ALWAYS DO THIS FIRST)
table = con.table("MY_TABLE")
print(table.schema())

# Simple remote operations (these work reliably)
filtered = table.filter([_.PRICE < 1000, _.STATUS == 'ACTIVE'])
summary = filtered.aggregate([_.count(), _.PRICE.mean()])

# Categorical to numeric (USE THIS PATTERN)
scored = filtered.mutate(
    score=(
        _.CATEGORY.case()
        .when('A', 1)
        .when('B', 2)
        .when('C', 3)
        .else_(0)
        .end()
    ).cast('int64')
)

# Cache for complex operations
from xorq.caching import ParquetCache
cached = scored.cache(ParquetCache.from_kwargs())

# Complex aggregation with UDF
from xorq.expr.udf import agg
optimize_udf = agg.pandas_df(...)
result = cached.aggregate(solution=optimize_udf.on_expr(cached))

# Post-process with pandas
df = result.execute()
mode_value = df['column'].mode()[0]  # For unsupported operations
```

## GOLDEN RULE

**When in doubt, cache and use pandas**. The pattern that always works:
```
Remote Filter → Add Scores (.case().when()) → Cache → Pandas/UDF → Results
```