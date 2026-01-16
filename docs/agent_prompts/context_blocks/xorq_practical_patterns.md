# XorQ Practical Patterns - What Actually Works

## ⚠️ REALITY CHECK - Based on Real-World Testing

### ❌ NEVER USE THESE (They WILL Fail)

1. **`.substitute()` for String → Integer mapping - THIS IS BROKEN, DO NOT USE IT**
   ```python
   # ❌ FAILS with XorqTypeError: Cannot compute precedence for 'int8' and 'string'
   # DO NOT USE .substitute() for categorical to numeric conversion!!!
   _.CLARITY.substitute({'SI1': 3, 'VS2': 4})

   # ❌ This pattern ALWAYS fails - common mistake!
   score_map = {'SI1': 1, 'VS2': 2, 'VS1': 3}
   table.mutate(score=_.CLARITY.substitute(score_map))  # WILL FAIL
   ```

   **IMPORTANT:** `.substitute()` cannot handle string keys mapping to integer values.
   NEVER use it for categorical to numeric conversion. Use `.case().when().end()` instead!

2. **`ibis.cases()` with deferred expressions**
   ```python
   # ❌ FAILS with signature validation
   ibis.cases((_.CLARITY == 'SI1', 3), ...)
   ```

3. **Column `.cases()` method**
   ```python
   # ❌ ALSO FAILS despite documentation
   _.CUT.cases(('Ideal', 5), ('Good', 2), else_=0)
   ```

4. **`ibis.row_number()` without ORDER BY**
   ```python
   # ❌ FAILS in Snowflake
   table.mutate(id=ibis.row_number())
   ```

### ✅ USE THESE PATTERNS INSTEAD

## Pattern 0: Categorical to Numeric - `.case().when().else_().end()`

**When to use**: Converting categorical values to numeric scores

**CRITICAL:** DO NOT use `.substitute()` - it WILL fail. Use `.case().when().end()` instead.

```python
# ✅ THIS IS THE ONLY WORKING PATTERN for categorical → numeric
# DO NOT use .substitute() - it fails with type errors
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

**Remember:** Always use `.case().when().else_().end()` - NEVER `.substitute()` for this!

## Pattern 1: Filter Remote → Cache → Transform Local

**When to use**: Complex transformations, scoring, feature engineering

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

## Pattern 2: Simple Aggregations Stay Remote

**When to use**: Basic statistics, grouping, counting

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

## Pattern 3: UDFs Only After Caching

**When to use**: Custom calculations, ML scoring, row-wise transformations

```python
# ❌ DON'T try complex UDFs on remote data
# ✅ DO cache first, then apply UDFs locally

cached = remote_table.cache(ParquetCache.from_kwargs())

@make_pandas_udf(...)
def custom_calculation(df):
    # Your pandas logic here
    return result

result = cached.mutate(new_col=custom_calculation(cached))
```

## Pattern 4: UDAFs for Single-Row Outputs

**When to use**: When you need to aggregate multiple rows into ONE output row

```python
# UDAFs = User Defined Aggregate Functions
# Use when: Multiple rows → Single row result (like sum, mean, optimal selection)

from xorq.types import make_pandas_udaf

# Example: Find optimal portfolio from multiple items
@make_pandas_udaf(
    input_schema={'PRICE': 'float64', 'QUALITY': 'float64', 'ID': 'string'},
    output_schema={'selected_ids': 'string', 'total_value': 'float64'}
)
def optimize_portfolio(df):
    # Complex optimization logic
    selected = optimization_algorithm(df)
    return pd.DataFrame([{
        'selected_ids': ','.join(selected['ID']),
        'total_value': selected['PRICE'].sum()
    }])

# Apply UDAF to get single row result
result = cached.aggregate([optimize_portfolio(cached)])
```

### UDF vs UDAF Decision Guide

```
Need to transform data?
├─ Each row → New row? → Use UDF
│   Example: df['new_col'] = df['old_col'] * 2
└─ Many rows → One row? → Use UDAF
    Example: Find optimal subset, custom aggregation
```

## Pattern 5: Optimization = Pure Pandas

**When to use**: Linear programming, complex algorithms

```python
# For optimization problems:
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
3. **Use UDAFs for single-row outputs**: When aggregating many rows → one result
4. **Embrace pandas**: Once cached/executed, pandas is your friend
5. **Don't fight the API**: If it doesn't work easily, use the simpler pattern

## Quick Decision Tree

```
What do you need to do?
├─ Simple filter/select? → Use ibis directly
├─ Basic aggregation (sum/mean/count)? → Use ibis.aggregate()
├─ Complex transformation?
│   ├─ Row-wise (each row → new row)? → Cache → UDF
│   └─ Aggregate (many rows → one row)? → Cache → UDAF
└─ Optimization/ML algorithm? → Cache → Execute → Pure pandas
```

## Example: Complete Pipeline

```python
# 1. Connect and explore
con = xo.snowflake.connect_env_keypair()
table = con.table("DATA")
print(table.schema())  # ALWAYS check schema first

# 2. Simple operations in Snowflake with categorical scoring
filtered = table.filter(_.PRICE < 1000)

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

Remember: **Pragmatism > Purity**. Get the job done efficiently!