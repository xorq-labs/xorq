# Expression API

## Overview

Xorq's expression API provides a fluent interface for building data transformation pipelines that compile to efficient backend queries.

---

## Core Expression Patterns

### 1. Fluent Data Transformations

**Pattern**: Chain operations to build expression trees

```python
import xorq.api as xo

# Connect to backend
con = xo.connect()  # defaults to embedded DuckDB

# Build expression pipeline
expr = (
    xo.examples.iris.fetch(backend=con)
    .filter([xo._.sepal_length > 5])
    .group_by("species")
    .agg(xo._.sepal_width.sum())
    .select("species", "sepal_width_sum")
)

# Execute when ready
result = expr.execute()
```

**Key APIs:**
- `xo._` - Column reference (like ibis `_`)
- `.filter()` - Row filtering with predicates
- `.select()` - Column selection/projection
- `.mutate()` - Add/modify columns
- `.group_by()` + `.agg()` - Aggregations
- `.join()` - Table joins
- `.execute()` - Materialize results

---

### 2. Column References with xo._

**Pattern**: Use `xo._` for column references in expressions

```python
# Simple column reference
xo._.column_name

# Comparison
xo._.age > 18

# Multiple conditions
[xo._.age > 18, xo._.status == "active"]

# Method chaining
xo._.price.mean()
xo._.name.upper()
xo._.timestamp.date()
```

**Common operations:**
- Arithmetic: `xo._.a + xo._.b`, `xo._.x * 2`
- Comparison: `xo._.x > 5`, `xo._.name == "test"`
- String: `xo._.text.upper()`, `xo._.text.contains("pattern")`
- Null checks: `xo._.value.isnull()`, `xo._.value.notnull()`
- Set membership: `xo._.status.isin(["active", "pending"])`

---

### 3. Filtering

**Pattern**: Filter rows using predicates

```python
# Single condition
filtered = data.filter(xo._.age > 18)

# Multiple conditions (AND)
filtered = data.filter([
    xo._.age > 18,
    xo._.status == "active"
])

# Complex conditions
filtered = data.filter(
    (xo._.age > 18) & (xo._.status == "active") |
    (xo._.role == "admin")
)

# String operations
filtered = data.filter(xo._.name.contains("Smith"))

# Set membership
filtered = data.filter(xo._.category.isin(["A", "B", "C"]))

# Date filtering
filtered = data.filter(xo._.timestamp >= "2024-01-01")
```

---

### 4. Selection and Projection

**Pattern**: Choose columns to include in result

```python
# Select specific columns
selected = data.select("id", "name", "age")

# Select with expressions
selected = data.select(
    "id",
    "name",
    age_group=xo._.age // 10
)

# Deselect columns
selected = data.drop("internal_id", "temp_col")

# Select all except
selected = data.select(~xo.s.cols("temp_*"))

# Rename columns
renamed = data.relabel({"old_name": "new_name"})
```

---

### 5. Mutation (Adding/Modifying Columns)

**Pattern**: Add or modify columns with `.mutate()`

```python
# Add new column
mutated = data.mutate(age_squared=xo._.age ** 2)

# Multiple columns
mutated = data.mutate(
    age_squared=xo._.age ** 2,
    age_cubed=xo._.age ** 3,
    full_name=xo._.first_name + " " + xo._.last_name
)

# Conditional mutation
mutated = data.mutate(
    age_category=xo.case()
        .when(xo._.age < 18, "minor")
        .when(xo._.age < 65, "adult")
        .else_("senior")
        .end()
)

# Replace existing column
mutated = data.mutate(price=xo._.price * 1.1)
```

---

### 6. Aggregations

**Pattern**: Group and aggregate data

```python
# Simple aggregation
agg = data.group_by("category").agg(
    count=xo._.id.count(),
    total=xo._.value.sum(),
    avg=xo._.value.mean()
)

# Multiple grouping columns
agg = data.group_by(["category", "region"]).agg(
    count=xo._.id.count()
)

# Without grouping (aggregate all)
agg = data.agg(
    total_count=xo._.id.count(),
    avg_value=xo._.value.mean()
)

# Complex aggregations
agg = data.group_by("category").agg(
    count=xo._.id.count(),
    total=xo._.value.sum(),
    avg=xo._.value.mean(),
    min_val=xo._.value.min(),
    max_val=xo._.value.max(),
    stddev=xo._.value.std()
)
```

**Common aggregation functions:**
- `count()` - Count rows
- `sum()` - Sum values
- `mean()`, `avg()` - Average
- `min()`, `max()` - Min/max
- `std()`, `var()` - Standard deviation, variance
- `first()`, `last()` - First/last value
- `nunique()` - Count distinct
- `collect()` - Collect into array

---

### 7. Joins

**Pattern**: Combine tables

```python
# Inner join
joined = left.join(
    right,
    left.id == right.left_id,
    how="inner"
)

# Left join
joined = left.join(
    right,
    left.id == right.left_id,
    how="left"
)

# Multiple join conditions
joined = left.join(
    right,
    [left.id == right.left_id, left.type == right.type],
    how="inner"
)

# Self join
self_joined = data.join(
    data.relabel(lambda c: f"{c}_right"),
    data.id == data.id_right,
    how="inner"
)
```

**Join types:**
- `inner` - Only matching rows
- `left` - All left rows, matching right rows
- `right` - All right rows, matching left rows
- `outer` - All rows from both tables
- `semi` - Left rows with matching right rows
- `anti` - Left rows without matching right rows

---

### 8. Window Functions

**Pattern**: Apply functions over partitions

```python
# Rank within groups
ranked = data.mutate(
    rank=xo._.value.rank().over(
        group_by="category",
        order_by=xo._.value.desc()
    )
)

# Running sum
running = data.mutate(
    running_total=xo._.value.sum().over(
        group_by="category",
        order_by="timestamp"
    )
)

# Moving average
moving = data.mutate(
    moving_avg=xo._.value.mean().over(
        group_by="category",
        order_by="timestamp",
        rows=(2, 0)  # 2 preceding rows, current row
    )
)
```

---

### 9. Ordering

**Pattern**: Sort results

```python
# Single column ascending
ordered = data.order_by("age")

# Multiple columns
ordered = data.order_by(["category", "timestamp"])

# Descending
ordered = data.order_by(xo._.age.desc())

# Mixed ordering
ordered = data.order_by([
    xo._.category,
    xo._.timestamp.desc()
])
```

---

### 10. Limiting

**Pattern**: Limit result rows

```python
# Top N rows
top10 = data.limit(10)

# With offset
paginated = data.limit(10, offset=20)

# Combined with ordering
top_earners = (
    data
    .order_by(xo._.salary.desc())
    .limit(10)
)
```

---

## Deferred Execution

### Pattern: Lazy Data Loading

**Problem**: Loading large datasets into memory is expensive

**Solution**: Use deferred reading to build expressions without loading data

```python
from xorq.common.utils.defer_utils import deferred_read_parquet

# Build expression tree without executing
pipeline = (
    deferred_read_parquet(
        path="/data/input.parquet",
        connection=con,
        name="input_data"
    )
    .filter([xo._.status == "active"])
    .select("id", "value", "timestamp")
    .group_by("category")
    .agg(total=xo._.value.sum())
)

# Execute only when needed
df = pipeline.execute()
```

**Benefits:**
- No memory overhead during expression building
- Backend can optimize entire query
- Easy to compose and modify pipelines
- Execution happens once with all optimizations

**Deferred operations:**
- `deferred_read_parquet()` - Lazy parquet reading
- `deferred_read_csv()` - Lazy CSV reading
- All transformations are deferred until `.execute()`

---

## Backend Integration

### Pattern: Multi-Backend Support

**Xorq expressions work across different backends:**

```python
# Pandas in-memory
con = xo.connect()
df = pd.DataFrame({"a": [1, 2, 3]})
table = con.create_table("my_table", df)

# DuckDB (default embedded)
con = xo.connect()  # embedded DuckDB

# Snowflake
con = xo.connect("snowflake://account/database/schema")

# Postgres
con = xo.connect("postgresql://user:pass@host/db")

# DataFusion
con = xo.connect("datafusion://")

# Use same expression on any backend
expr = con.table("my_table").filter(xo._.a > 1)
result = expr.execute()
```

**Backend features:**
- Automatic query compilation per backend
- Backend-specific optimizations
- Cross-backend expression portability
- Consistent API regardless of backend

---

## Struct Operations

### Pattern: Pack/Unpack Columns

**Use case**: Preserve original data while transforming

```python
import toolz

@toolz.curry
def as_struct(expr, name=None):
    """Pack all columns into a struct"""
    struct = xo.struct({c: expr[c] for c in expr.columns})
    if name:
        struct = struct.name(name)
    return struct

# Pack columns into struct
packed = data.mutate(original=as_struct(name="original_row"))

# Transform with original preserved
transformed = packed.mutate(
    value_normalized=xo._.value / xo._.value.max()
)

# Unpack struct back to columns
unpacked = transformed.unpack("original_row")
```

**Struct patterns:**
- Pack multiple columns for preservation
- Useful in pipelines that transform column sets
- `.unpack()` to expand structs
- Pass structs to UDFs for batch processing

---

## Data Sources

### Pattern: Loading Data

```python
# Built-in examples
iris = xo.examples.iris.fetch()
penguins = xo.examples.penguins.fetch()

# From pandas DataFrame
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
table = con.create_table("my_table", df)

# From parquet (deferred)
from xorq.common.utils.defer_utils import deferred_read_parquet

parquet_data = deferred_read_parquet(
    path="data.parquet",
    connection=con,
    name="data"
)

# From CSV (deferred)
from xorq.common.utils.defer_utils import deferred_read_csv

csv_data = deferred_read_csv(
    path="data.csv",
    connection=con,
    name="data"
)

# In-memory table
mem_table = xo.memtable(
    {"col1": [1, 2, 3], "col2": ["a", "b", "c"]},
    schema=xo.schema({"col1": int, "col2": str})
)

# From existing backend table
table = con.table("existing_table")
```

---

## Type System

### Pattern: Working with Types

```python
import xorq.expr.datatypes as dt

# Basic types
dt.int64
dt.float64
dt.string
dt.boolean
dt.timestamp
dt.date

# Complex types
dt.array(dt.int64)
dt.struct({"a": dt.int64, "b": dt.string})
dt.map(dt.string, dt.int64)

# Schema creation
schema = xo.schema({
    "id": dt.int64,
    "name": dt.string,
    "value": dt.float64,
    "tags": dt.array(dt.string)
})

# Type casting
casted = data.mutate(
    value_as_int=xo._.value.cast(dt.int64)
)
```

---

## Best Practices

### 1. Build Expressions Declaratively

```python
# Good: declarative pipeline
pipeline = (
    data
    .filter(xo._.status == "active")
    .select("id", "value")
    .group_by("category")
    .agg(xo._.value.sum())
)

# Avoid: imperative loops
result = []
for row in data:
    if row.status == "active":
        result.append(row)
```

### 2. Push Filtering Early

```python
# Good: filter before expensive operations
expr = (
    data
    .filter(xo._.date >= "2024-01-01")  # Filter first
    .expensive_transformation()
    .group_by("category")
    .agg(xo._.value.sum())
)

# Avoid: filter after expensive operations
expr = (
    data
    .expensive_transformation()
    .group_by("category")
    .agg(xo._.value.sum())
    .filter(xo._.date >= "2024-01-01")  # Too late
)
```

### 3. Use Deferred Loading for Large Data

```python
# Good: deferred reading
expr = deferred_read_parquet(path, con, "data")

# Avoid: eager loading
df = pd.read_parquet(path)  # loads all into memory
expr = con.create_table("data", df)
```

### 4. Batch Operations

```python
# Good: batch transformations
expr = data.mutate(
    col1=transformation1,
    col2=transformation2,
    col3=transformation3
)

# Avoid: sequential mutations
expr = data.mutate(col1=transformation1)
expr = expr.mutate(col2=transformation2)
expr = expr.mutate(col3=transformation3)
```

### 5. Use Appropriate Backends

```python
# DuckDB for analytics on local files
con = xo.connect()

# Snowflake for warehouse data
con = xo.connect("snowflake://...")

# Postgres for transactional data
con = xo.connect("postgresql://...")
```

---

## Debugging

### Inspect Expressions

```python
# Check schema
print(expr.schema())

# Inspect operation tree
print(expr.op())

# Get column names
print(expr.columns)

# Check expression type
print(type(expr))

# Explain query plan (backend-specific)
print(expr.explain())
```

### Test Incrementally

```python
# Build and test step by step
step1 = data.filter(xo._.status == "active")
print(step1.execute())

step2 = step1.select("id", "value")
print(step2.execute())

step3 = step2.group_by("category").agg(xo._.value.sum())
print(step3.execute())
```

---

## Summary

**Key concepts:**
1. **Fluent API**: Chain operations to build pipelines
2. **Deferred execution**: Expressions compile to optimized queries
3. **Backend agnostic**: Same code works across backends
4. **Type safe**: Explicit schemas and type system
5. **Composable**: Expressions are first-class values

**Common operations:**
- Filter, select, mutate, group_by, agg, join
- Window functions, ordering, limiting
- Struct pack/unpack for data preservation
- Deferred reading for large datasets
