# POST-AGGREGATION COLUMN ACCESS

## ⚠️ CRITICAL: Cannot Use _.count, _.mean, etc. After Aggregation

After `.aggregate()` or `.group_by().aggregate()`, column names may conflict with deferred method names!

### The Problem

```python
# Create aggregated column named 'count'
grouped = table.group_by(_.category).aggregate([
    _.count().name('count'),
    _.price.mean().name('mean_price')
])

# ❌ WRONG - _.count is a METHOD, not the column!
filtered = grouped.filter(_.count > 100)
# ERROR: TypeError: '>' not supported between instances of 'method' and 'int'

# ❌ WRONG - _.mean is also a METHOD
ordered = grouped.order_by(_.mean_price.desc())
# ERROR: AttributeError: 'function' object has no attribute 'desc'
```

### The Solution

Use **bracket notation** or **string names** for aggregated columns:

```python
# ✅ CORRECT - Use bracket notation
filtered = grouped.filter(_['count'] > 100)

# ✅ CORRECT - Use string name with ibis functions
from xorq.vendor import ibis
ordered = grouped.order_by(ibis.desc('count'))

# ✅ CORRECT - Avoid naming conflicts
grouped = table.group_by(_.category).aggregate([
    _.count().name('n'),  # Use 'n' instead of 'count'
    _.price.mean().name('avg_price')  # No conflict
])
filtered = grouped.filter(_.n > 100)  # Now _.n works!
```

## Conflicting Names to Avoid

These deferred method names will cause conflicts if used as column names:

- `count` - conflicts with `_.count()`
- `mean` - conflicts with `_.mean()`
- `sum` - conflicts with `_.sum()`
- `min` - conflicts with `_.min()`
- `max` - conflicts with `_.max()`
- `std` - conflicts with `_.std()`
- `var` - conflicts with `_.var()`

## Best Practices

### 1. Use Descriptive Column Names
```python
# ✅ Good - clear and no conflicts
aggregate([
    _.count().name('total_count'),
    _.price.mean().name('average_price'),
    _.amount.sum().name('total_amount')
])
```

### 2. Use Short Aliases
```python
# ✅ Good - short and no conflicts
aggregate([
    _.count().name('n'),
    _.price.mean().name('avg'),
    _.amount.sum().name('tot')
])
```

### 3. When You Must Use Simple Names

If you must name columns 'count', 'mean', etc.:

```python
# Create aggregation
agg = table.group_by(_.cat).aggregate([
    _.count().name('count')
])

# Use bracket notation for filtering
filtered = agg.filter(_['count'] > 100)

# Use string names for ordering
from xorq.vendor import ibis
ordered = agg.order_by(ibis.desc('count'))
```

## Common Error Patterns

### Pattern 1: Filter After Aggregation
```python
# ❌ WRONG
grouped.filter(_.count > 100)

# ✅ CORRECT
grouped.filter(_['count'] > 100)
```

### Pattern 2: Order By Aggregated Column
```python
# ❌ WRONG
grouped.order_by(_.count.desc())

# ✅ CORRECT
from xorq.vendor import ibis
grouped.order_by(ibis.desc('count'))
```

### Pattern 3: Mutate Using Aggregated Column
```python
# ❌ WRONG
grouped.mutate(pct=_.count / _.count.sum())

# ✅ CORRECT
grouped.mutate(pct=_['count'] / _['count'].sum())
```

## Quick Reference

| Operation | Wrong | Correct |
|-----------|-------|---------|
| Filter | `_.count > 100` | `_['count'] > 100` |
| Order | `_.count.desc()` | `ibis.desc('count')` |
| Select | `_.count` | `_['count']` or rename first |
| Mutate | `_.count * 2` | `_['count'] * 2` |

## Summary

**After aggregation with conflicting names:**
1. Use `_['column_name']` for column access
2. Use `ibis.desc('column_name')` for ordering
3. Or better: avoid conflicts by using descriptive column names

This is a limitation of the deferred execution system where method names shadow column names.
