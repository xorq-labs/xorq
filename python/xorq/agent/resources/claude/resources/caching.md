# Caching Strategies

## Overview

Xorq provides first-class caching support to avoid recomputing expensive operations, with multiple storage backends and cache invalidation strategies.

---

## Core Caching Concepts

### Why Cache?

**Problems caching solves:**
- Avoid recomputing expensive transformations
- Reduce data fetching from remote sources
- Speed up iterative development
- Minimize compute costs

**When to cache:**
- Expensive aggregations or joins
- Data loaded from remote sources
- Intermediate results used multiple times
- Stable transformations (infrequently changing)

**When not to cache:**
- Fast operations (filtering, selecting)
- Operations on small datasets
- Frequently changing data
- One-time computations

---

## Parquet Storage

### Pattern: Local Parquet Cache

**Most common caching backend - fast local disk storage**

```python
from pathlib import Path
from xorq.caching import ParquetStorage
import xorq.api as xo

# Setup connection
con = xo.connect()

# Create cache storage
storage = ParquetStorage(
    source=con,
    relative_path=Path("./parquet-cache")
)

# Cache expensive operations
expr = (
    expensive_data_load()
    .expensive_transformation()
    .cache(storage=storage)  # Cache here
    .more_operations()
)

# First execution: computes and caches
result1 = expr.execute()

# Second execution: reads from cache
result2 = expr.execute()  # Much faster!
```

**Features:**
- Fast parquet read/write
- Automatic cache key generation
- Invalidation on expression changes
- Works with any backend

---

### Pattern: Custom Cache Directory

```python
# Project-specific cache
storage = ParquetStorage(
    source=con,
    relative_path=Path("./cache/project-a")
)

# User-specific cache
storage = ParquetStorage(
    source=con,
    relative_path=Path.home() / ".cache" / "xorq"
)

# Temporary cache
import tempfile
storage = ParquetStorage(
    source=con,
    relative_path=Path(tempfile.gettempdir()) / "xorq-cache"
)
```

---

## Snowflake Stage Caching

### Pattern: Cache to Snowflake Stage

**Use case**: Cache results in Snowflake for team sharing

```python
from xorq.caching import SnowflakeStageStorage

# Connect to Snowflake
con = xo.connect("snowflake://account/database/schema")

# Create stage storage
storage = SnowflakeStageStorage(
    connection=con,
    stage_name="@MY_STAGE",
    path_prefix="xorq_cache/"
)

# Cache to stage
expr = (
    data
    .expensive_transformation()
    .cache(storage=storage)
)

result = expr.execute()
```

**Benefits:**
- Shared cache across team
- Leverages Snowflake storage
- Persistent across sessions
- Integrates with Snowflake security

---

## Postgres Caching

### Pattern: Cache in Postgres Tables

**Use case**: Cache in database for SQL access

```python
from xorq.caching import PostgresStorage

# Connect to Postgres
con = xo.connect("postgresql://user:pass@host/db")

# Create postgres storage
storage = PostgresStorage(
    connection=con,
    schema_name="xorq_cache",
    table_prefix="cache_"
)

# Cache to postgres
expr = (
    data
    .expensive_transformation()
    .cache(storage=storage)
)

result = expr.execute()
```

**Benefits:**
- SQL-accessible cache
- ACID guarantees
- Transactional consistency
- Standard database features

---

## Cache Placement Strategies

### Pattern: Cache After Expensive Operations

```python
# Good: cache after expensive operation
pipeline = (
    deferred_read_parquet("huge_file.parquet", con, "data")
    .expensive_join_with_another_table()
    .expensive_aggregation()
    .cache(storage=storage)  # Cache here
    .filter(xo._.category == "A")  # Fast filter after cache
    .select("id", "value")
)
```

### Pattern: Multiple Cache Points

```python
# Cache at multiple levels
pipeline = (
    raw_data
    .expensive_transformation1()
    .cache(storage=storage, name="stage1")  # Cache intermediate
    .expensive_transformation2()
    .cache(storage=storage, name="stage2")  # Cache final
)
```

### Pattern: Conditional Caching

```python
# Cache only in production
import os

if os.getenv("ENV") == "production":
    expr = expr.cache(storage=storage)
```

---

## Cache Management

### Pattern: List Cached Nodes

```python
# Build expression with cache
cached_expr = expr.cache(storage=storage)

# List all cached nodes in expression
print(cached_expr.ls.cached_nodes)

# Output: List of cache locations
```

### Pattern: Clear Cache

```python
# Clear specific cache directory
import shutil
cache_dir = Path("./parquet-cache")
if cache_dir.exists():
    shutil.rmtree(cache_dir)

# Recreate for next run
cache_dir.mkdir(exist_ok=True)
```

### Pattern: Named Caches

```python
# Use descriptive cache names
stage1 = expr.cache(storage=storage, name="data_cleaning")
stage2 = stage1.more_ops().cache(storage=storage, name="feature_engineering")
stage3 = stage2.more_ops().cache(storage=storage, name="aggregation")

# Easy to identify and manage
```

---

## Cache Invalidation

### Automatic Invalidation

**Xorq automatically invalidates cache when:**
- Expression definition changes
- Input data changes (detected via hash)
- Cache file is deleted

```python
# First run: computes and caches
expr1 = data.filter(xo._.a > 1).cache(storage=storage)
result1 = expr1.execute()

# Second run: reads from cache
result2 = expr1.execute()  # Cached

# Changed expression: cache miss
expr2 = data.filter(xo._.a > 2).cache(storage=storage)
result3 = expr2.execute()  # Recomputes (different expression)
```

### Manual Invalidation

```python
# Delete cache files for specific expression
cache_key = "my_cache_key"
cache_file = Path("./parquet-cache") / f"{cache_key}.parquet"
if cache_file.exists():
    cache_file.unlink()

# Or clear entire cache directory
import shutil
shutil.rmtree("./parquet-cache")
Path("./parquet-cache").mkdir()
```

---

## Performance Patterns

### Pattern: Benchmark Cache Impact

```python
import time

# Without cache
start = time.time()
result1 = expr.execute()
uncached_time = time.time() - start

# With cache (first run - write)
cached_expr = expr.cache(storage=storage)
start = time.time()
result2 = cached_expr.execute()
write_time = time.time() - start

# With cache (second run - read)
start = time.time()
result3 = cached_expr.execute()
read_time = time.time() - start

print(f"Uncached: {uncached_time:.2f}s")
print(f"Cache write: {write_time:.2f}s")
print(f"Cache read: {read_time:.2f}s")
print(f"Speedup: {uncached_time / read_time:.1f}x")
```

### Pattern: Cache Size Monitoring

```python
# Check cache directory size
def get_dir_size(path: Path) -> int:
    """Get directory size in bytes"""
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

cache_dir = Path("./parquet-cache")
size_bytes = get_dir_size(cache_dir)
size_mb = size_bytes / (1024 * 1024)
print(f"Cache size: {size_mb:.2f} MB")

# Cleanup if too large
if size_mb > 1000:  # 1 GB
    print("Cache too large, clearing...")
    shutil.rmtree(cache_dir)
    cache_dir.mkdir()
```

---

## Advanced Patterns

### Pattern: Tiered Caching

```python
# Fast local cache for development
dev_storage = ParquetStorage(
    source=con,
    relative_path=Path("./local-cache")
)

# Slow remote cache for production
prod_storage = SnowflakeStageStorage(
    connection=con,
    stage_name="@PROD_CACHE"
)

# Choose based on environment
import os
storage = dev_storage if os.getenv("ENV") == "dev" else prod_storage

expr = data.cache(storage=storage)
```

### Pattern: Partial Pipeline Caching

```python
# Cache only expensive parts
def build_pipeline(data, use_cache=True):
    # Always fast operations
    filtered = data.filter(xo._.status == "active")

    # Optionally cache expensive operation
    if use_cache:
        transformed = (
            filtered
            .expensive_operation()
            .cache(storage=storage)
        )
    else:
        transformed = filtered.expensive_operation()

    # Continue pipeline
    return transformed.final_operations()

# Development: no cache for fast iteration
dev_result = build_pipeline(data, use_cache=False)

# Production: use cache
prod_result = build_pipeline(data, use_cache=True)
```

### Pattern: Conditional Cache Strategy

```python
def smart_cache(expr, storage, threshold_seconds=5.0):
    """Cache only if operation is slow"""
    import time

    # Time first execution
    start = time.time()
    result = expr.execute()
    duration = time.time() - start

    # Cache if slow
    if duration > threshold_seconds:
        print(f"Slow operation ({duration:.2f}s), enabling cache")
        return expr.cache(storage=storage)
    else:
        print(f"Fast operation ({duration:.2f}s), no cache needed")
        return expr

# Use smart caching
expr = smart_cache(expensive_expr, storage)
```

---

## Cache Storage Comparison

| Storage Type | Speed | Sharing | Persistence | Use Case |
|-------------|-------|---------|-------------|----------|
| **ParquetStorage** | ⚡ Fast | Local only | Until deleted | Development, single-user |
| **SnowflakeStageStorage** | 🐢 Slow | Team-wide | High | Production, team sharing |
| **PostgresStorage** | ⚡ Fast | Database access | High | SQL-accessible cache |
| **Remote** | 🐢 Slow | Distributed | High | Distributed computing |

---

## Best Practices

### 1. Cache After Expensive Operations

```python
# Good: cache after expensive op
expr = (
    data
    .expensive_join()
    .expensive_agg()
    .cache(storage=storage)  # Cache here
    .cheap_filter()
)

# Avoid: cache before expensive ops
expr = (
    data
    .cache(storage=storage)  # Too early
    .expensive_join()
    .expensive_agg()
)
```

### 2. Use Descriptive Cache Names

```python
# Good: descriptive names
cleaned = raw.clean().cache(storage=storage, name="cleaned_data")
featured = cleaned.engineer().cache(storage=storage, name="features")

# Avoid: anonymous caches
cleaned = raw.clean().cache(storage=storage)
featured = cleaned.engineer().cache(storage=storage)
```

### 3. Monitor Cache Size

```python
# Good: periodic cleanup
if cache_dir.stat().st_size > MAX_CACHE_SIZE:
    cleanup_old_caches()

# Avoid: unbounded cache growth
expr.cache(storage=storage)  # Never cleaned up
```

### 4. Document Cache Strategy

```python
# Good: document why caching
def process_data(data):
    """Process data with caching.

    Caches after the expensive join operation because:
    - Join takes ~30 seconds
    - Result is stable (reference data rarely changes)
    - Subsequent operations are fast filters
    """
    return (
        data
        .join(reference_data)  # Expensive
        .cache(storage=storage, name="joined_with_reference")
        .filter(xo._.active == True)  # Fast
    )
```

### 5. Test With and Without Cache

```python
# Good: verify cache correctness
result_no_cache = expr.execute()
result_with_cache = expr.cache(storage=storage).execute()

assert result_no_cache.equals(result_with_cache)
```

---

## Troubleshooting

### Issue: Cache Not Working

**Symptom**: Every execution recomputes

**Check:**
1. Cache directory exists and is writable
2. Expression is exactly the same
3. No errors during cache write

**Solution:**
```python
# Verify cache directory
cache_dir = Path("./parquet-cache")
cache_dir.mkdir(exist_ok=True)
assert cache_dir.is_dir()
assert os.access(cache_dir, os.W_OK)

# Check for cache files
cached_expr = expr.cache(storage=storage)
result = cached_expr.execute()
cache_files = list(cache_dir.glob("*.parquet"))
print(f"Cache files: {cache_files}")
```

### Issue: Stale Cache

**Symptom**: Using old data despite changes

**Solution:**
```python
# Clear cache before run
import shutil
cache_dir = Path("./parquet-cache")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
cache_dir.mkdir()

# Then run with fresh cache
expr = data.transform().cache(storage=storage)
result = expr.execute()
```

### Issue: Cache Directory Full

**Symptom**: Disk space errors

**Solution:**
```python
# Implement cache rotation
def cleanup_old_cache(cache_dir: Path, max_age_days: int = 7):
    """Remove cache files older than max_age_days"""
    import time

    cutoff = time.time() - (max_age_days * 86400)
    for cache_file in cache_dir.rglob("*.parquet"):
        if cache_file.stat().st_mtime < cutoff:
            cache_file.unlink()
            print(f"Removed old cache: {cache_file}")

cleanup_old_cache(Path("./parquet-cache"), max_age_days=7)
```

### Issue: Cache Slower Than Recompute

**Symptom**: Cached execution is slower

**Possible causes:**
1. Very small dataset (serialization overhead)
2. Slow storage (network filesystem)
3. Cache after fast operation

**Solution:**
```python
# Remove cache if operation is fast
import time

start = time.time()
result = expr.execute()
duration = time.time() - start

if duration < 1.0:  # Less than 1 second
    print("Operation is fast, don't cache")
    final_expr = expr
else:
    print("Operation is slow, use cache")
    final_expr = expr.cache(storage=storage)
```

---

## Summary

**Caching patterns:**
- `ParquetStorage` - Fast local caching
- `SnowflakeStageStorage` - Team-wide caching
- `PostgresStorage` - SQL-accessible caching
- Place cache after expensive operations
- Use descriptive names for cache management

**Best practices:**
- Cache expensive operations only
- Monitor cache size and implement rotation
- Test with and without cache
- Document caching strategy
- Benchmark cache impact

**Common mistakes:**
- Caching fast operations (overhead > benefit)
- Not monitoring cache size
- Caching before filtering (cache too much data)
- Anonymous caches (hard to manage)
