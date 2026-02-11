# Cache Expression Results - Xorq Tutorial

This tutorial shows you how Xorq's caching system works through hands-on examples. You'll see cache hits and misses in real time, and understand when Xorq reuses results versus recomputing them.

After completing this tutorial, you'll know how to use caching to speed up your workflows.

## Why caching matters

Running the same query twice shouldn't mean doing the work twice. Xorq caches expression results so repeated queries return instantly from the cache instead of recomputing.

This is especially powerful for expensive operations:
- Loading large datasets from remote databases
- Training machine learning models
- Calling external APIs
- Running complex aggregations

> **Tip:** Xorq uses content-addressed hashing to determine if an expression matches cached results. Same computation = same hash = cache hit.

## Prerequisites

Before starting, install Xorq:

```bash
pip install xorq jupyter
```

## How to follow along

Run the code examples in order using any of these methods:

- **Jupyter notebook**: Create a new notebook and run each code block in a separate cell.
- **Python interactive shell**: Open a terminal, run `python`, then copy and paste each code block.
- **Python script**: Copy all code blocks into a `.py` file and run it with `python script.py`.

The code blocks build on each other. Variables like `iris`, `storage`, and `cached_expr` are created in earlier blocks and used in later ones.

---

## Set up caching

You'll start by connecting to a backend and setting up a cache storage location.

```python
import xorq.api as xo
from xorq.caching import SourceCache

# Connect to the embedded backend where cached data is stored
con = xo.connect()

# Create a SourceCache object that manages the cache
storage = SourceCache.from_kwargs(source=con)

print(f"Connected to: {con}")
print(f"Cache storage ready!")
```

**What this does:**
1. Connect to the embedded backend where cached data is stored.
2. Create a SourceCache object that manages the cache.

`SourceCache` stores cached results in your backend as tables. When you run an expression with `.cache()`, Xorq saves the results and reuses them on subsequent runs.

---

## Cache your first expression

Now you'll build an expression and add caching to it.

```python
import xorq.api as xo
from xorq.caching import SourceCache

# Setup (from previous step)
con = xo.connect()
storage = SourceCache.from_kwargs(source=con)

# Load the iris dataset
iris = xo.examples.iris.fetch(backend=con)

# Build a filter expression with caching
cached_expr = (
    iris
    .filter(xo._.sepal_length > 6)
    .cache(cache=storage)  # Add caching here
)

print(f"Expression with caching: {type(cached_expr)}")
```

**What this does:**
1. Load the iris dataset.
2. Build a filter expression.
3. Add caching with `.cache(cache=storage)`.

The `.cache()` method tells Xorq to store results from this expression. On the first run, Xorq computes and caches the results. On subsequent runs, it retrieves them directly from cache.

---

## Observe cache miss (first run)

You'll execute the expression for the first time. This will be a cache miss, Xorq has to compute the results.

```python
import xorq.api as xo
import time
from xorq.caching import SourceCache

# Setup (from previous steps)
con = xo.connect()
storage = SourceCache.from_kwargs(source=con)
iris = xo.examples.iris.fetch(backend=con)
cached_expr = (
    iris
    .filter(xo._.sepal_length > 6)
    .cache(cache=storage)
)

# Start timing the execution
print("First execution (cache miss)...")
start = time.time()

# Execute the expression - triggers computation and caching
result1 = cached_expr.execute()

# Print how long it took
elapsed = time.time() - start
print(f"✗ Cache miss: computed in {elapsed:.4f} seconds")
print(f"Result shape: {result1.shape}")
print(f"\nFirst few rows:")
print(result1.head(3))
```

Since this is the first run, Xorq computed the filter operation and stored the results in cache.

---

## Observe cache hit (second run)

Now you can run the same expression again. This time you'll see a cache hit.

```python
import xorq.api as xo
import time
from xorq.caching import SourceCache

# Setup (from previous steps)
con = xo.connect()
storage = SourceCache.from_kwargs(source=con)
iris = xo.examples.iris.fetch(backend=con)
cached_expr = (
    iris
    .filter(xo._.sepal_length > 6)
    .cache(cache=storage)
)
result1 = cached_expr.execute()  # From previous step

# Time the second execution
print("\nSecond execution (cache hit)...")
start = time.time()

# Run the same expression again
result2 = cached_expr.execute()

# See how much faster it was
elapsed = time.time() - start
print(f"✓ Cache hit: returned in {elapsed:.4f} seconds")
print(f"Results match: {result1.equals(result2)}")
```

The second execution should be significantly faster because Xorq fetched results from cache instead of recomputing the filter operation.

> **Note:** Xorq computes a hash from your expression's structure and data sources. If the expression is identical, then the hash matches, and you get a cache hit.

---

## Understand cache invalidation

What happens if you change the expression? You'll modify the filter and see cache invalidation in action.

```python
import xorq.api as xo
import time
from xorq.caching import SourceCache

# Setup (from previous steps)
con = xo.connect()
storage = SourceCache.from_kwargs(source=con)
iris = xo.examples.iris.fetch(backend=con)

# Create a new expression with a different filter threshold
modified_expr = (
    iris
    .filter(xo._.sepal_length > 6.5)  # Changed from > 6 to > 6.5
    .cache(cache=storage)
)

# Execute the modified expression
print("Modified expression (different filter)...")
start = time.time()
result3 = modified_expr.execute()
elapsed = time.time() - start

# Cache miss because the expression changed
print(f"✗ Cache miss: computed in {elapsed:.4f} seconds")
print(f"Different result shape: {result3.shape}")
```

Since you changed the filter threshold, Xorq computed a different hash. The cache from the previous expression doesn't match, so Xorq recomputed.

---

## Compare multiple runs

You'll run several executions and see the timing difference between cache hits and misses.

```python
import xorq.api as xo
import time
from xorq.caching import SourceCache

# Setup (from previous steps)
con = xo.connect()
storage = SourceCache.from_kwargs(source=con)
iris = xo.examples.iris.fetch(backend=con)
cached_expr = (
    iris
    .filter(xo._.sepal_length > 6)
    .cache(cache=storage)
)

# Create a helper function to time executions
def time_execution(expr, label):
    start = time.time()
    result = expr.execute()
    elapsed = time.time() - start
    return elapsed, len(result)

# Print a header for the comparison
print("\nTiming comparison:")
print("-" * 50)

# Run the same expression three times
t1, rows1 = time_execution(cached_expr, "First run")
print(f"Run 1 (miss):  {t1:.4f}s - {rows1} rows")

t2, rows2 = time_execution(cached_expr, "Second run")
print(f"Run 2 (hit):   {t2:.4f}s - {rows2} rows")

t3, rows3 = time_execution(cached_expr, "Third run")
print(f"Run 3 (hit):   {t3:.4f}s - {rows3} rows")

# Calculate the speedup from caching
speedup = t1 / t2 if t2 > 0 else float('inf')
print(f"\nSpeedup from caching: {speedup:.1f}x faster")
```

The first execution is a cache miss (slower), but the second and third are cache hits (much faster). This shows how caching eliminates redundant computation.

> **Warning:** SourceCache keeps cached data in your backend as tables. Make sure you have enough storage space for cached results, especially with large datasets.

---

## Chain cached expressions

You can cache multiple steps in a pipeline. Each cached expression can reuse results from previous runs.

```python
import xorq.api as xo
from xorq.caching import SourceCache

# Setup (from previous steps)
con = xo.connect()
storage = SourceCache.from_kwargs(source=con)
iris = xo.examples.iris.fetch(backend=con)

# Cache the filtered dataset
step1 = iris.filter(xo._.sepal_length > 5).cache(cache=storage)

# Build on the cached result and cache the aggregation too
step2 = step1.group_by("species").agg(
    avg_width=xo._.sepal_width.mean()
).cache(cache=storage)

# First execution caches both steps
print("First execution of step2...")
result_a = step2.execute()

# Second execution hits cache for both steps
print("\nSecond execution of step2...")
result_b = step2.execute()

print("\nBoth steps now cached!")
print(result_a)
```

When you cache multiple steps, Xorq can reuse intermediate results, making complex pipelines faster on repeated runs.

---

## Use persistent ParquetCache

So far, you've used SourceCache, which stores cached data in your backend. But what if you want cache that persists across Python sessions? That's where ParquetCache comes in.

**The problem with SourceCache:** When you restart Python, the cached data might not persist depending on your backend configuration. You lose the cache and have to recompute.

**The solution:** ParquetCache writes cached results as `.parquet` files to disk. These files survive across sessions, so your cache persists even after closing Python.

```python
from pathlib import Path
from xorq.caching import ParquetCache
import xorq.api as xo

# Connect to backend and define a directory for cache files
con = xo.connect()
cache_dir = Path.cwd() / "xorq_cache"

# Create ParquetCache with base_path pointing to the cache directory
parquet_cache = ParquetCache.from_kwargs(source=con, base_path=cache_dir)

# Build an expression with filtering and aggregation
iris = xo.examples.iris.fetch(backend=con)
cached_with_parquet = (
    iris
    .filter(xo._.sepal_length > 6)
    .group_by("species")
    .agg(
        avg_sepal=xo._.sepal_length.mean(),
        count=xo._.species.count()
    )
    .cache(cache=parquet_cache)  # Cache using ParquetCache instead of SourceCache
)

# Execute and cache the results to disk
print("First execution with ParquetCache...")
result = cached_with_parquet.execute()
print(f"Result shape: {result.shape}")
print(result)
```

After running this, you can check the cache files on disk:

```python
from pathlib import Path
import xorq.api as xo
from xorq.caching import ParquetCache

# Setup (from previous step)
con = xo.connect()
cache_dir = Path.cwd() / "xorq_cache"
parquet_cache = ParquetCache.from_kwargs(source=con, base_path=cache_dir)

iris = xo.examples.iris.fetch(backend=con)
cached_with_parquet = (
    iris
    .filter(xo._.sepal_length > 6)
    .group_by("species")
    .agg(avg_sepal=xo._.sepal_length.mean(), count=xo._.species.count())
    .cache(cache=parquet_cache)
)
result = cached_with_parquet.execute()  # From previous step

# Find all .parquet files in the cache directory
cache_files = list(cache_dir.glob("*.parquet"))

# Print information about the cached files
print(f"\nCache files created: {len(cache_files)}")
print(f"Cache directory: {cache_dir}")

if cache_files:
    print(f"Cache file: {cache_files[0].name}")
```

You'll see actual `.parquet` files in the `xorq_cache` directory. These files contain your cached results and persist even after you close Python.

> **Tip: When to use each cache type**
> 
> **SourceCache**: Fast, session-scoped. Use for temporary caching during development.
> 
> **ParquetCache**: Persistent across sessions. Use when you want cache to survive Python restarts (local development, iterative analysis).

Now when you restart Python and rerun the same expression with ParquetCache pointing to the same directory, Xorq finds the cached `.parquet` files and returns results instantly without recomputation.

---

## Complete example

Here's a full caching workflow in one place:

```python
import xorq.api as xo
from xorq.caching import SourceCache

# Set up connection and load data
con = xo.connect()
storage = SourceCache.from_kwargs(source=con)
iris = xo.examples.iris.fetch(backend=con)

# Build cached expression
cached_expr = (
    iris
    .filter(xo._.sepal_length > 6)
    .cache(cache=storage)
)

# First run: cache miss
result1 = cached_expr.execute()
print("First run complete (cached)")

# Second run: cache hit
result2 = cached_expr.execute()
print("Second run complete (from cache)")
```

---

## Next steps

Now you understand how caching works. Continue learning:

- [Switch backends](switch_backends.qmd) shows how caching works when moving data between engines
- [Your first build](../tutorials/core_tutorials/your_first_build.qmd) explains how cached expressions become portable artifacts
- [Optimize pipeline performance](/guides/performance_workflows/optimize_pipeline_performance.qmd) covers advanced caching strategies

---

**Tutorial source:** [Xorq Documentation](https://xorq.dev)  
**License:** Check the Xorq repository for license information

