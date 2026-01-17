# Xorq Patterns and Best Practices

Common patterns and best practices for xorq development.

## Core Patterns

### Always Check Schema First

**Pattern:**
```python
import xorq.api as xo

# Connect
con = xo.connect()
table = con.table("data")

# CRITICAL: Check schema before building expression
print(table.schema())

# Now build expression with correct column names
expr = table.select("actual_column_name")
```

**Why:**
- Prevents runtime errors
- Shows exact column names and types
- Required by agent reliability guidelines

**Reference:**
```bash
xorq agent prompt show must_check_schema
```

---

### Use Vendored Ibis

**Pattern:**
```python
# Correct import order:
import xorq.api as xo
from xorq.vendor import ibis
from xorq.common.utils.ibis_utils import from_ibis
from xorq.caching import ParquetCache

# NOT:
import ibis  # ✗ Wrong
```

**Why:**
- xorq uses specific ibis version
- Ensures compatibility
- Required for builds to work

**Reference:**
```bash
xorq agent prompt show xorq_vendor_ibis
```

---

### Cache at Boundaries

**Pattern:**
```python
# Cache after expensive operations
expr = (
    from_ibis(
        table
        .filter(ibis._.status == "active")
        .group_by("user_id")
        .agg(total=ibis._.amount.sum())  # Expensive
    )
    .cache(ParquetCache.from_kwargs())  # Cache here
)

# Further transformations work on cached result
final = from_ibis(expr.select(["user_id", "total"]))
```

**Why:**
- Avoids recomputing expensive operations
- Input-addressed: same inputs = cache hit
- Speeds up iteration

**Where to cache:**
- After aggregations
- After joins
- After expensive UDFs
- Before backend transitions

---

### Deferred Execution

**Pattern:**
```python
# Build expression (deferred)
expr = (
    table
    .filter(ibis._.col > 0)
    .group_by("key")
    .agg(metric=ibis._.value.mean())
)

# Nothing executed yet!

# Execute when needed
result = expr.execute()  # Now it runs

# Or build to manifest
# xorq build expr.py -e expr  # Builds without executing
```

**Why:**
- Allows optimization before execution
- Enables multi-engine composition
- Supports caching at any node

**Reference:**
```bash
xorq agent prompt show xorq_core
```

---

## Multi-Engine Patterns

### DuckDB → DataFusion

**Pattern:**
```python
import xorq.api as xo
from xorq.common.utils.ibis_utils import from_ibis

# Start with DuckDB
duckdb_con = xo.connect()
source = duckdb_con.table("large_table")

# Move to DataFusion for specific ops
datafusion_con = xo.datafusion.connect()
expr = (
    from_ibis(source)
    .into_backend(datafusion_con)
)
```

**When:**
- Need DataFusion-specific features
- Switching engines for performance

---

### Local → Remote

**Pattern:**
```python
# Start local for development
local_con = xo.connect()  # DuckDB
table = local_con.table("sample_data")

# Build expression
expr = table.filter(ibis._.col > 0)

# Switch to production backend
prod_con = xo.snowflake.connect(...)
prod_expr = from_ibis(expr).into_backend(prod_con)
```

**When:**
- Developing locally, deploying remotely
- Testing with sample before full data

---

### Cache Before Backend Switch

**Pattern:**
```python
# Expensive query in source backend
source_expr = (
    from_ibis(remote_table.group_by("key").agg(...))
    .cache(ParquetCache.from_kwargs())  # Cache first
)

# Then switch backend
target_expr = source_expr.into_backend(target_con)
```

**Why:**
- Avoids re-executing expensive remote query
- Cache is backend-agnostic

---

## ML Patterns

### Deferred sklearn Pipeline

**Pattern:**
```python
from xorq.expr.ml.pipeline_lib import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# Create sklearn pipeline
sklearn_pipe = make_pipeline(
    StandardScaler(),
    RandomForestClassifier()
)

# Convert to xorq (deferred)
xorq_pipe = Pipeline.from_instance(sklearn_pipe)

# Use in expression
train_expr = xorq_pipe.fit(train_data)
predict_expr = xorq_pipe.predict(test_data)
```

**Why:**
- Keeps pipeline deferred
- Enables caching of fitted models
- Integrates with xorq manifest system

---

### Train/Test Split

**Pattern:**
```python
from xorq.vendor import ibis

# Split by percentage
train = table.filter(ibis._.split_col < 0.8)
test = table.filter(ibis._.split_col >= 0.8)

# Or by date
train = table.filter(ibis._.date < "2024-01-01")
test = table.filter(ibis._.date >= "2024-01-01")
```

---

### Feature Engineering

**Pattern:**
```python
# Create features
features = (
    table
    .mutate(
        # Derived columns
        total=ibis._.price * ibis._.quantity,
        is_premium=ibis._.amount > 1000,
        # Aggregates
        user_avg=ibis._.amount.mean().over(
            ibis.window(group_by="user_id")
        )
    )
    .select(["feature1", "feature2", "total", "is_premium", "user_avg"])
)

# Cache features
cached_features = (
    from_ibis(features)
    .cache(ParquetCache.from_kwargs())
)
```

**Why:**
- Reusable features across models
- Cached to avoid recomputation
- Versioned with manifest

---

## Caching Patterns

### Parquet Cache

**Pattern:**
```python
from xorq.caching import ParquetCache

# Default cache location
cache = ParquetCache.from_kwargs()

# Custom location
cache = ParquetCache.from_kwargs(path="/custom/cache")

expr = from_ibis(query).cache(cache)
```

**When:**
- Standard tabular data
- Local development
- Fast read/write

---

### Multi-Level Caching

**Pattern:**
```python
# Cache at multiple stages
stage1 = (
    from_ibis(source_query)
    .cache(ParquetCache.from_kwargs())
)

stage2 = (
    from_ibis(
        stage1
        .group_by("key")
        .agg(metric=ibis._.value.sum())
    )
    .cache(ParquetCache.from_kwargs())
)

final = (
    from_ibis(stage2.filter(ibis._.metric > 100))
    .cache(ParquetCache.from_kwargs())
)
```

**Why:**
- Each stage cached independently
- Change stage2 = only recomputes stage2+
- Faster iteration

---

## Build Patterns

### Single Expression

**Pattern:**
```python
# In expr.py
import xorq.api as xo
from xorq.vendor import ibis
from xorq.common.utils.ibis_utils import from_ibis

con = xo.connect()
table = con.table("data")

expr = from_ibis(table.filter(ibis._.col > 0))
```

```bash
xorq build expr.py -e expr
```

---

### Multiple Expressions

**Pattern:**
```python
# In pipeline.py
train_expr = ...
test_expr = ...
predictions = ...

# Export one for building
expr = predictions
```

```bash
# Build the exported one
xorq build pipeline.py -e expr

# Or build different ones
xorq build pipeline.py -e train_expr
xorq build pipeline.py -e test_expr
```

---

### Reproducible Build with uv

**Pattern:**
```bash
# Build with dependencies captured
xorq uv-build expr.py -e expr

# Output includes:
# builds/<hash>/sdist.tar.gz  # Python deps
# builds/<hash>/expr.yaml      # Expression manifest

# Share build
tar -czf build.tar.gz builds/<hash>/

# Recipient can reproduce exactly
```

**When:**
- Need reproducibility
- Sharing with team
- Production deployment

---

## Catalog Patterns

### Versioned Aliases

**Pattern:**
```bash
# Add with version in alias
xorq catalog add builds/<hash1> --alias pipeline-v1
xorq catalog add builds/<hash2> --alias pipeline-v2
xorq catalog add builds/<hash3> --alias pipeline-v3

# Keep latest pointer
xorq catalog rm pipeline-latest
xorq catalog add builds/<hash3> --alias pipeline-latest
```

**Why:**
- Track versions
- Easy rollback
- Clear history

---

### Feature Catalog

**Pattern:**
```bash
# Catalog feature expressions
xorq catalog add builds/<hash1> --alias feature-user-activity
xorq catalog add builds/<hash2> --alias feature-item-popularity
xorq catalog add builds/<hash3> --alias feature-user-item-interaction

# Reuse in pipelines
BUILD=$(xorq catalog info feature-user-activity --build-path)
xorq run $BUILD -o user_activity.parquet
```

**Why:**
- Reusable features
- Feature store pattern
- Team sharing

---

## Serving Patterns

### Model Serving

**Pattern:**
```python
# 1. Build model pipeline
# model_pipeline.py
train_expr = pipeline.fit(train_data)
predict_expr = pipeline.predict(...)

expr = predict_expr
```

```bash
# 2. Build and serve
xorq build model_pipeline.py -e expr

# 3. Identify input node to unbind
# (the node where you want to inject new data)
cat builds/<hash>/expr.yaml
# Find: '@read_abc123'

# 4. Serve
xorq serve-unbound builds/<hash> \
  --to_unbind_hash abc123 \
  --port 9002
```

```python
# 5. Client calls with new data
import xorq.api as xo

backend = xo.flight.connect(port=9002)
f = backend.get_exchange("default")

new_data = {"feature1": [1, 2], "feature2": [3, 4]}
predictions = xo.memtable(new_data).pipe(f).execute()
```

**When:**
- Real-time inference
- Model as microservice
- Low-latency serving

---

## Error Handling Patterns

### Defensive Schema Checking

**Pattern:**
```python
import xorq.api as xo

con = xo.connect()
table = con.table("data")

# Always check schema
schema = table.schema()
print(schema)

# Verify required columns
required = ["col1", "col2", "col3"]
available = [field.name for field in schema]

for col in required:
    if col not in available:
        raise ValueError(f"Missing column: {col}")

# Now build safely
expr = table.select(required)
```

---

### Graceful Backend Fallback

**Pattern:**
```python
import xorq.api as xo

# Try preferred backend
try:
    con = xo.snowflake.connect(...)
except Exception:
    # Fall back to local
    print("Snowflake unavailable, using DuckDB")
    con = xo.connect()

table = con.table("data")
expr = from_ibis(table.filter(ibis._.col > 0))
```

---

## Agent Workflow Patterns

### Planning Phase

**Pattern:**
```bash
# 1. Get onboarding
xorq agent onboard

# 2. Read core prompts
xorq agent prompt show planning_phase
xorq agent prompt show xorq_core

# 3. List available skills
xorq agent templates list

# 4. Initialize project
xorq init -t <template>
```

**Reference:**
```bash
xorq agent prompt show planning_phase
```

---

### Sequential Execution

**Pattern:**
```bash
# 1. Build
xorq build expr.py -e expr

# 2. Verify
ls builds/<hash>/
cat builds/<hash>/expr.yaml

# 3. Test run
xorq run builds/<hash> -o test.parquet

# 4. Check output
head test.parquet  # or appropriate tool

# 5. Catalog
xorq catalog add builds/<hash> --alias my-expr

# 6. Verify catalog
xorq catalog ls
```

**Reference:**
```bash
xorq agent prompt show sequential_execution
```

---

## Integration Patterns

### With bd (beads)

**Pattern:**
```bash
# Session start
bd ready
bd show XQ-123
bd update XQ-123 --status in_progress

# Do work
xorq build expr.py -e expr

# Update issue with details
bd update XQ-123 --notes "Built expression
Build: builds/abc123def
Alias: feature-pipeline
"

# Complete
xorq catalog add builds/abc123def --alias feature-pipeline
bd close XQ-123
bd sync
```

---

### With Git

**Pattern:**
```bash
# Build expression
xorq build expr.py -e expr

# Commit expression source and manifest
git add expr.py builds/
git commit -m "Add feature pipeline

Expression for user activity features
Build: abc123def
Cached: aggregations, joins
"

# Push
git push
```

---

## Quick Reference

| Pattern | When to Use |
|---------|-------------|
| Always check schema | Every expression |
| Use vendored ibis | All imports |
| Cache at boundaries | After expensive ops |
| Deferred execution | Building pipelines |
| Multi-engine | Need specific features |
| Versioned aliases | Production pipelines |
| Model serving | Real-time inference |

---

For more details, see:
- [WORKFLOWS.md](WORKFLOWS.md) - Step-by-step guides
- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Command syntax
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
