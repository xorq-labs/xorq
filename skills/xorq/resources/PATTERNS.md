# Xorq Patterns and Best Practices

Common patterns and best practices for xorq development.

## Core Patterns

### Arrow IPC Streaming Pattern

**Pattern:** Stream xorq outputs through Arrow IPC for composition and analysis.

```bash
# Basic: Single source to DuckDB
xorq run source -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow; SELECT * FROM read_arrow('/dev/stdin') LIMIT 10"

# Advanced: Chain with run-unbound
xorq run source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound transform \
    --to_unbind_hash <hash> \
    --typ xorq.expr.relations.Read \
    -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow; SELECT * FROM read_arrow('/dev/stdin')"
```

**Key points:**
- Use `-f arrow` for Arrow IPC format (efficient binary serialization)
- Use `-o /dev/stdout` to enable piping (default is `/dev/null`)
- Redirect stderr with `2>/dev/null` to keep output clean
- DuckDB requires `LOAD arrow;` (community extension) before `read_arrow()`
- Use `xorq catalog sources <alias>` to find node hashes for unbinding

**When to use:**
- Interactive SQL exploration without writing Python
- Ad-hoc data validation and inspection
- Dynamic pipeline composition (source → transform1 → transform2 → SQL)
- Testing pipeline outputs quickly
- Integrating xorq with SQL-based tools (DuckDB, DataFusion, etc.)

**Workflow:**
```bash
# 1. Find unbound hashes
xorq catalog sources my-transform

# 2. Verify source schema
xorq catalog schema my-source

# 3. Compose and analyze
xorq run my-source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound my-transform \
    --to_unbind_hash abc123 \
    --typ xorq.expr.relations.Read \
    -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow;
    SELECT col1, COUNT(*) FROM read_arrow('/dev/stdin')
    GROUP BY col1"
```

**Troubleshooting:**
- **Empty results:** Check filters match your data (e.g., yearID >= 2020 but data is 2015)
- **Arrow extension error:** Run `duckdb -c "INSTALL arrow FROM community; LOAD arrow;"`
- **Schema mismatch:** Verify with `xorq catalog schema` and `xorq catalog sources`

---

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
xorq agents prompt show must_check_schema
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
xorq agents prompt show xorq_vendor_ibis
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
xorq agents prompt show xorq_core
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

### Memtable Placeholder Pattern

**Pattern:** Build transform expressions independently using memtable placeholders, then compose with real sources later.

```python
# In transform.py - Define transform with memtable placeholder
import xorq.api as xo
from xorq.vendor import ibis
from xorq.common.utils.ibis_utils import from_ibis
from xorq.caching import ParquetCache

# Create sample data matching expected source schema
sample_data = {
    "playerID": ["player1", "player2"],
    "yearID": [2020, 2021],
    "H": [150, 180],
    "AB": [500, 550],
    "teamID": ["NYY", "LAD"]
}

# Memtable as placeholder
source = xo.memtable(sample_data)
print(source.schema())  # Check schema

# Build transform on memtable
transform_expr = (
    from_ibis(
        source
        .mutate(batting_avg=ibis._.H / ibis._.AB)
        .filter(ibis._.batting_avg > 0.250)
        .group_by("playerID")
        .agg(
            total_hits=ibis._.H.sum(),
            total_at_bats=ibis._.AB.sum()
        )
    )
    .cache(ParquetCache.from_kwargs())
)

expr = transform_expr
```

**Building and cataloging:**
```bash
# Build transform with memtable
xorq build transform.py -e expr

# Catalog it
xorq catalog add builds/<hash> --alias batting-transform

# Find memtable node hash for unbinding
xorq catalog sources batting-transform
# Output: Hash: abc123def (this is the memtable node)
```

**Composing with real source:**
```bash
# Replace memtable with real source via Arrow IPC
xorq run real-source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound batting-transform \
    --to_unbind_hash abc123def \
    --typ xorq.expr.relations.Read \
    -o output.parquet
```

**Stream to DuckDB for exploration:**
```bash
# Interactive SQL on composed pipeline
xorq run real-source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound batting-transform \
    --to_unbind_hash abc123def \
    --typ xorq.expr.relations.Read \
    -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow;
    SELECT * FROM read_arrow('/dev/stdin')
    WHERE total_hits > 100
    ORDER BY total_at_bats DESC"
```

**Why:**
- **Independent development:** Build transform without waiting for source
- **Reusability:** Same transform works with multiple sources
- **Testing:** Test logic with small sample data first
- **Composition:** Combine via Arrow IPC at runtime
- **Flexibility:** Easy to swap sources or chain transforms

**When:**
- Multi-stage data pipelines
- Developing transforms in parallel with sources
- Creating reusable transformation components
- Testing complex logic before production
- Dynamic pipeline composition

**Key points:**
- Memtable schema must match source output schema
- Use `xorq catalog sources` to find node hash
- Use `xorq catalog schema` to verify compatibility
- Can chain multiple unbound transforms together
- Enables interactive exploration via DuckDB

**Python API for programmatic replacement:**
```python
from xorq.flight.exchanger import replace_one_unbound
from xorq.common.utils.node_utils import expr_to_unbound
import xorq.api as xo

# Load and replace programmatically
unbound_expr = xo.load_expr("builds/transform-hash")
replacement = xo.memtable({"col1": [1, 2]})
bound_expr = replace_one_unbound(unbound_expr, replacement)
result = bound_expr.execute()
```

**Utilities:**
- `replace_one_unbound(unbound_expr, table)` - Replace unbound with table
- `expr_to_unbound(expr, hash, tag, typs)` - Make expression unbound
- `replace_by_expr_hash(expr, hash, replace_with)` - Replace by hash

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

### Column Reference in Mutate

**Pattern:**
```python
# ❌ WRONG: Cannot reference newly created columns in same mutate
expr = data.mutate(
    total=ibis._.price * ibis._.quantity,
    profit=ibis._.total - ibis._.cost  # Error! total not available yet
)

# ✅ CORRECT: Chain mutate calls
expr = (
    data
    .mutate(total=ibis._.price * ibis._.quantity)
    .mutate(profit=ibis._.total - ibis._.cost)  # Now total is available
)

# ✅ ALTERNATIVE: Repeat the expression
expr = data.mutate(
    total=ibis._.price * ibis._.quantity,
    profit=(ibis._.price * ibis._.quantity) - ibis._.cost  # Inline
)
```

**Why:**
- Ibis evaluates all columns in a single `.mutate()` **in parallel**
- Newly created columns are not yet part of the table context
- This mirrors SQL behavior (all SELECT columns are peer expressions)

**When to chain vs batch:**
- **Batch** (single mutate) when columns are independent
- **Chain** (multiple mutates) when columns depend on each other

---

## Agent Workflow Patterns

### Planning Phase

**Pattern:**
```bash
# 1. Get onboarding
xorq agents onboard

# 2. Read core prompts
xorq agents prompt show planning_phase
xorq agents prompt show xorq_core

# 3. List available skills
xorq agents templates list

# 4. Initialize project
xorq init -t <template>
```

**Reference:**
```bash
xorq agents prompt show planning_phase
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
xorq agents prompt show sequential_execution
```

---

## Integration Patterns

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
