# Xorq Workflows

Step-by-step patterns for common xorq tasks.

## Basic Workflows

### 1. New Project from Template

**Goal:** Start a new xorq project using a template.

**Steps:**
```bash
# 1. Initialize from template
xorq init -t penguins

# 2. Setup agent guides (optional)
xorq agents init
# Creates: AGENTS.md, CLAUDE.md

# 3. Create your expression file
# (Write from scratch or use examples as reference)
```

**When to use:**
- Starting a new project
- Learning xorq patterns
- Need a working example

---

### 2. Build Expression from Scratch

**Goal:** Create and build a custom expression.

**Steps:**

1. **Write expression file (my_expr.py):**
```python
import xorq.api as xo
from xorq.vendor import ibis
from xorq.common.utils.ibis_utils import from_ibis
from xorq.caching import ParquetCache

# Connect to data source
con = xo.connect()
table = con.table("data")

# CRITICAL: Check schema first
print(table.schema())

# Build expression
expr = (
    from_ibis(
        table
        .filter(ibis._.column.notnull())
        .group_by("key")
        .agg(metric=ibis._.value.mean())
    )
    .cache(ParquetCache.from_kwargs())
)
```

2. **Build the expression:**
```bash
xorq build my_expr.py -e expr
# Output: builds/abc123def/
```

3. **Verify build:**
```bash
ls builds/abc123def/
# expr.yaml  metadata.json  profiles.yaml  database_tables/
```

**When to use:**
- Custom data pipeline
- Specific business logic
- Not covered by templates

---

### 3. Catalog and Reuse Builds

**Goal:** Register builds for discovery and reuse.

**Steps:**
```bash
# 1. Build expression
xorq build expr.py -e expr

# 2. Add to catalog with alias
xorq catalog add builds/abc123def --alias my-pipeline

# 3. List catalog
xorq catalog ls

# 4. Get build path from alias
BUILD_PATH=$(xorq catalog info my-pipeline --build-path)

# 5. Run by alias
xorq run $BUILD_PATH -o output.parquet

# 6. Show lineage
xorq lineage my-pipeline
```

**When to use:**
- Sharing builds across sessions
- Team collaboration
- Reusable components

---

### 4. Multi-Engine Pipeline

**Goal:** Compose computation across multiple engines.

**Steps:**

1. **Write multi-engine expression:**
```python
import xorq.api as xo
from xorq.vendor import ibis
from xorq.common.utils.ibis_utils import from_ibis
from xorq.caching import ParquetCache

# Start with DuckDB
duckdb_con = xo.connect()
source_table = duckdb_con.table("source")

# Process in DuckDB
filtered = source_table.filter(ibis._.status == "active")

# Move to SQLite for aggregation
sqlite_con = xo.sqlite.connect()
expr = (
    from_ibis(filtered)
    .into_backend(sqlite_con)
    .cache(ParquetCache.from_kwargs())
)
```

2. **Build and run:**
```bash
xorq build multi_engine.py -e expr
xorq run builds/<hash> -o output.parquet
```

**When to use:**
- Need specific engine features
- Optimizing performance
- Gradual migration between engines

---

### 5. ML Pipeline Development

**Goal:** Build deferred scikit-learn pipeline.

**Steps:**

1. **Initialize from sklearn template:**
```bash
xorq init -t sklearn
```

2. **Scaffold skill:**
```python
# Create sklearn_pipeline.py with your pipeline
# (Example structure available in examples/ directory)
```

3. **Edit pipeline:**
```python
from xorq.expr.ml.pipeline_lib import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create sklearn pipeline
sklearn_pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier()
)

# Convert to xorq (deferred)
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

# Use in expression
expr = xorq_pipeline.fit(train_data).predict(test_data)
```

4. **Build and run:**
```bash
xorq build sklearn_pipeline.py -e expr
xorq run builds/<hash> -o predictions.parquet
```

**When to use:**
- Machine learning workflows
- Need deferred training/prediction
- Reproducible ML pipelines

---

### 6. Serve Expression via Arrow Flight

**Goal:** Deploy expression as a service.

**Steps:**

1. **Build expression:**
```bash
xorq build expr.py -e expr
# Output: builds/abc123def/
```

2. **Identify node to unbind:**
```bash
# Look at expr.yaml to find the node hash you want to unbind
cat builds/abc123def/expr.yaml
# Find node like: '@read_31f0a5be3771'
```

3. **Start server:**
```bash
xorq serve-unbound builds/abc123def \
  --to_unbind_hash 31f0a5be3771 \
  --host localhost \
  --port 9002
```

4. **Call from client:**
```python
import xorq.api as xo

backend = xo.flight.connect(host="localhost", port=9002)
f = backend.get_exchange("default")

# Send data
data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
result = xo.memtable(data).pipe(f).execute()
```

**When to use:**
- Model serving
- Real-time inference
- Microservice architecture

---

### 7. Debug with Lineage

**Goal:** Understand data flow and debug issues.

**Steps:**

1. **Build and catalog:**
```bash
xorq build expr.py -e expr
xorq catalog add builds/<hash> --alias my-expr
```

2. **Show full lineage:**
```bash
xorq lineage my-expr

# Output:
# Lineage for column 'result':
# Field:result #1
# └── Aggregate #2
#     └── Filter #3
#         └── Read #4
```

3. **Inspect manifest:**
```bash
cat $(xorq catalog info my-expr --build-path)/expr.yaml
```

4. **Check cache paths:**
```python
# In Python
expr.ls.get_cache_paths()
```

**When to use:**
- Debugging data issues
- Understanding dependencies
- Validating logic

---

### 8. Agent-Guided Workflow

**Goal:** Follow agent onboarding process.

**Steps:**

1. **Get onboarding guide:**
```bash
xorq agents onboard
```

2. **Follow each phase:**

**Phase 1: Initialize**
```bash
xorq agents init
xorq agents prompt show xorq_core
```

**Phase 2: Build**
```bash
# Create penguins_demo.py with your expression
xorq build penguins_demo.py -e expr
xorq agents prompt show must_check_schema
```

**Phase 3: Catalog**
```bash
xorq catalog add builds/<hash> --alias my-expr
xorq catalog ls
```

**Phase 4: Test**
```bash
xorq run builds/<hash> -o test.parquet
xorq lineage my-expr
```

**Phase 5: Land**
```bash
git push
```

**When to use:**
- First time using xorq
- Need structured guidance
- Working with AI agents

---

### 9. Arrow IPC Streaming with DuckDB (Interactive Exploration)

**Goal:** Stream xorq output to DuckDB for interactive SQL analysis and exploration.

**Prerequisites:**
```bash
# Install DuckDB CLI (if using Nix)
# Add to flake.nix: pkgs.duckdb

# Install arrow extension in DuckDB
duckdb -c "INSTALL arrow FROM community; LOAD arrow;"
```

**Steps:**

1. **Find source nodes for composition:**
```bash
# List all source nodes in a pipeline
xorq catalog sources lineup-transform

# Output shows:
# Source 1:
#   Hash: d43ad87ea8a989f3495aab5dff0b5746
#   Type: xorq.expr.relations.Read
#   Columns: 22
```

2. **Simple: Single source to DuckDB:**
```bash
# Stream Arrow IPC to DuckDB for SQL analysis
xorq run batting-source -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow;
    SELECT playerID, SUM(H) as hits
    FROM read_arrow('/dev/stdin')
    GROUP BY playerID
    ORDER BY hits DESC
    LIMIT 10"
```

3. **Advanced: Chain with run-unbound:**
```bash
# Compose pipelines via Arrow IPC streaming
xorq run batting-source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound lineup-transform \
    --to_unbind_hash d43ad87ea8a989f3495aab5dff0b5746 \
    --typ xorq.expr.relations.Read \
    -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow;
    SELECT playerID, leadoff_fit, two_fit
    FROM read_arrow('/dev/stdin')
    ORDER BY leadoff_fit DESC
    LIMIT 10"
```

4. **Multi-stage composition:**
```bash
# Chain multiple transforms together
xorq run source1 -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound transform1 \
    --to_unbind_hash <hash1> \
    --typ xorq.expr.relations.Read \
    -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound transform2 \
    --to_unbind_hash <hash2> \
    --typ xorq.expr.relations.Read \
    -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow;
    SELECT * FROM read_arrow('/dev/stdin')"
```

**Key Points:**
- Use `-f arrow` for Arrow IPC format
- Use `-o /dev/stdout` to pipe (not default `/dev/null`)
- DuckDB must `LOAD arrow;` extension first
- Use `read_arrow('/dev/stdin')` in SQL
- Redirect stderr with `2>/dev/null` to avoid mixed output

**When to use:**
- Interactive SQL exploration of xorq results
- Ad-hoc analytics without writing Python
- Quick data validation and inspection
- Composing multiple xorq pipelines dynamically
- Testing pipeline outputs before productionizing

**Troubleshooting:**

*Problem: Empty results from pipeline*
```bash
# Debug: Check each stage separately
xorq run source -f arrow -o /tmp/stage1.arrow
duckdb -c "LOAD arrow; SELECT count(*) FROM '/tmp/stage1.arrow'"

# Common cause: Filter eliminates all rows
# Solution: Verify data matches filter criteria
```

*Problem: Arrow extension not found*
```bash
# Install from community repository
duckdb -c "INSTALL arrow FROM community; LOAD arrow;"
```

*Problem: Schema mismatch in run-unbound*
```bash
# Check source schema
xorq catalog schema source-alias

# Check expected schema
xorq catalog sources transform-alias
```

---

### 10. Composing Cataloged Expressions

**Goal:** Compose and reuse cataloged expressions using the Python catalog API.

**Why this pattern:**
- Catalog is the single source of truth for all expressions
- Python-native, simple API
- Direct execution without intermediate steps
- Type-safe with actual schemas

**Complete Workflow:**

**1. Load and Compose Expressions Directly (RECOMMENDED):**

```python
import xorq.api as xo

# Load cataloged expressions
source = xo.catalog.get("batting-source")

# Compose by chaining operations
lineup_analysis = (
    source
    .select("playerID", "yearID", "H", "AB")
    .mutate(batting_avg=xo._.H / xo._.AB)
    .filter(xo._.batting_avg > 0.250)
)

# Execute when ready
result = lineup_analysis.execute()

# Or build and catalog for reuse
# xorq build lineup.py -e lineup_analysis
# xorq catalog add builds/<hash> --alias lineup-analysis
```

**Alternative: Build Transforms Using Catalog Placeholders:**

Use placeholders when you want to define transforms without loading full expressions:

```python
import xorq.api as xo

# Get placeholder memtable with same schema (doesn't load full expression)
source_placeholder = xo.catalog.get_placeholder("batting-source", tag="source")
print(source_placeholder.schema())  # Shows schema without loading

# Build transform using placeholder - lightweight and fast
lineup_transform = (
    source_placeholder
    .select("playerID", "yearID", "H", "AB")
    .mutate(batting_avg=xo._.H / xo._.AB)
    .filter(xo._.batting_avg > 0.250)
)

# Build and catalog
# xorq build transform.py -e lineup_transform
# xorq catalog add builds/<hash> --alias lineup-transform

# Find the placeholder node hash using the tag
# xorq catalog sources lineup-transform  # Will show tag="source"
```

**2. Verify Schemas:**

```bash
# Check source output schema
xorq catalog schema batting-source

# Check transform input schema (shows placeholder)
xorq catalog sources lineup-transform
```

**3. CLI Composition via Arrow IPC:**

```bash
# Find source nodes
xorq catalog sources lineup-transform

# Compose via Arrow IPC streaming
xorq run batting-source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound lineup-transform \
    --to_unbind_hash d43ad87ea8a989f3495aab5dff0b5746 \
    --typ xorq.expr.relations.Read \
    -o output.parquet

# Stream to DuckDB for exploration
xorq run batting-source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound lineup-transform \
    --to_unbind_hash d43ad87ea8a989f3495aab5dff0b5746 \
    --typ xorq.expr.relations.Read \
    -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow;
    SELECT playerID, leadoff_fit
    FROM read_arrow('/dev/stdin')
    ORDER BY leadoff_fit DESC
    LIMIT 10"
```

**Schema Compatibility:**

```bash
# Check source output schema
xorq catalog schema batting-source

# Check transform input schema
xorq catalog sources lineup-transform

# Verify compatibility before composing
```

**Complete Example:**

See [examples/catalog_composition_example.py](../examples/catalog_composition_example.py) for a full working example.

**Advantages:**
- Direct composition with `xo.catalog.get()` (simplest, recommended)
- Fast placeholder creation with `xo.catalog.get_placeholder()` (for build-time transforms)
- Python-native composition for building transforms
- Type-safe schema inspection
- Tag-based composition for clarity

---

## Integration Workflows

### With Git Version Control

**Goal:** Version control xorq manifests.

**Steps:**

1. **Build expression:**
```bash
xorq build expr.py -e expr
```

2. **Commit manifest:**
```bash
git add builds/
git add expr.py
git commit -m "Add feature pipeline expression

Build hash: abc123def
Includes: filtering, aggregation, caching"
```

3. **Share with team:**
```bash
git push
# Team can rebuild from manifest
```

**When to use:**
- Team collaboration
- Reproducibility requirements
- Audit trail needed

---

## Troubleshooting Workflows

### Schema Mismatch

**Problem:** Build fails with schema error.

**Solution:**
```bash
# 1. Check agent prompt
xorq agents prompt show fix_schema_errors

# 2. Verify schema in Python
python -c "
import xorq.api as xo
con = xo.connect()
table = con.table('data')
print(table.schema())
"

# 3. Update expression to match schema
# 4. Rebuild
xorq build expr.py -e expr
```

### Import Errors

**Problem:** Import fails during build.

**Solution:**
```bash
# 1. Check prompt
xorq agents prompt show fix_import_errors

# 2. Verify vendored import
# Use: from xorq.vendor import ibis
# NOT: import ibis

# 3. Rebuild
xorq build expr.py -e expr
```

### Cache Not Working

**Problem:** Expression rebuilds instead of using cache.

**Solution:**
```bash
# 1. Check cache paths
python -c "
expr.ls.get_cache_paths()
"

# 2. Verify expression didn't change
# (Input-addressed: change = new hash)

# 3. Check cache directory exists
ls ~/.cache/xorq/parquet/
```

---

## Best Practices & Patterns

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

### Versioned Catalog Aliases

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

### UDF Name Parameters (Critical!)

**Pattern:**
```python
from xorq.expr.udf import agg, make_pandas_expr_udf
import xorq.expr.datatypes as dt

# ❌ WRONG - missing name parameter
train_udf = agg.pandas_df(
    fn=train_model,
    schema=data.schema(),
    return_type=dt.binary
    # ERROR: Missing name parameter
)

# ✅ CORRECT - always provide name
train_udf = agg.pandas_df(
    fn=train_model,
    schema=data.schema(),
    return_type=dt.binary,
    name='train_model'  # ← REQUIRED
)

predict_udf = make_pandas_expr_udf(
    computed_kwargs_expr=train_udf.on_expr(train),
    fn=predict,
    schema=data.schema(),
    return_type=dt.float64,
    name='predict_model'  # ← REQUIRED
)
```

**Why:**
- Both `agg.pandas_df()` and `make_pandas_expr_udf()` **require** explicit `name=` parameter
- Omitting it causes unclear errors during execution
- Name is used for debugging and manifest representation

**Reference:**
- [ML Pipelines - Example 7](ml-pipelines.md#example-7-advanced---udaf--exprscalarudf-for-unsupported-models)

---

For more troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
