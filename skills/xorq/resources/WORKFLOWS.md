# Xorq Workflows

Step-by-step patterns for common xorq tasks.

## Basic Workflows

### 1. New Project from Template

**Goal:** Start a new xorq project using a template.

**Steps:**
```bash
# 1. Initialize from template
xorq init -t penguins

# 2. Explore generated files
ls
# skills/penguins_demo.py
# (if --agent used: AGENTS.md, CLAUDE.md)

# 3. Check available skills
xorq agent templates list

# 4. Scaffold a skill if needed
xorq agent templates scaffold penguins_demo
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
```bash
xorq agent templates scaffold sklearn_pipeline
# Creates: skills/sklearn_pipeline.py
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
xorq agent onboard
```

2. **Follow each phase:**

**Phase 1: Initialize**
```bash
xorq init --agent
xorq agent prompt show xorq_core
```

**Phase 2: Build**
```bash
xorq agent templates scaffold penguins_demo
xorq build skills/penguins_demo.py -e expr
xorq agent prompt show must_check_schema
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
# If using bd:
bd sync
git push
```

**When to use:**
- First time using xorq
- Need structured guidance
- Working with AI agents

---

## Integration Workflows

### With bd (beads) Issue Tracker

**Goal:** Track xorq work with bd issues.

**Steps:**

1. **Start session:**
```bash
bd ready
bd show XQ-123
bd update XQ-123 --status in_progress
```

2. **Build expression:**
```bash
xorq build expr.py -e expr
# Output: builds/abc123def/
```

3. **Update issue:**
```bash
bd update XQ-123 --notes "
Built expression for feature pipeline
Build hash: abc123def
Cataloged as: feature-pipeline-v1
"
```

4. **Complete work:**
```bash
xorq catalog add builds/abc123def --alias feature-pipeline-v1
bd close XQ-123 --reason "Expression built and cataloged"
bd sync
```

**When to use:**
- Multi-session work
- Team collaboration
- Persistent context needed

---

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
xorq agent prompt show fix_schema_errors

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
xorq agent prompt show fix_import_errors

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

For more troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
