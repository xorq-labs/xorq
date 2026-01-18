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

# 3. Check available skills
xorq agents templates list

# 4. Scaffold a skill if needed
xorq agents templates scaffold penguins_demo
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
xorq agents templates scaffold sklearn_pipeline
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
xorq agents templates scaffold penguins_demo
xorq build skills/penguins_demo.py -e expr
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

### 10. Building Transform Expressions with Memtable Pattern

**Goal:** Build transform expressions separately from their source data using memtable placeholders, enabling independent development and testing before composition.

**Why this pattern:**
- Develop transforms without depending on specific source expressions
- Test transform logic with sample data before applying to real sources
- Build and catalog source and transform expressions independently
- Compose them later via Arrow IPC streaming with `run-unbound`

**Complete Workflow:**

**Phase 1: Build Source Expression**

1. **Create source expression (source.py):**
```python
import xorq.api as xo
from xorq.vendor import ibis
from xorq.common.utils.ibis_utils import from_ibis
from xorq.caching import ParquetCache

# Connect to data source
con = xo.connect()
batting = con.table("batting")

# Check schema first (MANDATORY)
print(batting.schema())

# Build source expression
source_expr = (
    from_ibis(
        batting
        .filter(ibis._.yearID >= 2015)
        .select(["playerID", "yearID", "H", "AB", "teamID"])
    )
    .cache(ParquetCache.from_kwargs())
)

# Export for building
expr = source_expr
```

2. **Build and catalog source:**
```bash
# Build the source
xorq build source.py -e expr
# Output: builds/abc123def/

# Catalog it
xorq catalog add builds/abc123def --alias batting-source

# Verify the schema
xorq catalog schema batting-source
```

**Phase 2: Build Transform Expression with Memtable**

3. **Create transform expression using memtable (transform.py):**
```python
import xorq.api as xo
from xorq.vendor import ibis
from xorq.common.utils.ibis_utils import from_ibis
from xorq.caching import ParquetCache

# Create sample data that matches expected source schema
# This is your placeholder - it defines the interface
sample_data = {
    "playerID": ["player1", "player2", "player3"],
    "yearID": [2020, 2020, 2021],
    "H": [150, 120, 180],
    "AB": [500, 450, 550],
    "teamID": ["NYY", "BOS", "LAD"]
}

# Create memtable as source placeholder
source_table = xo.memtable(sample_data)

# Check the schema (MANDATORY)
print(source_table.schema())

# Build your transform logic on the memtable
transform_expr = (
    from_ibis(
        source_table
        .mutate(
            batting_avg=ibis._.H / ibis._.AB,
            at_bats_per_year=ibis._.AB
        )
        .filter(ibis._.batting_avg > 0.250)
        .group_by("playerID")
        .agg(
            total_hits=ibis._.H.sum(),
            total_at_bats=ibis._.AB.sum(),
            years_played=ibis._.yearID.nunique(),
            avg_batting_avg=ibis._.batting_avg.mean()
        )
        .mutate(
            career_avg=ibis._.total_hits / ibis._.total_at_bats
        )
    )
    .cache(ParquetCache.from_kwargs())
)

# Export for building
expr = transform_expr
```

4. **Build and catalog transform:**
```bash
# Build the transform
xorq build transform.py -e expr
# Output: builds/xyz789abc/

# Catalog it
xorq catalog add builds/xyz789abc --alias batting-transform

# Find source nodes that need to be unbound
xorq catalog sources batting-transform
# Output shows the memtable node hash (e.g., d43ad87ea8a989f3495aab5dff0b5746)
```

**Phase 3: Compose Source and Transform**

5. **Test with sample data first:**
```bash
# Run transform with its built-in memtable (for testing)
xorq run batting-transform -o test_output.parquet

# Verify output
duckdb test_output.parquet -c "SELECT * FROM test_output LIMIT 5"
```

6. **Compose with real source via Arrow IPC:**
```bash
# Pipe source into transform, replacing memtable with real data
xorq run batting-source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound batting-transform \
    --to_unbind_hash d43ad87ea8a989f3495aab5dff0b5746 \
    --typ xorq.expr.relations.Read \
    -o final_output.parquet

# Verify final output
duckdb final_output.parquet -c "SELECT * FROM final_output ORDER BY career_avg DESC LIMIT 10"
```

7. **Interactive exploration with DuckDB:**
```bash
# Stream through the entire pipeline into DuckDB for ad-hoc queries
xorq run batting-source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound batting-transform \
    --to_unbind_hash d43ad87ea8a989f3495aab5dff0b5746 \
    --typ xorq.expr.relations.Read \
    -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow;
    SELECT
      playerID,
      career_avg,
      total_hits,
      years_played
    FROM read_arrow('/dev/stdin')
    WHERE years_played >= 3
    ORDER BY career_avg DESC
    LIMIT 20"
```

**Key Benefits:**

1. **Independent Development:**
   - Source and transform can be developed by different people
   - Transform can be tested without waiting for source data
   - Each component is cataloged and versioned independently

2. **Flexible Composition:**
   - Same transform can be applied to different sources (if schemas match)
   - Multiple transforms can be chained together
   - Easy to swap components without rebuilding

3. **Iterative Testing:**
   - Test transform logic with small sample data (memtable)
   - Verify transform works correctly before applying to large datasets
   - Debug issues in isolation

4. **Interactive Exploration:**
   - Stream composed pipeline to DuckDB for ad-hoc SQL queries
   - Explore results without writing Python
   - Rapid iteration on analysis queries

**Schema Compatibility Checklist:**

```bash
# Before composing, verify schemas match:

# 1. Check source output schema
xorq catalog schema batting-source

# 2. Check transform expected input schema
xorq catalog sources batting-transform

# 3. Ensure column names and types match
# If mismatch, update transform.py memtable to match source schema
```

**Common Patterns:**

**Pattern A: Multiple Sources, One Transform**
```bash
# Same transform applied to different sources
xorq run source-2015-2019 -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound batting-transform --to_unbind_hash <hash> --typ xorq.expr.relations.Read -o output_2015_2019.parquet

xorq run source-2020-2024 -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound batting-transform --to_unbind_hash <hash> --typ xorq.expr.relations.Read -o output_2020_2024.parquet
```

**Pattern B: Chained Transforms**
```bash
# Multiple transforms in sequence
xorq run source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound transform1 --to_unbind_hash <hash1> --typ xorq.expr.relations.Read -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound transform2 --to_unbind_hash <hash2> --typ xorq.expr.relations.Read -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound transform3 --to_unbind_hash <hash3> --typ xorq.expr.relations.Read -o final.parquet
```

**Pattern C: Branch and Merge**
```bash
# Source splits into multiple transforms, results analyzed separately
xorq run source -f arrow -o /tmp/source.arrow

# Branch 1: Statistical analysis
cat /tmp/source.arrow | xorq run-unbound stats-transform --to_unbind_hash <hash> --typ xorq.expr.relations.Read -o stats.parquet

# Branch 2: ML features
cat /tmp/source.arrow | xorq run-unbound features-transform --to_unbind_hash <hash> --typ xorq.expr.relations.Read -o features.parquet
```

**When to use:**
- Building data pipelines with multiple stages
- Developing transforms before source data is ready
- Creating reusable transform components
- Testing complex transformations with sample data
- Composing pipelines dynamically at runtime

**Advanced: Python API for Programmatic Replacement**

For advanced use cases, you can replace nodes programmatically in Python:

```python
from xorq.flight.exchanger import replace_one_unbound
from xorq.common.utils.node_utils import expr_to_unbound, replace_by_expr_hash
import xorq.api as xo

# Load unbound expression
unbound_expr = xo.load_expr("builds/transform-hash")

# Create replacement table
replacement_table = xo.memtable({"col1": [1, 2], "col2": [3, 4]})

# Replace unbound node with table
bound_expr = replace_one_unbound(unbound_expr, replacement_table)

# Execute
result = bound_expr.execute()
```

**Utilities:**
- `replace_one_unbound(unbound_expr, table)` - Replace single unbound table with data
- `expr_to_unbound(expr, hash, tag, typs)` - Convert expression to unbound form
- `replace_by_expr_hash(expr, hash, replace_with, typs)` - Replace by hash

**Reference:** `python/xorq/flight/exchanger.py` and `python/xorq/common/utils/node_utils.py`

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

For more troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
