# Xorq - Manifest-Driven Compute for ML

A compute manifest system providing persistent, cacheable, and portable expressions for ML workflows. Expressions are tools that compose via Arrow.

## Core Concepts

**Expression** - Deferred computation graph built with Ibis, executes across multiple engines
**Manifest** - YAML representation with lineage, caching, and metadata
**Build** - Versioned artifact containing manifest, cached data, and dependencies
**Catalog** - Registry for discovering and reusing builds across sessions

## Quick Start

```bash
# Initialize (one-time setup)
xorq init -t penguins
# Or for agent workflows
xorq agents onboard

# Core workflow
print(table.schema())           # ALWAYS check schema first
xorq build expr.py -e expr      # Build expression
xorq catalog add builds/<hash> --alias my-expr
xorq run my-expr -o output.parquet
```

## Essential CLI Commands

| Command | Purpose |
|---------|---------|
| `xorq init -t <template>` | Initialize project from template |
| `xorq build <file> -e <expr>` | Build expression to manifest |
| `xorq run <alias>` | Execute cataloged build |
| `xorq catalog add/ls` | Manage build registry |
| `xorq lineage <alias>` | Show column-level lineage |
| `xorq agents prime` | Get workflow context (source of truth) |
| `xorq agents onboard` | Guided workflow for agents |
| `xorq agents templates list` | List available templates |

**Full reference:** Run `xorq --help` or see [resources/CLI_REFERENCE.md](resources/CLI_REFERENCE.md)

## Python API Essentials

### Imports and Connection

```python
import xorq.api as xo
from xorq.vendor import ibis  # ALWAYS use xorq.vendor.ibis

# Connect to backend
con = xo.connect()  # DuckDB default
# Or: xo.connect("snowflake://...")
```

### Expression Building Patterns

```python
# MANDATORY: Check schema first
table = con.table("data")
print(table.schema())  # Required before any operations

# Build deferred expression
expr = (
    table
    .filter(xo._.column.notnull())
    .select("id", "value", "category")
    .group_by("category")
    .agg(total=xo._.value.sum())
)

# Execute when ready
result = expr.execute()
```

### Deferred Loading (Large Files)

```python
from xorq.common.utils.defer_utils import deferred_read_parquet

# Lazy loading - doesn't read until execute()
expr = deferred_read_parquet("large.parquet", con, "data")
```

### Caching Expensive Operations

```python
from xorq.caching import ParquetCache
from xorq.common.utils.ibis_utils import from_ibis

cached_expr = (
    from_ibis(expensive_query)
    .cache(ParquetCache.from_kwargs())  # Cache here
)
```

### ML Pipeline (Preferred Pattern)

```python
import toolz
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xorq.expr.ml.pipeline_lib import Pipeline

# 1. Create as_struct helper (REQUIRED)
@toolz.curry
def as_struct(expr, name=None):
    struct = xo.struct({c: expr[c] for c in expr.columns})
    return struct.name(name) if name else struct

# 2. Create sklearn pipeline
sklearn_pipeline = SkPipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor())
])
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

# 3. Fit on training data
fitted = xorq_pipeline.fit(train, features=FEATURES, target="target")

# 4. Predict with struct pattern (MANDATORY)
predictions = (
    test
    .mutate(as_struct(name="original_row"))  # Pack columns
    .pipe(fitted.predict)                     # Predict
    .unpack("original_row")                   # Unpack
    .mutate(predicted=_.predicted)            # Use result
)
```

## Critical Rules

### Schema Checks (NON-NEGOTIABLE)

```python
# ✅ ALWAYS do this first
table = con.table("data")
print(table.schema())  # Mandatory before operations

# Then build expression
expr = table.filter(xo._.UPPERCASE_COL > 0)  # Match case from schema
```

### Column Case Sensitivity

- **Snowflake**: UPPERCASE columns
- **DuckDB**: lowercase columns
- **Match exactly** as shown in schema

### Deferred Execution Only

```python
# ✅ Good: Deferred xorq expressions
expr = table.filter(xo._.status == "active")

# ❌ Avoid: Pandas/NumPy scripts
df = pd.read_parquet("data.parquet")  # Eager loading
```

### Catalog Management

```bash
# After building, always catalog
xorq catalog add builds/<hash> --alias my-pipeline

# View cataloged builds
xorq catalog ls

# CRITICAL: Commit catalog before session end
git add .xorq/catalog.yaml builds/
git commit -m "Add pipeline to catalog"
```

## Session Protocol

### Start Session

```bash
# Get dynamic workflow context
xorq agents prime
```

### Development Loop

```bash
# 1. Check schema (mandatory)
python -c "import xorq.api as xo; con = xo.connect(); print(con.table('data').schema())"

# 2. Build expression
xorq build expr.py -e expr

# 3. Catalog the build
xorq catalog add builds/<hash> --alias my-expr

# 4. Test it
xorq run my-expr -o test.parquet

# 5. Check lineage
xorq lineage my-expr
```

### Close Session (CRITICAL)

```bash
# 1. Commit catalog and builds
git add .xorq/catalog.yaml builds/
git commit -m "Update catalog"

# 2. Push changes
git push

# 3. Generate handoff
xorq agents prime
```

## Advanced Workflow Patterns

### DuckDB CLI Exploration

**Pattern:** Stream xorq outputs through Arrow IPC to DuckDB for interactive SQL exploration.

```bash
# Simple: Stream source to DuckDB
xorq run source -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow; SELECT * FROM read_arrow('/dev/stdin') LIMIT 10"

# Advanced: Compose pipeline and explore interactively
xorq run source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound transform \
    --to_unbind_hash <hash> \
    --typ xorq.expr.relations.Read \
    -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow;
    SELECT col1, COUNT(*) FROM read_arrow('/dev/stdin')
    GROUP BY col1"
```

**When to use:**
- Interactive SQL exploration without writing Python
- Ad-hoc data validation and quick analysis
- Testing pipeline outputs rapidly

**Reference:** [Workflows #9](resources/WORKFLOWS.md#9-arrow-ipc-streaming-with-duckdb-interactive-exploration)

---

### Catalog Composition Pattern (PREFERRED)

**Pattern:** Compose cataloged expressions directly in Python.

```python
# Load from catalog
import xorq.api as xo

# Direct catalog loading (preferred method)
source = xo.catalog.get("my-source")
transform = xo.catalog.get("my-transform")

# Inspect sources for composition
sources = xo.catalog.list_source_nodes(transform)
for src in sources:
    print(f"Node: {src['name']}, Hash: {src['hash'][:12]}...")
    print(f"Schema: {src['schema']}")

# Create composable transform
composable = xo.catalog.replace_as_root_memtable(
    transform,
    node_hash=sources[0]['hash']
)

# Build and catalog the composed expression
# xorq build pipeline.py -e composable
# xorq catalog add builds/<hash> --alias my-pipeline
```

**Why this pattern:**
- Uses catalog as single source of truth
- Python-native composition (no CLI piping needed)
- Type-safe with actual schemas
- Programmatically discoverable with list_source_nodes

**Reference:** See [examples/catalog_composition_example.py](resources/examples.md)

---

### Memtable Placeholder Pattern (Alternative)

**Pattern:** Build transforms independently when source not yet cataloged.

```python
# In transform.py - Define transform with memtable placeholder
import xorq.api as xo
from xorq.vendor import ibis
from xorq.common.utils.ibis_utils import from_ibis

# Sample data matching expected source schema
sample_data = {"col1": [1, 2], "col2": [3, 4]}
source = xo.memtable(sample_data)
print(source.schema())  # Check schema

# Build transform on memtable
expr = from_ibis(
    source
    .mutate(total=ibis._.col1 + ibis._.col2)
    .filter(ibis._.total > 3)
)
```

```bash
# Build transform with memtable
xorq build transform.py -e expr
xorq catalog add builds/<hash> --alias my-transform

# Find memtable node hash
xorq catalog sources my-transform

# Compose with real source later
xorq run real-source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound my-transform \
    --to_unbind_hash <hash> \
    --typ xorq.expr.relations.Read \
    -o output.parquet
```

**When to use memtable pattern:**
- Source data doesn't exist yet
- Building transforms independently before data pipeline ready
- Testing transform logic with sample data

**Key principle:** Use `xo.catalog.get()` first. Fall back to memtable only when source not cataloged.

**Reference:** [Workflows #10](resources/WORKFLOWS.md#10-building-transform-expressions-with-memtable-pattern) | [Patterns](resources/PATTERNS.md#memtable-placeholder-pattern)

---

## Common Expression Patterns

### Filtering and Selection

```python
# Filter with conditions
filtered = table.filter([
    xo._.age > 18,
    xo._.status == "active"
])

# Select columns
selected = table.select("id", "name", "value")

# Add computed columns
mutated = table.mutate(
    value_squared=xo._.value ** 2,
    full_name=xo._.first + " " + xo._.last
)
```

### Aggregations

```python
# Group and aggregate
agg = table.group_by("category").agg(
    count=xo._.id.count(),
    total=xo._.value.sum(),
    avg=xo._.value.mean()
)
```

### Joins

```python
# Inner join
joined = left.join(
    right,
    left.id == right.left_id,
    how="inner"
)
```

### Window Functions

```python
# Rank within groups
ranked = table.mutate(
    rank=xo._.value.rank().over(
        group_by="category",
        order_by=xo._.value.desc()
    )
)
```

## Agent-Native Features

### Prompts (Workflow Context)

```bash
# List all prompts
xorq agents prompt list

# Show specific prompt
xorq agents prompt show xorq_core

# Get workflow context (use this!)
xorq agents prime
```

### Templates (Starter Code)

```bash
# List available templates
xorq agents templates list

# Show template details
xorq agents templates show sklearn_pipeline

# Scaffold from template
xorq agents templates scaffold penguins_demo
```

Available templates:
- `penguins_demo` - Minimal multi-engine example
- `sklearn_pipeline` - Deferred sklearn with train/predict
- `cached_fetcher` - Hydrate and cache upstream tables

### Onboarding Workflow

```bash
xorq agents onboard
```

Steps: **init → build → catalog → test → land**

## Multi-Engine Support

```python
# Start with DuckDB
duckdb_con = xo.connect()

# Move to Snowflake
snowflake_con = xo.connect("snowflake://...")

# Compose across backends
expr = (
    from_ibis(duckdb_con.table("local_data"))
    .into_backend(snowflake_con)  # Transit via Arrow
)
```

## Resources (Progressive Disclosure)

| Resource | Content |
|----------|---------|
| [Expression API](resources/expression-api.md) | Fluent transformations, filters, joins, window functions |
| [ML Pipelines](resources/ml-pipelines.md) | Sklearn integration, deferred fit/predict, pipelines |
| [Caching Strategies](resources/caching.md) | Performance optimization, storage backends |
| [UDFs & Flight Servers](resources/udf-udxf.md) | Custom functions, distributed processing |
| [Examples](resources/examples.md) | End-to-end working examples |
| [CLI Reference](resources/CLI_REFERENCE.md) | Complete command documentation |
| [Workflows](resources/WORKFLOWS.md) | Step-by-step patterns |
| [Troubleshooting](resources/TROUBLESHOOTING.md) | Common issues and fixes |

## Troubleshooting

### Expression won't execute
- Check schema: `print(table.schema())`
- Verify column names match case
- Check connection: `con.list_tables()`

### Column not found
- Run `print(table.schema())` first
- Match exact case (Snowflake=UPPERCASE, DuckDB=lowercase)

### Cache not working
- Verify cache directory exists
- Check expression is identical (cache key = expression hash)

### Import errors
- Use `from xorq.vendor import ibis` (not `import ibis`)
- Ensure xorq is installed: `pip show xorq`

## Best Practices

1. **Always check schema first** - `print(table.schema())`
2. **Use deferred loading** - `deferred_read_parquet()` for large files
3. **Cache strategically** - After expensive operations
4. **Push filters early** - Filter before expensive transformations
5. **Batch operations** - Combine mutations instead of sequential
6. **Catalog everything** - Register all builds for reuse
7. **Commit catalog** - Always `git add .xorq/catalog.yaml`

## Key Differences from Other Tools

| xorq | Traditional |
|------|------------|
| Manifest = context | Metadata in separate DB |
| Input-addressed cache | TTL or manual invalidation |
| Multi-engine compose | Engine lock-in |
| Arrow RecordBatch streams | Task DAGs with state |
| Build = portable artifact | Orchestrator config |

## Documentation Links

- **Workflow Context**: `xorq agents prime` (dynamic, context-aware)
- **GitHub**: [github.com/xorq-labs/xorq](https://github.com/xorq-labs/xorq)
- **Docs**: [docs.xorq.dev](https://docs.xorq.dev)
