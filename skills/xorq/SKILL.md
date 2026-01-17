---
name: xorq
description: >
  Compute manifest and composable tools for ML. Build, catalog, and serve deferred
  expressions with input-addressed caching, multi-engine execution, and Arrow-native
  data flow. Use for ML pipelines, feature engineering, and model serving.
allowed-tools: "Read,Bash(xorq:*),Bash(python:*)"
version: "0.1.0"
author: "Xorq Labs <https://github.com/xorq-labs>"
license: "Apache-2.0"
---

# Xorq - Manifest-Driven Compute for ML

A compute manifest system that provides persistent, cacheable, and portable expressions for ML workflows. Expressions are tools that compose via Arrow.

## Core Concepts

**Expression** - Deferred computation graph built with Ibis that can execute across multiple engines
**Manifest** - YAML representation of an expression with lineage, caching, and metadata
**Build** - Versioned artifact containing manifest, cached data, and dependencies
**Catalog** - Registry for discovering and reusing builds across sessions

## Prerequisites

```bash
xorq --version  # Requires xorq CLI
python -m pip show xorq  # Python package installed
```

- **xorq CLI** installed and in PATH
- **Python environment** with xorq package
- **Git repository** (optional, for version control)

## Session Protocol

### 1. Initialize Project
```bash
xorq init -t penguins  # Start with template
# Or for agent workflows
xorq agent onboard  # Guided workflow
```

### 2. Build Expression
```bash
# Create/edit expression file (expr.py)
xorq build expr.py -e expr  # Build to get manifest
# Output: builds/<hash>/
```

### 3. Catalog Build
```bash
xorq catalog add builds/<hash> --alias my-expr
xorq catalog ls  # List cataloged builds
```

### 4. Run Expression
```bash
xorq run builds/<hash> -o output.parquet
# Or by alias
xorq run $(xorq catalog info my-expr --build-path) -o output.parquet
```

### 5. Debug with Lineage
```bash
xorq lineage my-expr  # Column-level lineage
```

### 6. Serve (Optional)
```bash
xorq serve-unbound builds/<hash>/ \
  --to_unbind_hash <node-hash> \
  --host localhost --port 9002
```

## CLI Reference

**Run `xorq --help`** for all commands.
**Run `xorq <command> --help`** for specific usage.

Essential commands:
- `xorq init` - Initialize project from template
- `xorq build` - Build expression to manifest
- `xorq run` - Execute build
- `xorq catalog` - Manage build registry
- `xorq lineage` - Show column lineage
- `xorq agent` - Agent-native helpers

### Agent Commands

| Command | Purpose |
|---------|---------|
| `xorq agent onboard` | Guided workflow for agents |
| `xorq agent prompt list` | List available prompts |
| `xorq agent prompt show <name>` | Show specific prompt |
| `xorq agent templates list` | List registered templates |
| `xorq agent templates scaffold <name>` | Generate starter file |

## Core Workflow Patterns

### Pattern 1: Build and Run
```bash
# 1. Build expression
xorq build expr.py -e expr

# 2. Catalog it
xorq catalog add builds/<hash> --alias my-pipeline

# 3. Run it
xorq run builds/<hash> -o results.parquet
```

### Pattern 2: Template-Based Start
```bash
# 1. Initialize from template
xorq init -t sklearn

# 2. Use scaffolded skill
xorq agent templates scaffold sklearn_pipeline

# 3. Edit generated file, then build
xorq build sklearn_pipeline.py -e pipeline
```

### Pattern 3: Multi-Engine Composition
```python
import xorq.api as xo
from xorq.vendor import ibis

# Start with one backend
con = xo.connect()  # DuckDB by default
table = con.table("data")

# Move to another backend
expr = from_ibis(table).into_backend(xo.sqlite.connect())

# xorq handles data transit via Arrow
```

## Expression Construction

### Basic Structure
```python
import xorq.api as xo
from xorq.vendor import ibis
from xorq.common.utils.ibis_utils import from_ibis
from xorq.caching import ParquetCache

# Connect to data source
con = xo.connect()
table = con.table("my_table")

# IMPORTANT: Always check schema first
print(table.schema())

# Build deferred expression
expr = (
    table
    .filter(ibis._.column.notnull())
    .group_by("key")
    .agg(metric=ibis._.value.mean())
)

# Add caching
cached_expr = (
    from_ibis(expr)
    .cache(ParquetCache.from_kwargs())
)
```

### Key Rules
1. **Always check schema** - Run `print(table.schema())` before building expressions
2. **Use `from xorq.vendor import ibis`** - Not `import ibis` directly
3. **Deferred execution** - Expressions don't run until `.execute()` or build
4. **Cache strategically** - Add `.cache()` at expensive computation boundaries
5. **Input-addressed** - Same inputs = same hash = cache hit

## Agent-Native Features

xorq has built-in agent support with prompts, skills, and workflows.

### Prompts (Context Blocks)
```bash
# List all prompts
xorq agent prompt list

# Core tier prompts
xorq agent prompt list --tier core

# Show specific prompt
xorq agent prompt show xorq_core
```

Prompts provide guidance on:
- Planning and sequential execution
- Ibis vendor import rules
- Schema checking requirements
- Error fixes and workarounds
- ML patterns and optimizations

### Skills (Templates)
```bash
# List available skills
xorq agent templates list

# Show skill details
xorq agent templates show sklearn_pipeline

# Scaffold from skill
xorq agent templates scaffold penguins_demo
```

Current skills:
- `penguins_demo` - Minimal multi-engine example
- `sklearn_pipeline` - Deferred sklearn with train/predict
- `cached_fetcher` - Hydrate and cache upstream tables

### Onboarding Workflow
```bash
xorq agent onboard
```

Provides step-by-step guidance:
1. **init** - Project setup
2. **build** - Author expressions
3. **catalog** - Publish artifacts
4. **test** - Validate builds
5. **land** - Session completion

## Integration with bd (beads)

If using bd for issue tracking:
```bash
# At start of session
bd ready  # Find work
bd show <id>  # Get context
bd update <id> --status in_progress

# During work
bd update <id> --notes "Built expr for feature X, hash: abc123"

# At end
bd close <id> --reason "Expression built and cataloged"
bd sync  # Persist to git
```

## Key Differences from Other Tools

| xorq | Traditional Approach |
|------|---------------------|
| Manifest = context | Metadata in separate DB |
| Input-addressed cache | TTL or manual invalidation |
| Multi-engine compose | Engine lock-in or rewrites |
| Arrow RecordBatch streams | Task DAGs with state |
| Build = portable artifact | Orchestrator-specific config |

## Common Patterns

### Cache Expensive Operations
```python
# Cache after remote computation
expr = (
    from_ibis(remote_query)
    .cache(ParquetCache.from_kwargs())
)
```

### Schema Inspection
```python
# ALWAYS check schema before building
con = xo.connect()
table = con.table("data")
print(table.schema())  # Required!
```

### Multi-Engine Flow
```python
# Start in DuckDB, move to SQLite
duckdb_con = xo.connect()  # Default is DuckDB
sqlite_con = xo.sqlite.connect()

expr = (
    from_ibis(duckdb_con.table("source"))
    .into_backend(sqlite_con)
)
```

### ML Pipeline
```python
from xorq.expr.ml.pipeline_lib import Pipeline

sklearn_pipeline = ...  # Your sklearn pipeline
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

# Pipeline is now deferred
```

## Resources

| Resource | Content |
|----------|---------|
| [CLI_REFERENCE.md](resources/CLI_REFERENCE.md) | Complete command syntax |
| [WORKFLOWS.md](resources/WORKFLOWS.md) | Step-by-step patterns |
| [TROUBLESHOOTING.md](resources/TROUBLESHOOTING.md) | Common issues |
| [PATTERNS.md](resources/PATTERNS.md) | Best practices |

## Full Documentation

- **xorq agent onboard**: Guided agent workflow
- **GitHub**: [github.com/xorq-labs/xorq](https://github.com/xorq-labs/xorq)
- **Docs**: [docs.xorq.dev](https://docs.xorq.dev)

## Version Compatibility

| Version | Features |
|---------|----------|
| v0.1.0+ | Agent prompts, skills, onboarding |
| Earlier | Core: build, run, catalog, lineage |
