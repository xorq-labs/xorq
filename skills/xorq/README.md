# Xorq Skill for Claude Code

A comprehensive skill for using [xorq](https://github.com/xorq-labs/xorq) manifest-driven compute with Claude Code.

## What This Skill Does

This skill teaches Claude Code how to use xorq effectively for:
- **ML pipeline development** - Build deferred expressions with caching
- **Multi-engine execution** - Compose across DuckDB, SQLite, Snowflake, etc.
- **Manifest-based versioning** - Git-trackable compute artifacts
- **Input-addressed caching** - Deterministic cache keys from content
- **Agent-native workflows** - Prompts, skills, and onboarding guidance

## Installation

### Global Installation
Copy the `xorq/` skill directory to your global Claude Code skills location:

```bash
# Global installation
cp -r skills/xorq ~/.claude/skills/

# Verify installation
ls ~/.claude/skills/xorq/SKILL.md
```

### Project-Local Installation
For project-specific use:

```bash
# In your project root
mkdir -p .claude/skills
cp -r /path/to/xorq/skills/xorq .claude/skills/

# Verify
ls .claude/skills/xorq/SKILL.md
```

## When Claude Uses This Skill

The skill activates when conversations involve:
- "build expression", "xorq pipeline", "deferred computation"
- "multi-engine", "cache results", "manifest"
- ML workflows with scikit-learn, feature engineering
- Data pipelines needing versioning and reproducibility
- "Arrow Flight", "serve model", "catalog builds"

## Prerequisites

Before using this skill, ensure:

1. **xorq CLI installed**:
   ```bash
   pip install xorq
   xorq --version
   ```

2. **Python environment** with xorq package:
   ```bash
   python -m pip show xorq
   ```

## File Structure

```
xorq/
├── SKILL.md                 # Main skill file (Claude reads this first)
├── README.md                # This file (for humans)
└── resources/               # Detailed documentation (optional)
    ├── CLI_REFERENCE.md     # Command reference
    ├── WORKFLOWS.md         # Step-by-step patterns & best practices
    └── TROUBLESHOOTING.md   # Common issues
```

## Key Concepts

### Manifest = Context

Every xorq expression becomes a structured, input-addressed YAML manifest. The manifest IS the context - no separate metadata store needed.

```yaml
nodes:
  '@read_31f0a5be3771':
    op: Read
    source: builds/.../data.parquet

  '@filter_23e7692b7128':
    op: Filter
    parent: '@read_31f0a5be3771'
    predicates: [NotNull(column)]
```

### Input-Addressed Caching

Same inputs = same hash = cache hit. No expiration logic, no invalidation to debug.

```python
expr = from_ibis(query).cache(ParquetCache.from_kwargs())
# Hash automatically computed from expression graph
```

### Multi-Engine Composition

One expression, many engines. xorq handles data transit:

```python
# Start in DuckDB, move to SQLite
expr = from_ibis(duckdb_table).into_backend(sqlite_con)
```

### Build = Artifact

Build directory contains everything needed to reproduce:

```
builds/28ecab08754e/
├── expr.yaml           # Manifest
├── metadata.json       # Build info
├── profiles.yaml       # Engine configs
└── database_tables/    # Cached data
```

## Quick Start

### 1. Initialize Project
```bash
# From template
xorq init -t penguins

# For agent workflows
xorq agents onboard
```

### 2. Write Expression
```python
import xorq.api as xo
from xorq.vendor import ibis
from xorq.common.utils.ibis_utils import from_ibis
from xorq.caching import ParquetCache

con = xo.connect()
table = con.table("data")
print(table.schema())  # Always check first!

expr = (
    from_ibis(table.filter(ibis._.col.notnull()))
    .cache(ParquetCache.from_kwargs())
)
```

### 3. Build and Catalog
```bash
xorq build expr.py -e expr
xorq catalog add builds/<hash> --alias my-expr
```

### 4. Run
```bash
xorq run builds/<hash> -o output.parquet
```

## Agent Features

xorq has first-class agent support:

### Prompts
```bash
# List all prompts
xorq agents prompt list

# Show specific guidance
xorq agents prompt show xorq_core
xorq agents prompt show must_check_schema
```

Prompts cover:
- Core rules (deferred execution, imports)
- Reliability (schema checks, error fixes)
- Advanced (ML patterns, optimizations)

### Skills
```bash
# List available skills
xorq agents templates list

# Scaffold from skill
xorq agents templates scaffold sklearn_pipeline
```

Built-in skills:
- `penguins_demo` - Minimal example
- `sklearn_pipeline` - ML pipeline template
- `cached_fetcher` - Data hydration pattern

### Onboarding
```bash
xorq agents onboard
```

Step-by-step workflow:
1. **init** - Project setup
2. **build** - Expression authoring
3. **catalog** - Artifact publishing
4. **test** - Validation
5. **land** - Session completion

## Common Workflows

### Build, Catalog, Run
```bash
# 1. Build
xorq build my_expr.py -e expr

# 2. Catalog
xorq catalog add builds/abc123 --alias my-pipeline

# 3. Run
xorq run builds/abc123 -o output.parquet

# 4. Inspect lineage
xorq lineage my-pipeline
```

### Template-Based Development
```bash
# Start from template
xorq init -t sklearn

# Scaffold skill
xorq agents templates scaffold sklearn_pipeline

# Edit, build, run
xorq build sklearn_pipeline.py -e pipeline
```

### Multi-Engine Pipeline
```python
# DuckDB → compute → SQLite → cache
duckdb_con = xo.connect()
sqlite_con = xo.sqlite.connect()

expr = (
    from_ibis(duckdb_con.table("source"))
    .into_backend(sqlite_con)
    .cache(ParquetCache.from_kwargs())
)
```

## Integration with bd

If using bd for issue tracking:

```bash
# Session start
bd ready
bd show <id>
bd update <id> --status in_progress

# During work
xorq build expr.py -e expr
bd update <id> --notes "Built expr, hash: abc123"

# Session end
xorq catalog add builds/abc123 --alias feature-x
bd close <id> --reason "Expression built and cataloged"
bd sync
```

## Troubleshooting

### Schema Errors
Always check schema before building:
```python
print(table.schema())  # Required!
```

Use agent prompts for fixes:
```bash
xorq agents prompt show fix_schema_errors
```

### Import Errors
Use vendored ibis:
```python
from xorq.vendor import ibis  # ✓ Correct
# NOT: import ibis            # ✗ Wrong
```

### Cache Not Working
Cache keys are content-addressed. If expression changes, hash changes:
```bash
# Inspect cache paths
expr.ls.get_cache_paths()
```

### More Help
```bash
# All prompts
xorq agents prompt list

# Specific error fixes
xorq agents prompt show fix_attribute_errors
xorq agents prompt show fix_data_errors
```

## Differences from Similar Tools

| Feature | xorq | Airflow/Prefect | Feature Stores |
|---------|------|-----------------|----------------|
| State | Stateless | Stateful tasks | Stateful registry |
| Caching | Input-addressed | Time-based | Manual |
| Lineage | In manifest | Separate tool | Separate tool |
| Versioning | Git-friendly YAML | Database | Database |
| Retry | Just rerun | Complex state | N/A |
| Composition | Multi-engine | Single engine | Single engine |

## Resources

- **Documentation**: [docs.xorq.dev](https://docs.xorq.dev)
- **GitHub**: [github.com/xorq-labs/xorq](https://github.com/xorq-labs/xorq)
- **Examples**: `/path/to/xorq/examples/`
- **Templates**: `xorq init -t <template>`

## Contributing

This skill is maintained at [github.com/xorq-labs/xorq](https://github.com/xorq-labs/xorq) in the `skills/xorq/` directory.

Issues and PRs welcome for:
- Documentation improvements
- New workflow patterns
- Additional troubleshooting scenarios
- Resource documentation

## License

Apache-2.0 (same as xorq)
