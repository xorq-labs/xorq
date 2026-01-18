# Xorq CLI Reference

Complete command reference for xorq CLI.

## Core Commands

### `xorq build`

Build an expression to create a manifest artifact.

```bash
xorq build expr.py -e expr [options]
```

**Arguments:**
- `expr.py` - Python file containing the expression
- `-e, --expr` - Name of the expression variable to build

**Options:**
- `--pdb` - Drop into pdb on failure
- `--pdb-runcall` - Invoke with pdb.runcall

**Output:**
- `builds/<hash>/` directory containing manifest, metadata, and caches

**Example:**
```bash
xorq build my_pipeline.py -e pipeline
# Output: builds/28ecab08754e/
```

---

### `xorq run`

Execute a built expression.

```bash
xorq run builds/<hash> [options]
```

**Arguments:**
- `builds/<hash>` - Path to build directory or catalog alias

**Options:**
- `-o, --output-path` - Output file path (default: `/dev/null`)
- `-f, --format` - Output format: `csv`, `json`, `parquet`, `arrow` (default: parquet)
- `--limit` - Limit number of rows to output

**Examples:**
```bash
# Write to file
xorq run builds/28ecab08754e -o output.parquet

# Write Arrow IPC format
xorq run my-source -f arrow -o output.arrow

# Stream to stdout for piping
xorq run my-source -f arrow -o /dev/stdout

# Use with catalog alias
xorq run batting-source -o results.parquet
```

**Streaming to DuckDB:**
```bash
# Pipe Arrow IPC to DuckDB for SQL analysis
xorq run my-source -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow; SELECT * FROM read_arrow('/dev/stdin') LIMIT 10"
```

---

### `xorq catalog`

Manage the build catalog.

#### `xorq catalog add`

Add a build to the catalog.

```bash
xorq catalog add builds/<hash> --alias <name>
```

**Example:**
```bash
xorq catalog add builds/28ecab08754e --alias my-pipeline
```

#### `xorq catalog ls`

List all catalog entries.

```bash
xorq catalog ls
```

**Output:**
```
Aliases:
my-pipeline    a498016e-5bea-4036-aec0-a6393d1b7c0f    r1

Entries:
a498016e-5bea-4036-aec0-a6393d1b7c0f    r1      28ecab08754e
```

#### `xorq catalog info`

Show information about a catalog entry.

```bash
xorq catalog info <alias>
```

**Options:**
- `--build-path` - Show build directory path

**Example:**
```bash
xorq catalog info my-pipeline --build-path
# Output: builds/28ecab08754e
```

#### `xorq catalog rm`

Remove a catalog entry or alias.

```bash
xorq catalog rm <alias>
```

#### `xorq catalog diff-builds`

Compare two build artifacts.

```bash
xorq catalog diff-builds builds/<hash1> builds/<hash2>
```

#### `xorq catalog sources`

List source nodes in an expression (for composition via `run-unbound`).

```bash
xorq catalog sources <alias>
```

**Output:**
Shows all source nodes that can be "unbound" and replaced with piped data.

**Example:**
```bash
xorq catalog sources lineup-transform

# Output:
# Alias: lineup-transform
# Sources (1 node(s)):
#
# Source 1:
#   Hash: d43ad87ea8a989f3495aab5dff0b5746
#   Type: xorq.expr.relations.Read
#   Name: xorq_cached_node_name_placeholder
#   Columns: 22
#
# To unbind this source:
#   xorq run <source-alias> -f arrow -o - | \
#     xorq run-unbound lineup-transform \
#       --to_unbind_hash d43ad87ea8a989f3495aab5dff0b5746 \
#       --typ xorq.expr.relations.Read \
#       -f parquet -o output.parquet
```

**Use case:** Finding nodes to unbind for pipeline composition.

#### `xorq catalog schema`

Show output schema of a cataloged expression.

```bash
xorq catalog schema <alias>
```

**Example:**
```bash
xorq catalog schema batting-source

# Output: Arrow schema with column names and types
```

**Use case:** Verifying schema compatibility before composition.

---

### `xorq lineage`

Show column-level lineage for a build.

```bash
xorq lineage <alias-or-hash>
```

**Example:**
```bash
xorq lineage my-pipeline

# Output:
Lineage for column 'avg_value':
Field:avg_value #1
└── Cache xorq_cached_node_name_placeholder #2
    └── Aggregate #3
        └── Mean #4
            └── Field:value #5
```

---

### `xorq init`

Initialize a new xorq project from a template.

```bash
xorq init -t <template>
```

**Templates:**
- `penguins` - Minimal multi-engine example
- `sklearn` - ML pipeline with train/predict

**Example:**
```bash
xorq init -t penguins
xorq agents init  # Setup agent guides after initialization
```

---

### `xorq serve-unbound`

Serve an expression via Arrow Flight.

```bash
xorq serve-unbound builds/<hash> \
  --to_unbind_hash <node-hash> \
  --host <host> \
  --port <port>
```

**Example:**
```bash
xorq serve-unbound builds/28ecab08754e \
  --to_unbind_hash 31f0a5be3771 \
  --host localhost \
  --port 9002
```

---

## Agent Commands

### `xorq agents onboard`

Show guided onboarding workflow.

```bash
xorq agents onboard
```

**Options:**
- `--step <init|build|catalog|test|land>` - Show specific step

**Example:**
```bash
xorq agents onboard
xorq agents onboard --step build
```

---

### `xorq agents prompt`

Manage agent prompts (context blocks).

#### `xorq agents prompt list`

List all available prompts.

```bash
xorq agents prompt list [options]
```

**Options:**
- `--tier <core|reliability|advanced>` - Filter by tier

**Example:**
```bash
xorq agents prompt list
xorq agents prompt list --tier core
```

#### `xorq agents prompt show`

Show a specific prompt.

```bash
xorq agents prompt show <name>
```

**Example:**
```bash
xorq agents prompt show xorq_core
xorq agents prompt show must_check_schema
```

---

### `xorq agents templates`

Manage agent skills (templates).

#### `xorq agents templates list`

List all registered templates.

```bash
xorq agents templates list
```

**Output:**
```
NAME              TEMPLATE        DESCRIPTION
penguins_demo     penguins        Minimal multi-engine example
sklearn_pipeline  sklearn         Deferred sklearn pipeline
cached_fetcher    cached-fetcher  Hydrate and cache upstream tables
```

#### `xorq agents templates show`

Show details for a specific skill.

```bash
xorq agents templates show <skill-name>
```

**Example:**
```bash
xorq agents templates show sklearn_pipeline
```

#### `xorq agents templates scaffold`

Generate a starter file from a skill.

```bash
xorq agents templates scaffold <skill-name>
```

**Example:**
```bash
xorq agents templates scaffold penguins_demo
# Creates: skills/penguins_demo.py
```

---

## Advanced Commands

### `xorq uv-build`

Build with a custom Python environment using uv.

```bash
xorq uv-build expr.py -e expr
```

Creates build with sdist for reproducibility.

---

### `xorq uv-run`

Run with a custom Python environment using uv.

```bash
xorq uv-run builds/<hash> -o output.parquet
```

---

### `xorq run-unbound`

Run an unbound expression by reading Arrow IPC from stdin. This enables composing arbitrary pipelines by "unbinding" source nodes and replacing them with piped data.

```bash
cat data.arrow | xorq run-unbound builds/<hash> \
  --to_unbind_hash <node-hash> \
  --typ xorq.expr.relations.Read
```

**Arguments:**
- `builds/<hash>` - Path to build directory or catalog alias

**Required Options:**
- `--to_unbind_hash` - Hash of the node to unbind (from `xorq catalog sources`)
- `--typ` - Type of node to unbind (usually `xorq.expr.relations.Read`)

**Output Options:**
- `-o, --output-path` - Output file path (default: `/dev/null`)
- `-f, --format` - Output format: `csv`, `json`, `parquet`, `arrow` (default: parquet)

**Finding Unbound Hashes:**
```bash
# List source nodes that can be unbound
xorq catalog sources lineup-transform

# Output shows:
# Source 1:
#   Hash: d43ad87ea8a989f3495aab5dff0b5746
#   Type: xorq.expr.relations.Read
#   Columns: 22
```

**Examples:**

**Basic composition:**
```bash
# Replace source node with piped data
xorq run source1 -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound transform \
    --to_unbind_hash d43ad87ea8a989f3495aab5dff0b5746 \
    --typ xorq.expr.relations.Read \
    -f parquet -o output.parquet
```

**Multi-stage pipeline:**
```bash
# Chain multiple transforms via Arrow IPC
xorq run source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound transform1 \
    --to_unbind_hash <hash1> \
    --typ xorq.expr.relations.Read \
    -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound transform2 \
    --to_unbind_hash <hash2> \
    --typ xorq.expr.relations.Read \
    -o final_output.parquet
```

**Stream to DuckDB:**
```bash
# Compose pipeline and analyze with SQL
xorq run source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound transform \
    --to_unbind_hash abc123 \
    --typ xorq.expr.relations.Read \
    -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow;
    SELECT col1, COUNT(*) as cnt
    FROM read_arrow('/dev/stdin')
    GROUP BY col1"
```

---

### `xorq serve-flight-udxf`

Serve a build via Flight Server with UDF support.

```bash
xorq serve-flight-udxf builds/<hash> --port 9002
```

---

### `xorq ps`

List running xorq servers.

```bash
xorq ps
```

---

## Global Options

All commands support:
- `--pdb` - Drop into Python debugger on failure
- `--pdb-runcall` - Invoke with pdb.runcall
- `--help` - Show command-specific help

---

## Common Workflows

### Build → Catalog → Run
```bash
xorq build expr.py -e expr
xorq catalog add builds/<hash> --alias my-expr
xorq run builds/<hash> -o output.parquet
xorq lineage my-expr
```

### Agent-Guided Development
```bash
xorq agents onboard
xorq agents prompt list --tier core
xorq agents templates scaffold sklearn_pipeline
xorq build sklearn_pipeline.py -e pipeline
```

### Serve Expression
```bash
xorq build expr.py -e expr
xorq serve-unbound builds/<hash> \
  --to_unbind_hash <node> \
  --port 9002
```

---

For more details on any command, run:
```bash
xorq <command> --help
```
