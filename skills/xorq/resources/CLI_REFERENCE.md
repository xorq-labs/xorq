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
- `builds/<hash>` - Path to build directory

**Options:**
- `-o, --output` - Output file path (e.g., `-o results.parquet`)
- `--format` - Output format (default: parquet)

**Example:**
```bash
xorq run builds/28ecab08754e -o output.parquet
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

**Options:**
- `--agent` - Include agent guides (AGENTS.md, CLAUDE.md)

**Example:**
```bash
xorq init -t penguins
xorq init -t sklearn --agent
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

### `xorq agent onboard`

Show guided onboarding workflow.

```bash
xorq agent onboard
```

**Options:**
- `--step <init|build|catalog|test|land>` - Show specific step

**Example:**
```bash
xorq agent onboard
xorq agent onboard --step build
```

---

### `xorq agent prompt`

Manage agent prompts (context blocks).

#### `xorq agent prompt list`

List all available prompts.

```bash
xorq agent prompt list [options]
```

**Options:**
- `--tier <core|reliability|advanced>` - Filter by tier

**Example:**
```bash
xorq agent prompt list
xorq agent prompt list --tier core
```

#### `xorq agent prompt show`

Show a specific prompt.

```bash
xorq agent prompt show <name>
```

**Example:**
```bash
xorq agent prompt show xorq_core
xorq agent prompt show must_check_schema
```

---

### `xorq agent templates`

Manage agent skills (templates).

#### `xorq agent templates list`

List all registered templates.

```bash
xorq agent templates list
```

**Output:**
```
NAME              TEMPLATE        DESCRIPTION
penguins_demo     penguins        Minimal multi-engine example
sklearn_pipeline  sklearn         Deferred sklearn pipeline
cached_fetcher    cached-fetcher  Hydrate and cache upstream tables
```

#### `xorq agent templates show`

Show details for a specific skill.

```bash
xorq agent templates show <skill-name>
```

**Example:**
```bash
xorq agent templates show sklearn_pipeline
```

#### `xorq agent templates scaffold`

Generate a starter file from a skill.

```bash
xorq agent templates scaffold <skill-name>
```

**Example:**
```bash
xorq agent templates scaffold penguins_demo
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

Run an unbound expression by reading Arrow IPC from stdin.

```bash
cat data.arrow | xorq run-unbound builds/<hash>
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
xorq agent onboard
xorq agent prompt list --tier core
xorq agent templates scaffold sklearn_pipeline
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
