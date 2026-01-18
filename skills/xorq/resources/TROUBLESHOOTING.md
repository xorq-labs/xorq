# Xorq Troubleshooting

Common issues and solutions when using xorq.

## Build Errors

### Schema Mismatch

**Error:**
```
SchemaError: Column 'foo' not found in table
```

**Cause:** Expression references column that doesn't exist in source.

**Solution:**

1. **Always check schema first:**
```python
import xorq.api as xo
con = xo.connect()
table = con.table("data")
print(table.schema())  # Do this BEFORE building expression
```

2. **Use agent prompt:**
```bash
xorq agents prompt show fix_schema_errors
```

3. **Fix column references:**
```python
# Wrong: assumes column exists
expr = table.select("nonexistent_column")

# Right: check schema first
print(table.schema())
expr = table.select("actual_column")
```

---

### Import Errors

**Error:**
```
ImportError: No module named 'ibis'
ModuleNotFoundError: No module named 'ibis.expr.types'
```

**Cause:** Using wrong ibis import.

**Solution:**

1. **Use vendored ibis:**
```python
# Right:
from xorq.vendor import ibis

# Wrong:
import ibis
```

2. **Check agent prompt:**
```bash
xorq agents prompt show fix_import_errors
xorq agents prompt show xorq_vendor_ibis
```

3. **Verify imports in file:**
```python
# Correct import order:
import xorq.api as xo
from xorq.vendor import ibis
from xorq.common.utils.ibis_utils import from_ibis
from xorq.caching import ParquetCache
```

---

### Attribute Errors

**Error:**
```
AttributeError: 'Table' object has no attribute 'foo'
```

**Cause:** Calling wrong method or accessing non-existent attribute.

**Solution:**

1. **Check agent prompt:**
```bash
xorq agents prompt show fix_attribute_errors
```

2. **Verify API:**
```python
# Check available methods
dir(table)

# Use correct Ibis API
table.filter(ibis._.col > 0)  # Right
table.where(...)  # Check if method exists
```

---

### Build Variable Not Found

**Error:**
```
NameError: name 'expr' is not defined
```

**Cause:** Variable name doesn't match `-e` argument.

**Solution:**

```bash
# File contains: my_pipeline = ...
xorq build file.py -e my_pipeline  # ✓ Match variable name

# File contains: expr = ...
xorq build file.py -e expr  # ✓ Match variable name

# Don't mismatch:
xorq build file.py -e wrong_name  # ✗ NameError
```

---

## Execution Errors

### Backend Not Found

**Error:**
```
BackendError: Could not connect to backend
```

**Cause:** Backend not available or not configured.

**Solution:**

1. **Check connection:**
```python
import xorq.api as xo

# Test connection
con = xo.connect()  # DuckDB (default)
print(con)

# For other backends:
con = xo.sqlite.connect()
con = xo.snowflake.connect(...)
```

2. **Verify backend available:**
```bash
python -c "import duckdb; print(duckdb.__version__)"
python -c "import sqlite3; print(sqlite3.version)"
```

3. **Check agent prompt:**
```bash
xorq agents prompt show xorq_connection
```

---

### Data Type Mismatch

**Error:**
```
DataTypeError: Cannot cast string to int64
```

**Cause:** Column types don't match expected types.

**Solution:**

1. **Check schema:**
```python
print(table.schema())
```

2. **Cast explicitly:**
```python
# Wrong: assumes column is int
expr = table.filter(ibis._.col > 10)

# Right: cast if needed
expr = table.filter(ibis._.col.cast("int64") > 10)
```

3. **Check agent prompt:**
```bash
xorq agents prompt show fix_data_errors
```

---

### UDF Errors

**Error:**
```
UDFError: Function execution failed
```

**Cause:** UDF not compatible with backend.

**Solution:**

1. **Check backend compatibility:**
```bash
xorq agents prompt show fix_udf_backend_errors
```

2. **Use appropriate UDF type:**
```python
# For pandas UDF:
from xorq.expr.udxf import pandas_udf

@pandas_udf
def my_udf(x):
    return x * 2
```

---

## Caching Issues

### Cache Not Used

**Problem:** Expression rebuilds instead of using cache.

**Cause:** Expression changed, creating new hash.

**Solution:**

1. **Understand input-addressing:**
```python
# Same expression = same hash = cache hit
expr1 = table.filter(ibis._.col > 0)
expr2 = table.filter(ibis._.col > 0)
# These have same hash, share cache

# Different expression = different hash = cache miss
expr3 = table.filter(ibis._.col > 1)
# Different predicate = different hash
```

2. **Check cache paths:**
```python
expr.ls.get_cache_paths()
# Shows where cache is stored
```

3. **Verify cache directory:**
```bash
ls ~/.cache/xorq/parquet/
```

---

### Cache Permission Errors

**Error:**
```
PermissionError: Cannot write to cache directory
```

**Solution:**

1. **Check permissions:**
```bash
ls -la ~/.cache/xorq/
```

2. **Fix permissions:**
```bash
chmod -R u+w ~/.cache/xorq/
```

3. **Or specify custom cache:**
```python
from xorq.caching import ParquetCache

cache = ParquetCache.from_kwargs(path="/custom/cache/path")
expr = from_ibis(query).cache(cache)
```

---

## Catalog Issues

### Alias Already Exists

**Error:**
```
CatalogError: Alias 'my-expr' already exists
```

**Solution:**

1. **List existing aliases:**
```bash
xorq catalog ls
```

2. **Remove old alias:**
```bash
xorq catalog rm my-expr
```

3. **Or use different alias:**
```bash
xorq catalog add builds/<hash> --alias my-expr-v2
```

---

### Catalog Entry Not Found

**Error:**
```
CatalogError: Entry 'my-expr' not found
```

**Solution:**

1. **List catalog:**
```bash
xorq catalog ls
```

2. **Check spelling:**
```bash
# Case-sensitive!
xorq catalog info my-expr  # ✓
xorq catalog info My-Expr  # ✗
```

3. **Add if missing:**
```bash
xorq catalog add builds/<hash> --alias my-expr
```

---

## Lineage Issues

### Lineage Too Large

**Problem:** Lineage output is huge and unreadable.

**Solution:**

1. **Focus on specific columns:**
```bash
# xorq lineage shows all columns
# Read output carefully for specific column you care about
```

2. **Inspect manifest directly:**
```bash
cat builds/<hash>/expr.yaml
# Find specific node you're interested in
```

3. **Use built-in tools:**
```python
# In Python REPL
expr.ls.nodes  # List all nodes
expr.ls.get_cache_paths()  # Check cache locations
```

---

## Agent Issues

### Prompt Not Found

**Error:**
```
PromptError: Prompt 'foo' not found
```

**Solution:**

1. **List available prompts:**
```bash
xorq agents prompt list
```

2. **Check spelling:**
```bash
# Use exact name from list
xorq agents prompt show xorq_core  # ✓
xorq agents prompt show XorqCore  # ✗
```

---

### Skill Not Found

**Error:**
```
SkillError: Skill 'foo' not found
```

**Solution:**

1. **List available skills:**
```bash
xorq agents templates list
```

2. **Initialize with template:**
```bash
# If skill corresponds to template
xorq init -t penguins
xorq agents templates scaffold penguins_demo
```

---

## Performance Issues

### Build Too Slow

**Problem:** `xorq build` takes too long.

**Solution:**

1. **Add caching:**
```python
# Cache expensive operations
expr = (
    from_ibis(expensive_query)
    .cache(ParquetCache.from_kwargs())
)
```

2. **Optimize query:**
```python
# Push filters down
expr = (
    table
    .filter(ibis._.date > "2024-01-01")  # Filter early
    .select(["col1", "col2"])  # Select only needed columns
    .group_by("col1")
    .agg(metric=ibis._.col2.sum())
)
```

3. **Check agent prompts:**
```bash
xorq agents prompt show optimization_patterns
```

---

### Execution Too Slow

**Problem:** `xorq run` takes too long.

**Solution:**

1. **Use appropriate backend:**
```python
# DuckDB for analytical queries
# SQLite for small data
# Snowflake for large data

# Move to faster backend:
expr = from_ibis(query).into_backend(duckdb_con)
```

2. **Add caching at right places:**
```python
# Cache after expensive aggregations
expr = (
    from_ibis(
        table
        .group_by("key")
        .agg(metric=ibis._.value.sum())
    )
    .cache(ParquetCache.from_kwargs())
)
```

---

## General Debugging

### Enable Debug Mode

```bash
# Run with pdb on error
xorq build expr.py -e expr --pdb

# Or with pdb.runcall
xorq build expr.py -e expr --pdb-runcall
```

### Inspect Build Artifacts

```bash
# After build, inspect contents
BUILD_DIR=builds/<hash>
ls $BUILD_DIR

# Check manifest
cat $BUILD_DIR/expr.yaml

# Check metadata
cat $BUILD_DIR/metadata.json

# Check profiles
cat $BUILD_DIR/profiles.yaml

# Check cached data
ls $BUILD_DIR/database_tables/
```

### Use Agent Prompts

```bash
# List all fix prompts
xorq agents prompt list --tier reliability

# Show specific fix
xorq agents prompt show fix_schema_errors
xorq agents prompt show fix_attribute_errors
xorq agents prompt show fix_data_errors
xorq agents prompt show fix_import_errors
xorq agents prompt show fix_udf_backend_errors
```

### Ask for Help

1. **Check documentation:**
   - https://docs.xorq.dev
   - `xorq --help`
   - `xorq <command> --help`

2. **Check examples:**
   ```bash
   ls examples/
   ```

3. **Open issue:**
   - https://github.com/xorq-labs/xorq/issues

---

## Quick Reference

| Issue | Command |
|-------|---------|
| Schema error | `xorq agents prompt show fix_schema_errors` |
| Import error | `xorq agents prompt show fix_import_errors` |
| Attribute error | `xorq agents prompt show fix_attribute_errors` |
| Data error | `xorq agents prompt show fix_data_errors` |
| UDF error | `xorq agents prompt show fix_udf_backend_errors` |
| Connection issue | `xorq agents prompt show xorq_connection` |
| All reliability | `xorq agents prompt list --tier reliability` |

---

For more help, see:
- [WORKFLOWS.md](WORKFLOWS.md) - Step-by-step patterns
- [PATTERNS.md](PATTERNS.md) - Best practices
- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Command reference
