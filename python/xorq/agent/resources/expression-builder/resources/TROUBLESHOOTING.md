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

3. **Verify imports in file:**
```python
# Correct import order:
import xorq.api as xo
from xorq.vendor import ibis
from xorq.common.utils.ibis_utils import from_ibis
from xorq.caching import ParquetCache
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

### xo.cases() with xo._ (Deferred) Inside .mutate()

**Error:**
```
SignatureValidationError: Literal((_.cut == 'Fair'), dtype=Boolean(nullable=True)) has failed...
`value`: (_.cut == 'Fair') is not matching Any() then anything except a Deferred

Expected signature: Literal(value: Annotated[Any, Not(pattern=InstanceOf(type=<class 'Deferred'>))])
```

**Cause:** `xo.cases()` function cannot handle `xo._` (Deferred) references inside `.mutate()` blocks.

**Solution:** Use the `.cases()` method on the column instead of `xo.cases()` function.

```python
import xorq.api as xo

# ❌ DOESN'T WORK - xo.cases() rejects xo._ (Deferred)
expr = table.mutate(
    ordinal=xo.cases(
        (xo._.cut == "Fair", 1),
        (xo._.cut == "Good", 2),
        else_=0
    )
)

# ✅ WORKS - Use .cases() method on the column
expr = table.mutate(
    ordinal=xo._.cut.cases(
        ("Fair", 1),
        ("Good", 2),
        ("Very Good", 3),
        else_=0
    )
)

# ✅ ALSO WORKS - Use expression reference instead of xo._
def add_ordinal(table):
    cut_col = table.cut  # Bind column outside
    return table.mutate(
        ordinal=xo.cases(
            (cut_col == "Fair", 1),
            (cut_col == "Good", 2),
            else_=0
        )
    )

# ✅ ALSO WORKS - Use xo.ifelse() for simple cases
expr = table.mutate(
    ordinal=xo.ifelse(
        xo._.cut == "Fair", 1,
        xo.ifelse(xo._.cut == "Good", 2, 0)
    )
)
```

**Best Practice:** Prefer `.cases()` method - it's cleaner and works with `xo._`:
```python
# Clean pattern for categorical encoding
diamonds.mutate(
    cut_ordinal=xo._.cut.cases(
        ("Fair", 1.0),
        ("Good", 2.0),
        ("Very Good", 3.0),
        ("Premium", 4.0),
        ("Ideal", 5.0),
        else_=0.0
    ),
    color_ordinal=xo._.color.cases(
        ("D", 7.0), ("E", 6.0), ("F", 5.0),
        ("G", 4.0), ("H", 3.0), ("I", 2.0), ("J", 1.0),
        else_=0.0
    )
)
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
con = xo.connect()
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
3. **Try `into_backend`

---

---

### Type Mismatch in sklearn UDF

**Error:**
```
ValueError: Failed to coerce arguments to satisfy a call to
'dumps_of_inner_fit_0' function: coercion from (..., Int64)
to signature Exact([..., Int8]) failed
```

**Cause:** Deferred sklearn fit functions expect specific int types (int8 vs int64).

**Solution:** Cast target column explicitly.

```python
is_breakout = xo.ifelse(_.ops_vs_career >= 1.20, 1, 0).into_backend(xo.connect())

train_data = training_data.filter(...).mutate(
    is_breakout=_.is_breakout.cast("int8")
).into_backend(xo.connect())

# Fit with properly typed target
fitted = xorq_pipeline.fit(
    train_data,
    features=feature_cols,
    target="is_breakout"  # Now int8 type
)
```

**Status:** Partial workaround - some edge cases may still occur with complex UDFs.

---

For more help, see:
- [WORKFLOWS.md](WORKFLOWS.md) - Step-by-step patterns and best practices
- [CLI_REFERENCE.md](CLI_REFERENCE.md) - Command reference
