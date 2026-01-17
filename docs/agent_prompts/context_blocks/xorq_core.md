# XORQ CORE - DEFERRED EXECUTION PRINCIPLES

## ⚠️ CRITICAL RULES - NEVER VIOLATE
1. **ALWAYS check schema first**: `print(table.schema())` before ANY operation
2. **NEVER use pandas for data operations** - Use ibis expressions only
3. **Build expressions first, execute later** - All operations return expressions
4. **Data from Xorq connections only** - No external data sources

## Import Patterns - Quick Reference

### Standard Setup
```python
# Core API (deferred operations, local backend, UDFs)
import xorq.api as xo
from xorq.api import _

# Snowflake backend (for remote data warehouse)
from xorq.backends.snowflake import Backend

# Caching (for backend switching)
from xorq.caching import ParquetCache

# UDFs (custom pandas logic)
from xorq.api import make_pandas_udf, make_pandas_udaf
```

### Common Connections
```python
# Local DuckDB backend (great for UDFs, ML, local data)
con = xo.connect()

# Snowflake backend (for cloud data warehouse)
from xorq.backends.snowflake import Backend
con = Backend.connect_env_keypair()
# Other Snowflake options: Backend.connect_env_mfa(), Backend.connect_env_password()
```

## Core Workflow Pattern
```python
import xorq.api as xo
from xorq.api import _
from xorq.backends.snowflake import Backend  # For Snowflake
from xorq.vendor import ibis  # For desc/asc ordering

# 1. Connect
con = Backend.connect_env_keypair()  # Snowflake
# OR: con = xo.connect()  # Local DuckDB

# 2. ALWAYS CHECK SCHEMA FIRST
table = con.table("DATA")
print(table.schema())  # ← CRITICAL! Match exact column casing

# 3. Build expression tree (deferred)
expr = (
    table
    .filter(_.VALUE > 100)  # Use exact case from schema
    .group_by(_.CATEGORY)
    .aggregate([_.AMOUNT.sum().name('total')])
    .order_by(ibis.desc('total'))  # Order by column NAME string
)

# 4. Execute only when needed
result = expr.execute()
```

## Pandas-Free Operations Reference
```python
# ❌ WRONG - Using pandas
import pandas as pd
df = table.execute()
df_filtered = df[df['PRICE'] > 100]

# ✅ RIGHT - Using ibis expressions
filtered = table.filter(_.PRICE > 100)
result = filtered.execute()  # Only at the end!

# Common replacements:
# pd.merge() → table1.join(table2, predicates)
# df.groupby() → table.group_by()
# df['new'] = ... → table.mutate(new=...)
# df.describe() → table.aggregate([_.col.mean(), _.col.std()])
# df.sort_values() → table.order_by()
# df.drop_duplicates() → table.distinct()
```

## Key Functions
- `xorq_connection_info()` - Show current connection
- `xorq_check_environment()` - Validate setup
- `print(table.schema())` - ALWAYS run before operations

## Remember
- Cache intermediate results when switching backends:
  ```python
  from xorq.caching import ParquetCache
  cached_table = table.cache(ParquetCache.from_kwargs())
  ```
- Use `_.column_name` notation for column references
- Execute only at the end of your pipeline
- For operations not in backend, use UDFs (see pandas_udf_patterns.md)