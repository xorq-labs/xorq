# XORQ CORE - DEFERRED EXECUTION PRINCIPLES

## ⚠️ CRITICAL RULES - NEVER VIOLATE
1. **ALWAYS check schema first**: `print(table.schema())` before ANY operation
2. **NEVER use pandas for data operations** - Use ibis expressions only
3. **Build expressions first, execute later** - All operations return expressions
4. **Data from Xorq connections only** - No external data sources

## Core Workflow Pattern
```python
import xorq.api as xo
from xorq.api import _
from xorq.vendor import ibis  # ✅ CORRECT: For desc/asc ordering

# 1. Connect (if needed)
con = xo.connect()  # Local backend
# OR: con = xo.snowflake.connect_env_keypair()  # Remote

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
    # NOT: xo.vendor.ibis.desc('total')  # ❌ Wrong! No .vendor attribute access
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
- Cache intermediate results when switching backends: `table.cache(ParquetCache.from_kwargs())`
- Use `_.column_name` notation for column references
- Execute only at the end of your pipeline
- For operations not in backend, use UDFs (see pandas_udf_patterns.md)