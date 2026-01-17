# Critical Rules

## Schema Checking
**ALWAYS run `print(table.schema())` before every new table operation**
- Match the exact column casing reported by the backend
- Confirm data types align with the planned operations (no implicit casts)
- Only proceed with transformations after schema validation is complete

## Column Case Rules
- **Snowflake:** UPPERCASE (_.PRICE, _.COLOR)
- **DuckDB/Postgres:** lowercase (_.price, _.color)
- **Match EXACTLY what schema shows**

## No Pandas in Main Code
**Do NOT import pandas for operations**
- Use ibis: filter(), select(), aggregate()
- Only execute() for final display
- Pandas is ONLY allowed inside UDF decorators (@make_pandas_udf)

## Vendor Ibis Import
**Always use vendored ibis:**
```python
from xorq.vendor import ibis  # ✅ CORRECT
```

**Never use:**
```python
import ibis  # ❌ WRONG - not available
import xorq.vendor.ibis  # ❌ WRONG - incorrect syntax
```

## Data Source Rules
- **Use the EXISTING connection if one is established**
- **Backend switching is ENCOURAGED when needed:**
  - Use `.cache()` to bring remote data local for UDFs
  - Use `.into_backend(xo.connect())` for explicit backend switch
  - This is the RIGHT pattern, not a workaround!
- Explore available tables: `con.list_tables()`
- Check table existence before accessing
