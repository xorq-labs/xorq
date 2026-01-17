XORQ CONNECTION PATTERNS:

## Correct Import Patterns
```python
# For Snowflake connections
from xorq.backends.snowflake import Backend
con = Backend.connect_env_keypair()  # Using keypair auth
# OR: Backend.connect_env_mfa()  # MFA auth
# OR: Backend.connect_env_password()  # Password auth

# For local/DuckDB connections
import xorq.api as xo
con = xo.connect()  # Great for UDFs and ML!
```

ALWAYS check what's available: `con.list_tables()`

## Common Import Errors

### ❌ WRONG:
```python
import xorq.api as xo
con = xo.snowflake.connect_env_keypair()  # AttributeError!
```

### ✅ CORRECT:
```python
from xorq.backends.snowflake import Backend
con = Backend.connect_env_keypair()
```

## SMART BACKEND USAGE:
- Start with remote (Snowflake) for large data
- Switch to local for UDFs:
  ```python
  from xorq.caching import ParquetCache
  local_table = snowflake_table.cache(ParquetCache.from_kwargs())
  # OR: local_table = snowflake_table.into_backend(xo.connect())
  ```
- xo.connect() is ideal when you need pandas UDFs or ML operations
