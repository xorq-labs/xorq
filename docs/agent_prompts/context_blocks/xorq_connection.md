XORQ CONNECTION PATTERNS:
For Snowflake: con = xo.snowflake.connect_env_keypair()
For local/DuckDB: con = xo.connect()  # Great for UDFs!
ALWAYS check: con.list_tables() to see what's available

SMART BACKEND USAGE:
- Start with remote (Snowflake) for large data
- Switch to local for UDFs: table.cache() or table.into_backend(xo.connect())
- xo.connect() is ideal when you need pandas UDFs or ML operations
