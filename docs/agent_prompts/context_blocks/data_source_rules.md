DATA SOURCE RULES:
- Use the EXISTING connection if one is established
- Backend switching is ENCOURAGED when needed:
  - Use .cache() to bring remote data local for UDFs
  - Use .into_backend(xo.connect()) for explicit backend switch
  - This is the RIGHT pattern, not a workaround!
- Explore available tables: con.list_tables()
- Check table existence before accessing
