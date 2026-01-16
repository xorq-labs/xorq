## PYTHON PACKAGE EXPLORATION GUIDE

When working with unfamiliar packages like `xorq`, use these exploration techniques to discover APIs and understand functionality.

### 1. Basic REPL Exploration Commands

```python
import xorq.api as xo

# List all available attributes and methods
dir(xo)

# Get help on the module
help(xo)

# Check module documentation
print(xo.__doc__)

# See module file location
print(xo.__file__)

# List only public methods (without underscores)
[attr for attr in dir(xo) if not attr.startswith('_')]
```

### 2. Exploring Specific Objects

```python
# For any object, check what you can do with it
obj = xo.snowflake.connect_env_keypair()

# List all methods
dir(obj)

# Get object type
type(obj)

# Check if it has specific methods
hasattr(obj, 'table')  # Returns True/False

# Get method signature
help(obj.table)

# See all tables available
tables = obj.list_tables()
print(tables)
```

### 3. Using the inspect Module

```python
import inspect

# Get function signature
sig = inspect.signature(xo.snowflake.connect_env_keypair)
print(sig)

# Get function source code
source = inspect.getsource(xo.snowflake.connect_env_keypair)
print(source)

# Get all classes in module
classes = inspect.getmembers(xo, inspect.isclass)
for name, cls in classes:
    print(f"Class: {name}")

# Get all functions in module
functions = inspect.getmembers(xo, inspect.isfunction)
for name, func in functions:
    print(f"Function: {name}{inspect.signature(func)}")

# Get method parameters
params = inspect.signature(obj.table).parameters
for name, param in params.items():
    print(f"  {name}: {param.annotation if param.annotation != param.empty else 'Any'}")
```

### 4. Exploring Xorq/Ibis Tables Specifically

```python
# After getting a table object
t = con.table("TABLENAME")

# Explore table structure
print(t.schema())  # ALWAYS do this first!

# List available operations
table_methods = [m for m in dir(t) if not m.startswith('_')]
print("Available table methods:", table_methods[:20])

# Common exploration patterns
print(t.columns)        # List column names
print(t.count().execute())  # Row count
print(t.limit(5).execute())  # Sample data

# Check column types
for col in t.columns:
    col_obj = getattr(t, col)
    print(f"{col}: {type(col_obj)}")

# Explore what operations work on columns
from xorq.api import _
col_methods = [m for m in dir(_.COLUMN_NAME) if not m.startswith('_')]
print("Column operations:", col_methods[:10])
```

### 5. Interactive Discovery Workflow

```python
# Step 1: Import and connect
import xorq.api as xo
con = xo.snowflake.connect_env_keypair()

# Step 2: Explore connection object
print("Connection methods:", [m for m in dir(con) if not m.startswith('_')][:10])

# Step 3: List available tables
tables = con.list_tables()
print(f"Found {len(tables)} tables")
if tables:
    print("First few tables:", tables[:5])

# Step 4: Pick a table and explore
if tables:
    t = con.table(tables[0])
    print(f"\nExploring table: {tables[0]}")
    print(f"Schema: {t.schema()}")
    print(f"Columns: {t.columns}")

    # Step 5: Try operations
    print(f"Row count: {t.count().execute()}")
    print(f"Sample row: {t.limit(1).execute()}")
```

### 6. Discovering Method Chaining Patterns

```python
# See what methods return the same type (chainable)
t = con.table("TABLENAME")
filtered = t.filter(_.PRICE > 100)

# Check if chainable
print(f"Original type: {type(t)}")
print(f"After filter: {type(filtered)}")
print(f"Chainable: {type(t) == type(filtered)}")

# Explore chain possibilities
chain_methods = []
for method_name in dir(t):
    if not method_name.startswith('_'):
        method = getattr(t, method_name)
        if callable(method):
            try:
                # Check method docstring for return type hints
                if method.__doc__ and 'Table' in method.__doc__:
                    chain_methods.append(method_name)
            except:
                pass

print("Potentially chainable methods:", chain_methods[:10])
```

### 7. Finding Examples in Docstrings

```python
# Many functions include examples in their docstrings
import xorq.api as xo

# Search for examples in module
for name in dir(xo):
    if not name.startswith('_'):
        obj = getattr(xo, name)
        if hasattr(obj, '__doc__') and obj.__doc__:
            if 'example' in obj.__doc__.lower() or '>>>' in obj.__doc__:
                print(f"\n{name} has examples:")
                print(obj.__doc__[:500])
```

### 8. Type Introspection for Better Understanding

```python
# Understanding return types
from typing import get_type_hints
import inspect

# For functions with type hints
try:
    hints = get_type_hints(xo.snowflake.connect_env_keypair)
    print("Type hints:", hints)
except:
    print("No type hints available")

# Check inheritance chain
t = con.table("TABLENAME")
print("Inheritance chain:")
for cls in type(t).__mro__:
    print(f"  - {cls.__name__}")
```

### 9. Systematic Package Exploration Template

```python
def explore_package(package_name):
    """Systematic exploration of a Python package"""
    try:
        # Import the package
        pkg = __import__(package_name)
        print(f"=== Exploring {package_name} ===\n")

        # Basic info
        print(f"Version: {getattr(pkg, '__version__', 'Unknown')}")
        print(f"File: {getattr(pkg, '__file__', 'Unknown')}")
        print(f"Package contents: {dir(pkg)[:10]}...")

        # Find submodules
        submodules = [name for name in dir(pkg) if not name.startswith('_')]
        print(f"\nSubmodules/attributes: {submodules[:5]}...")

        # Find classes
        import inspect
        classes = inspect.getmembers(pkg, inspect.isclass)
        print(f"\nClasses found: {[c[0] for c in classes[:5]]}...")

        # Find functions
        functions = inspect.getmembers(pkg, inspect.isfunction)
        print(f"\nFunctions found: {[f[0] for f in functions[:5]]}...")

        return pkg
    except ImportError as e:
        print(f"Could not import {package_name}: {e}")
        return None

# Example usage
explore_package('xorq.api')
```

### 10. PRACTICAL XORQ EXPLORATION EXAMPLE

```python
# When you need to understand how to use xorq for a specific task:

import xorq.api as xo
from xorq.api import _
import inspect

# 1. First, understand the connection
print("=== Exploring Xorq Connection ===")
print("Available connection methods:")
print([m for m in dir(xo) if 'connect' in m.lower()])

# 2. Connect and explore connection object
con = xo.snowflake.connect_env_keypair()
print("\n=== Connection Object Methods ===")
con_methods = [m for m in dir(con) if not m.startswith('_')]
print(con_methods[:10])

# 3. Get a table and understand its API
tables = con.list_tables()
if tables:
    t = con.table(tables[0])
    print(f"\n=== Table API for {tables[0]} ===")

    # Check schema first (CRITICAL!)
    print("Schema:", t.schema())

    # Explore filtering
    print("\nFilter method signature:")
    print(inspect.signature(t.filter))

    # Explore aggregation
    print("\nAggregate method signature:")
    print(inspect.signature(t.aggregate))

    # Try a simple operation
    result = t.limit(1).execute()
    print(f"\nSample result type: {type(result)}")
    print(f"Result: {result}")

# 4. Understand the _ (underscore) object for column references
print("\n=== Understanding _ (underscore) ===")
print(f"Type of _: {type(_)}")
print(f"Methods on _: {[m for m in dir(_) if not m.startswith('__')][:5]}")

# Try accessing a column through _
if tables and t.columns:
    col_name = t.columns[0]
    print(f"\nAccessing column '{col_name}' through _:")
    col_ref = getattr(_, col_name)
    print(f"Column reference type: {type(col_ref)}")
    print(f"Column methods: {[m for m in dir(col_ref) if not m.startswith('_')][:10]}")
```

## KEY EXPLORATION PRINCIPLES

1. **Always start with `dir()` and `help()`** - These are your primary discovery tools
2. **Check schemas first** - For data objects, always examine structure before operations
3. **Use inspect for deep introspection** - Get signatures, source code, and documentation
4. **Test small operations** - Try simple operations before complex ones
5. **Read error messages** - They often hint at correct usage
6. **Check return types** - Understanding what methods return helps with chaining
7. **Look for patterns** - Similar methods often have similar signatures
8. **Use docstrings** - Many packages include examples in docstrings
9. **Explore incrementally** - Start with the module, then classes, then methods
10. **Keep notes** - Document what works for future reference

## COMMON EXPLORATION PATTERNS

```python
# Pattern 1: "What can I do with this object?"
obj = get_some_object()
print(type(obj))
print(dir(obj))
help(obj)

# Pattern 2: "How do I call this function?"
import inspect
print(inspect.signature(function_name))
help(function_name)

# Pattern 3: "What modules are available?"
import pkg
print([m for m in dir(pkg) if not m.startswith('_')])

# Pattern 4: "What's the structure of this data?"
data = get_some_data()
print(type(data))
if hasattr(data, 'schema'):
    print(data.schema())
if hasattr(data, 'columns'):
    print(data.columns)
if hasattr(data, 'shape'):
    print(data.shape)

# Pattern 5: "Find all methods that contain a keyword"
keyword = 'filter'
methods = [m for m in dir(obj) if keyword in m.lower()]
print(f"Methods containing '{keyword}': {methods}")
```

Remember: The REPL is your laboratory. Experiment freely, test hypotheses, and learn by doing!