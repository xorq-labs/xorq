REPL EXPLORATION - USE THE INTERACTIVE SHELL TO DEBUG:

You have an interactive REPL! Use it to explore, test, and understand the API before implementing fixes.

## 🔍 STEP 1: EXPLORE WHAT EXISTS
```python
# See what methods are available
print(f"Type: {type(table)}")  # Check the object type
methods = [m for m in dir(table) if not m.startswith('_')]
print(f"Available methods ({len(methods)}):", methods[:20])

# Search for specific functionality
def find_methods(obj, pattern):
    return [m for m in dir(obj) if pattern.lower() in m.lower() and not m.startswith('_')]

print("Methods with 'agg':", find_methods(table, 'agg'))
print("Methods with 'group':", find_methods(table, 'group'))
```

## 🧪 STEP 2: TEST SMALL OPERATIONS
```python
# Test operations safely before using them
try:
    test = table.select(['col1'])
    print("✓ Selection works!")
except Exception as e:
    print(f"✗ Selection failed: {e}")

try:
    test = table.filter(_.col1 > 0)
    print("✓ Filter works!")
except Exception as e:
    print(f"✗ Filter failed: {e}")

try:
    test = table.aggregate([_.col1.mean()])
    print("✓ Aggregation works!")
except Exception as e:
    print(f"✗ Aggregation failed: {e}")
```

## 📚 STEP 3: GET HELP ON METHODS
```python
# Understand how methods work
help(table.aggregate)  # Full documentation
print(table.filter.__doc__)  # Quick docstring

# Check method signatures
import inspect
sig = inspect.signature(table.group_by)
print(f"group_by signature: {sig}")
```

## 🔬 STEP 4: DEBUG ATTRIBUTE ERRORS
```python
# When you get "AttributeError: 'Table' object has no attribute 'X'"
missing_method = 'corr'  # The method that failed

# Check if it really doesn't exist
if hasattr(table, missing_method):
    print(f"✓ {missing_method} exists!")
else:
    print(f"✗ {missing_method} not found")
    # Find similar methods
    similar = find_methods(table, missing_method[:3])
    print(f"Similar methods: {similar}")

    # Solution: Use cache + UDF for missing operations
    print("\nSolution: Cache to local and use UDF")
```

## 🚀 STEP 5: VALIDATE YOUR FIX
```python
# Before implementing, test your approach works
test_code = """
# Your proposed solution
result = table.select(['col1', 'col2']).filter(_.col1 > 0)
"""

try:
    exec(test_code)
    print("✅ Your approach works!")
except Exception as e:
    print(f"❌ Approach failed: {e}")
```

WORKFLOW:
1. EXPLORE with dir() - see what's available
2. TEST with try/except - verify it works
3. READ with help() - understand usage
4. VALIDATE - ensure your fix works
5. IMPLEMENT - only after validating!
