#!/usr/bin/env python3
"""PostToolUseFailure hook for Cortex Code - provides xorq troubleshooting guidance."""

import json
import os
import sys


def get_troubleshooting_guidance(error_output):
    """Generate troubleshooting guidance based on error output."""
    if not error_output:
        return None

    error_lower = error_output.lower()

    guidance = None

    # Type coercion errors
    if "cannot coerce" in error_lower or "type coercion" in error_lower:
        guidance = """
🔧 TYPE COERCION ERROR DETECTED

This error typically occurs when mixing backends or working with incompatible types.

**Quick fix:**
Use `.into_backend()` to explicitly convert expressions:

```python
# Convert to target backend
expr = expr.into_backend(backend)
```

**Common scenarios:**
- Mixing letsql/polars/datafusion expressions
- Using catalog expressions with different backends
- Combining tables from different sources

**Commands:**
- `xorq catalog ls` - Check expression backends
- `xorq agents onboard` - Review deferred execution patterns
"""

    # Expression not found
    elif "not found in catalog" in error_lower or "expression not found" in error_lower:
        guidance = """
📦 CATALOG ERROR DETECTED

The expression you're trying to use hasn't been cataloged yet.

**Steps to resolve:**
1. List available expressions:
   ```bash
   xorq catalog ls
   ```

2. Build and catalog your expression:
   ```bash
   xorq build expr.py -e expr
   xorq catalog add .xorq/builds/<hash> --alias <name>
   ```

3. Verify it was added:
   ```bash
   xorq catalog ls
   ```

**Tip:** Use `xorq agents vignette list` to see example patterns
"""

    # Build failures
    elif "build failed" in error_lower or "failed to build" in error_lower:
        guidance = """
🏗️ BUILD ERROR DETECTED

Your expression failed to build. Common causes:

**1. Missing schema inspection:**
```python
# Add explicit schema inspection
expr.schema()  # Force schema resolution
```

**2. Eager operations in expression:**
Check for: `.to_pandas()`, `.compute()`, `.execute()`, `pd.read_`, `.iloc`, `.loc`

**3. Invalid expression structure:**
Make sure your expression is:
- A valid xorq expression (xo._(...))
- Properly deferred (no eager evaluations)
- Has correct imports

**Debug commands:**
```bash
xorq agents vignette scaffold <name> --dest test.py  # Get working template
xorq build expr.py -e expr  # Try building
```
"""

    # Import errors
    elif "importerror" in error_lower or "modulenotfounderror" in error_lower:
        guidance = """
📥 IMPORT ERROR DETECTED

Missing imports or module not found.

**Check:**
1. Is xorq installed? `pip list | grep xorq`
2. Are required packages installed? (sklearn, polars, etc.)
3. Is the module name correct?

**For xorq expressions:**
```python
import xorq as xo
from xorq import manifest as xo_manifest
```

**Get help:**
- `xorq agents vignette list` - See working examples
- `xorq agents onboard` - Review workflow
"""

    # Deferred execution violations
    elif "to_pandas" in error_lower or "eager" in error_lower:
        guidance = """
⚠️ DEFERRED EXECUTION VIOLATION

Detected eager pandas/computation operations.

**Xorq principle:** Everything stays deferred until execution.

**Instead of:**
```python
df = expr.to_pandas()  # ❌ Don't do this
df.groupby('col').mean()
```

**Do this:**
```python
# Build expression
xorq build expr.py -e expr

# Run and pipe output
xorq run <alias> -f arrow -o result.arrow
```

**Learn more:**
- `xorq agents onboard` - Core workflow
- `xorq agents vignette list` - Pattern examples
"""

    return guidance


def main():
    """PostToolUseFailure hook handler."""
    try:
        # Read tool failure information from environment or stdin
        tool_output = os.environ.get("TOOL_OUTPUT", "")

        # Check if this looks like an xorq error
        if not any(keyword in tool_output.lower() for keyword in
                   ["xorq", "catalog", "manifest", "build", "deferred", "coerce"]):
            # Not an xorq-related error, skip
            return 0

        guidance = get_troubleshooting_guidance(tool_output)

        if guidance:
            # Append troubleshooting guidance
            print(guidance)

    except Exception:
        # Don't block on errors
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
