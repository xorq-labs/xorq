# Xorq Superpowers for Codex

<EXTREMELY_IMPORTANT>
You have xorq superpowers for building ML pipelines and deferred data analysis.

## Core xorq Workflow

**Non-negotiable rules:**
1. **ALWAYS check schema first**: `print(table.schema())` before any operations
2. **Use deferred expressions only**: No pandas/NumPy eager scripts
3. **Match column case exactly**: Snowflake=UPPERCASE, DuckDB=lowercase
4. **Catalog your builds**: `xorq catalog add builds/<hash> --alias name`
5. **Commit before session end**: Commit catalog and builds

## Essential Commands

```bash
# 1. Check schema (MANDATORY)
print(table.schema())

# 2. Build expression
xorq build expr.py -e expr

# 3. Catalog the build
xorq catalog add builds/<hash> --alias my-pipeline

# 4. Run when needed
xorq run my-pipeline -o output.parquet

# 5. Verify lineage
xorq lineage my-pipeline
```

## Tool Mapping for Codex

When xorq docs reference Claude-specific tools, use your Codex equivalents:
- `TodoWrite` → `update_plan` (your planning/task tracking tool)
- `Task` tool with subagents → Do the work directly (subagents not available in Codex)
- `Skill` tool → Not needed (you have this knowledge already)
- `Read`, `Write`, `Edit`, `Bash` → Use your native tools

## Critical Python Patterns

```python
# CRITICAL: Always use xorq's vendored ibis (NOT standalone ibis)
from xorq.vendor import ibis  # ✅ Has custom operators like .into_backend(), .cache(), ExprScalarUDF
# NOT: import ibis  # ❌ Missing xorq's deferred execution features
import xorq.api as xo

# Connect to backend
con = xo.connect()  # DuckDB default

# MANDATORY: Check schema first
table = con.table("data")
print(table.schema())  # Required!

# Build deferred expression
expr = (
    table
    .filter(xo._.column > 0)
    .select("id", "value")
    .group_by("category")
    .agg(total=xo._.value.sum())
)

# Execute when ready
result = expr.execute()
```

## ML Pipeline Pattern

```python
import toolz
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xorq.expr.ml.pipeline_lib import Pipeline

# 1. Create as_struct helper (REQUIRED)
@toolz.curry
def as_struct(expr, name=None):
    struct = xo.struct({c: expr[c] for c in expr.columns})
    return struct.name(name) if name else struct

# 2. Create and wrap sklearn pipeline
sklearn_pipeline = SkPipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor())
])
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

# 3. Fit on training data
fitted = xorq_pipeline.fit(train, features=FEATURES, target="target")

# 4. Predict with struct pattern (MANDATORY)
predictions = (
    test
    .mutate(as_struct(name="original_row"))
    .pipe(fitted.predict)
    .unpack("original_row")
    .mutate(predicted=_.predicted)
)
```

## Agent Workflow

1. **Start**: `xorq agents onboard` - Get context
2. **Build**: Create deferred expressions (check schema first!)
3. **Catalog**: Register builds with aliases
4. **Test**: Run and verify outputs
5. **Commit**: Ensure catalog and builds are committed

## When Working on xorq Projects

BEFORE writing any code:
1. Run `xorq agents onboard` to see project state
2. Check `xorq catalog ls` for existing pipelines
3. **Use vignettes for comprehensive examples**: `xorq agents vignette list`
   - Scaffold a vignette: `xorq agents vignette scaffold baseball_breakout_expr_scalar`
   - Vignettes show advanced patterns with ExprScalarUDF, ML pipelines, etc.
4. Check `ls examples/` for simple patterns
5. ALWAYS `print(table.schema())` before operations

AFTER completing work:
1. Verify catalog state with `xorq catalog ls`
2. Commit catalog and builds: `git add .xorq/catalog.yaml builds/`
3. Push changes

## Getting Help

- `xorq agents onboard` - Workflow context and status
- `xorq agents vignette list` - Comprehensive working examples with advanced patterns
- `ls examples/` - Simple example patterns
- `xorq catalog ls` - Cataloged builds
- `xorq --help` - Full command reference

IF A PROJECT HAS `.xorq/` DIRECTORY OR USES XORQ, YOU MUST FOLLOW THESE RULES.
</EXTREMELY_IMPORTANT>
