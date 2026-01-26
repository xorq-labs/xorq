---
name: xorq
description: >
    compute catalog with deferred expressions built on Ibis.
allowed-tools: "Read,Bash(xorq:*),Bash(python:*)"
version: "0.2.0"
author: "Xorq Labs <https://github.com/xorq-labs>"
license: "Apache-2.0"
---

Your goal is to generate deferred expressions using Xorq framework wrapping
thing that you already know, e.g. pandas and scikit-learn. Xorq is built on
ibis and as such exposes a 1:1 ibis compatible api with a few differences:
1.`cache` is deferred and can take in ParquetCache or SourceCache as arguments and is multi-engine
2. UDF mechanism is enhacned with pandas udf `xo.expr.udf import
   make_pandas_udf` and `make_pandas_expr_udf`, or `xo.expr.udf.agg`.
3. Provides scikit-learn pipeline object to make it deferred `xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)`

Since the the output of a deferred UDF can be pickled and binary types are
supported, we can also use them to export and save any matplotlib plots as
binary blog, allowing us to complete the loop of staying deferred even for results.

Here is an example flow:

```python
import xorq.api as xo
from xorq.expr.udf import agg
import xorq.expr.datatypes as dt

#source_expr = xo.catalog.get("expr-alias")

source_expr = xo.examples.diamonds.fetch()
expr = source_expr.filter(xo._.carat >1)

def complex_pandas_fn(df):
    # complex things
    return df


return_fields = {
    'carat': dt.float64,
    'cut': dt.string,
    'color': dt.string,
    'clarity': dt.string,
    'depth': dt.float64,
    'table': dt.float64,
    'price': dt.float64,
    'x': dt.float64,
    'y': dt.float64,
    'z': dt.float64,
}
return_type = dt.Array(dt.Struct(return_fields))

complex_pandas_udaf = agg.pandas_df(
    fn=complex_pandas_fn,
    schema=expr.schema(),
    return_type=return_type,
    name='optimize_portfolio'
)

expr = expr.aggregate(complex_pandas_udaf.on_expr)

expr.execute()
```


# do some complicated pandas df stuff


## Quick Start

**Start with a vignette (recommended):**
```bash
# See comprehensive working examples
xorq agents vignette list

# Scaffold a complete ML pipeline example
xorq agents vignette scaffold baseball_breakout_expr_scalar
```

**Or build from scratch:**
```bash
xorq agents prime
```

💡 **Vignettes show advanced patterns** like ExprScalarUDF, windowing, and ML pipelines with xorq's vendored ibis.

## Essential CLI Commands

| Command | Purpose |
|---------|---------|
| `xorq init -t <template>` | Initialize project from template |
| `xorq build <file> -e <expr>` | Build expression to manifest |
| `xorq run <alias>` | Execute cataloged build |
| `xorq catalog add/ls` | Manage build registry |
| `xorq lineage <alias>` | Show column-level lineage |
| `xorq agents onboard` | Guided workflow for agents |

**Full reference:** Run `xorq --help` or see [resources/CLI_REFERENCE.md](resources/CLI_REFERENCE.md)

## Python API Essentials

### Imports and Connection

**✅ Correct imports (CRITICAL):**
```python
import xorq.api as xo
from xorq.vendor import ibis  # ⚠️ ALWAYS use xorq's vendored ibis
from xorq.caching import ParquetCache

expr = xo.catalog.get("my-alias")           # Load from catalog
placeholder = xo.catalog.get_placeholder("my-alias", tag="tag")  # tag to easily use with xorq run-unbound --to_unbind_tag

con = xo.connect()  # DuckDB default
```

### Expression Building Patterns

```python
expr = (
    table
    .filter(xo._.column.notnull())
    .select("id", "value", "category")
    .group_by("category")
    .agg(total=xo._.value.sum())
)

result = expr.execute()
```

### Deferred Loading (Large Files)

```python
from xorq.common.utils.defer_utils import deferred_read_parquet

expr = deferred_read_parquet("large.parquet", con, "data")
```

### Caching Expensive Operations

```python
from xorq.caching import ParquetCache
from xorq.common.utils.ibis_utils import from_ibis

cached_expr = (
    from_ibis(expensive_query)
    .cache(ParquetCache.from_kwargs())  # Cache here
)
```

### ML Pipeline (Preferred Pattern)

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

# 2. Create sklearn pipeline
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
    .mutate(as_struct(name="original_row"))  # Pack columns
    .pipe(fitted.predict)                     # Predict
    .unpack("original_row")                   # Unpack
    .mutate(predicted=_.predicted)            # Use result
)
```
### Column Case Sensitivity

- **Snowflake**: UPPERCASE columns
- **DuckDB**: lowercase columns
- **Match exactly** as shown in schema

### Deferred Execution Only

```python
# ✅ Good: Deferred xorq expressions
expr = table.filter(xo._.status == "active")

# ❌ Avoid: Pandas/NumPy scripts
df = pd.read_parquet("data.parquet")  # Eager loading
```

### Catalog Management

```bash
# After building, always catalog
xorq catalog add builds/<hash> --alias my-pipeline

# View cataloged builds
xorq catalog ls

# CRITICAL: Commit catalog before session end
git add .xorq/catalog.yaml builds/
git commit -m "Add pipeline to catalog"
```


### Development Loop

```bash
# 1. Check schema (mandatory)
python -c "import xorq.api as xo; con = xo.connect(); print(con.table('data').schema())"

# 2. Build expression
xorq build expr.py -e expr

# 3. Catalog the build
xorq catalog add builds/<hash> --alias my-expr

# 4. Test it
xorq run my-expr -o test.parquet

# 5. Check lineage
xorq lineage my-expr
```

### Close Session (CRITICAL)

```bash
# 1. Commit catalog and builds
git add .xorq/catalog.yaml builds/
git commit -m "Update catalog"

# 2. Push changes
git push
```

## Advanced Workflow Patterns

### DuckDB CLI Exploration

**Pattern:** Stream xorq outputs through Arrow IPC to DuckDB for interactive SQL exploration.

```bash
# Simple: Stream source to DuckDB
xorq run source -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow; SELECT * FROM read_arrow('/dev/stdin') LIMIT 10"

# Advanced: Compose pipeline and explore interactively
xorq run source -f arrow -o /dev/stdout 2>/dev/null | \
  xorq run-unbound transform \
    --to_unbind_hash <hash> \
    --typ xorq.expr.relations.Read \
    -f arrow -o /dev/stdout 2>/dev/null | \
  duckdb -c "LOAD arrow;
    SELECT col1, COUNT(*) FROM read_arrow('/dev/stdin')
    GROUP BY col1"
```

**When to use:**
- Interactive SQL exploration without writing Python
- Ad-hoc data validation and quick analysis
- Testing pipeline outputs rapidly

**Reference:** [Workflows #9](resources/WORKFLOWS.md#9-arrow-ipc-streaming-with-duckdb-interactive-exploration)

---

## Common Expression Patterns

### Filtering and Selection

```python
# Filter with conditions
filtered = table.filter([
    xo._.age > 18,
    xo._.status == "active"
])

# Select columns
selected = table.select("id", "name", "value")

# Add computed columns
mutated = table.mutate(
    value_squared=xo._.value ** 2,
    full_name=xo._.first + " " + xo._.last
)
```

### Aggregations

```python
# Group and aggregate
agg = table.group_by("category").agg(
    count=xo._.id.count(),
    total=xo._.value.sum(),
    avg=xo._.value.mean()
)
```

### Joins

```python
# Inner join
joined = left.join(
    right,
    left.id == right.left_id,
    how="inner"
)
```

### Window Functions

```python
# Rank within groups
ranked = table.mutate(
    rank=xo._.value.rank().over(
        group_by="category",
        order_by=xo._.value.desc()
    )
)
```


### Templates (Starter Code)

```bash
# List available examples
ls examples/

# View example code
cat examples/sklearn_pipeline.py

# Copy an example to start
cp examples/penguins_demo.py my_pipeline.py
```

Available example patterns in examples/:
- `penguins_demo` - Minimal multi-engine example
- `sklearn_pipeline` - Deferred sklearn with train/predict
- `cached_fetcher` - Hydrate and cache upstream tables

### Onboarding Workflow

```bash
xorq agents onboard
```

Steps: **init → build → catalog → test → land**

## Multi-Engine Support

```python
# Start with DuckDB
duckdb_con = xo.connect()

# Move to Snowflake
snowflake_con = xo.connect("snowflake://...")

# Compose across backends
expr = (
    from_ibis(duckdb_con.table("local_data"))
    .into_backend(snowflake_con)  # Transit via Arrow
)
```

## Resources (Progressive Disclosure)

| Resource | Content |
|----------|---------|
| [Expression API](resources/expression-api.md) | Fluent transformations, filters, joins, window functions |
| [ML Pipelines](resources/ml-pipelines.md) | Sklearn integration, deferred fit/predict, pipelines |
| [Caching Strategies](resources/caching.md) | Performance optimization, storage backends |
| [UDFs & Flight Servers](resources/udf-udxf.md) | Custom functions, distributed processing |
| [Examples](resources/examples.md) | End-to-end working examples |
| [CLI Reference](resources/CLI_REFERENCE.md) | Complete command documentation |
| [Workflows](resources/WORKFLOWS.md) | Step-by-step patterns |
| [Troubleshooting](resources/TROUBLESHOOTING.md) | Common issues and fixes |

## Version

v0.2.0 - Consolidated skill with CLI + Python API coverage
