---
name: xorq
description: >
  Manifest-driven compute for ML. Build expressions, catalog them, compose pipelines.
  Everything flows through the catalog.
version: "0.3.0"
author: "Xorq Labs"
---

# Xorq

Build expressions. Catalog them. Compose pipelines. Run anywhere.

## The Workflow

```
build → catalog → compose → run
```

That's it. Every xorq session follows this loop.

## Quick Start

```bash
# 1. Build an expression from Python
xorq build pipeline.py -e predictions

# 2. Add to catalog with an alias
xorq catalog add builds/<hash> --alias predictions

# 3. Run it
xorq run predictions -o results.parquet
```

## Catalog is the API

The catalog is how you discover, share, and compose work.

```bash
xorq catalog ls                    # What's available?
xorq catalog add <build> --alias   # Register new work
xorq lineage <alias>               # How was this built?
```

In Python, the catalog is your entry point:

```python
import xorq.api as xo

# Load from catalog
expr = xo.catalog.get("predictions")

# Compose further
better = expr.filter(xo._.confidence > 0.9)
```

---

## Level 1: Building Expressions

An expression is a deferred computation. Build one in Python:

```python
# pipeline.py
import xorq.api as xo
from xorq.api import _

con = xo.connect()  # DuckDB default
data = con.table("my_table")

# This is an expression - nothing executes yet
expr = (
    data
    .filter(_.status == "active")
    .select("id", "value", "category")
    .group_by("category")
    .agg(total=_.value.sum())
)
```

Build it:

```bash
xorq build pipeline.py -e expr
# → builds/<content-hash>/
```

The build captures the expression graph, schema, and lineage.

---

## Level 2: Composing from Catalog

Once cataloged, expressions compose naturally:

```python
import xorq.api as xo

# Pull from catalog
base = xo.catalog.get("clean-data")
features = xo.catalog.get("feature-eng")

# Compose into new pipeline
combined = base.pipe(lambda t: features)

# Or just transform further
filtered = base.filter(xo._.quality_score > 0.8)
```

**Key insight**: Catalog aliases are stable references. The underlying build can change, but downstream code doesn't break.

---

## Level 3: ML Pipelines

Sklearn pipelines become deferred expressions:

```python
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from xorq.expr.ml.pipeline_lib import Pipeline
from xorq.caching import ParquetCache
import xorq.api as xo

# Define sklearn pipeline
sk_pipeline = SkPipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

# Wrap for deferred execution
xorq_pipeline = Pipeline.from_instance(sk_pipeline)

# Fit returns a fitted pipeline expression
train = xo.catalog.get("train-data")
fitted = xorq_pipeline.fit(train, features=FEATURES, target=TARGET)

# Predict is also deferred
test = xo.catalog.get("test-data")
predictions = fitted.predict(test)
```

Cache expensive operations:

```python
from xorq.caching import ParquetCache

cached_predictions = predictions.cache(ParquetCache.from_kwargs())
```

---

## Level 4: Multi-Engine Execution

Same expression, different engines:

```python
import xorq.api as xo
from xorq.common.utils.ibis_utils import from_ibis

# Local development
local = xo.connect()

# Production
prod = xo.connect("snowflake://...")

# Move expression between engines
expr = from_ibis(local.table("data")).into_backend(prod)
```

---

## CLI Reference

| Command | What it does |
|---------|--------------|
| `xorq build <file> -e <name>` | Build expression to manifest |
| `xorq catalog ls` | List cataloged builds |
| `xorq catalog add <path> --alias <name>` | Register a build |
| `xorq run <alias> -o <file>` | Execute and output |
| `xorq lineage <alias>` | Show column-level lineage |
| `xorq agents onboard` | Guided workflow |
| `xorq agents prime` | Context for current project |

---

## Patterns

### Deferred Loading (Large Files)

```python
from xorq.common.utils.defer_utils import deferred_read_parquet

expr = deferred_read_parquet("large.parquet", con, "data")
```

### Conditional Logic

```python
from xorq.vendor import ibis

score = ibis.cases(
    (expr.grade == "A", 3),
    (expr.grade == "B", 2),
    else_=1
)
```

### Train/Test Split

```python
train, test = xo.train_test_splits(
    data,
    test_sizes=0.2,
    num_buckets=1000,
    random_seed=42
)
```

### Metrics

```python
from xorq.expr.ml.metrics import deferred_sklearn_metric
from sklearn.metrics import accuracy_score

acc = deferred_sklearn_metric(
    expr=predictions,
    target="label",
    pred_col="predicted",
    metric_fn=accuracy_score
)
```

---

## Rules

**Column case matters**
- Snowflake: `UPPERCASE`
- DuckDB: `lowercase`

Always check: `print(table.schema())`

**Use xorq's ibis**
```python
from xorq.vendor import ibis  # Not: import ibis
```

**Commit your catalog**
```bash
git add .xorq/catalog.yaml builds/
git commit -m "Add pipeline"
```

---

## Session Protocol

```bash
# Start
xorq agents prime

# Work
xorq build → xorq catalog add → xorq run

# End
git add .xorq/catalog.yaml builds/
git commit -m "Session work"
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Column not found | `print(table.schema())` - match case exactly |
| Expression won't run | Check connection: `con.list_tables()` |
| Import error | Use `from xorq.vendor import ibis` |
| Cache miss | Expression hash changed - rebuild |

---

## Resources

Detailed documentation for deep dives:

| Resource | Description |
|----------|-------------|
| [expression-api.md](resources/expression-api.md) | Expression building patterns and API reference |
| [ml-pipelines.md](resources/ml-pipelines.md) | ML/sklearn integration with UDAF examples |
| [caching.md](resources/caching.md) | Performance optimization and caching strategies |
| [udf-udxf.md](resources/udf-udxf.md) | UDFs, UDAFs, and Flight server patterns |
| [optimization-patterns.md](resources/optimization-patterns.md) | Portfolio optimization and MILP patterns |
| [examples.md](resources/examples.md) | End-to-end examples with 41 complete scripts |
| [CLI_REFERENCE.md](resources/CLI_REFERENCE.md) | Complete CLI command documentation |
| [WORKFLOWS.md](resources/WORKFLOWS.md) | Step-by-step workflow patterns |
| [TROUBLESHOOTING.md](resources/TROUBLESHOOTING.md) | Common issues and solutions |

---

## Links

- [github.com/xorq-labs/xorq](https://github.com/xorq-labs/xorq)
- [docs.xorq.dev](https://docs.xorq.dev)
