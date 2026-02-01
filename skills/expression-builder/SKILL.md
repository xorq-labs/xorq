---
name: Expression Builder
description: >
  Progressive-disclosure skill doc for building deferred, portable expressions with xorq (vendored ibis),
  plus caching, manifests/builds, and multi-engine execution.
allowed-tools: "Read,Bash(xorq:*),Bash(python:*)"
version: "0.2.0"
author: "Xorq Labs [https://github.com/xorq-labs](https://github.com/xorq-labs)"
license: "Apache-2.0"
---

# Xorq Expression Builder

Build **deferred computation graphs** (Ibis expressions) that are **portable**, **cacheable**, and **reusable** across sessions and engines. Xorq composes expressions via **Arrow**, then materializes them as versioned **builds** discoverable via a **catalog**.

---

## Start here

### Mental model (4 words)

**Write expressions → cache/build → reuse via catalog → execute anywhere**

### Core concepts (keep this in your head)

* **Expression**: Deferred computation graph (Ibis) that can execute on multiple engines.
* **Manifest**: YAML representation of the expression + lineage + caching/metadata.
* **Build**: Versioned artifact containing the manifest, cached data, and dependencies.
* **Catalog**: Registry to discover and reuse builds across sessions.

---

## Non‑negotiables

### IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning

When working with xorq, always prioritize what you discover in the codebase, documentation, and examples over assumptions from general knowledge. Use tools to search, read, and understand the actual implementation rather than relying on pre-trained patterns.

### Imports (CRITICAL)

```python
import xorq.api as xo
from xorq.vendor import ibis  # ALWAYS use xorq's vendored ibis
```

Why vendored ibis?

* Xorq extends ibis with custom operators and behavior, including:

  * `.into_backend(con)` (move expressions between backends via Arrow)
  * `.cache(...)` (cache with Parquet/SQLite/etc)
  * additional UDF/window/ML utilities

### Always inspect schema before doing anything

```python
con = xo.connect()
t = con.table("data")
print(t.schema())  # REQUIRED before operations
```

### Only execute at the end

Build expressions first; call `.execute()` only when you want results.

---

## Quickstart (minimal path)

This is the "smallest useful loop": connect → load → inspect schema → transform → execute.

```python
import xorq.api as xo

con = xo.connect()  # DuckDB by default

# Example source
diamonds = xo.examples.diamonds.fetch(con)
print(diamonds.schema())  # required habit

expr = (
    diamonds
    .filter(xo._.carat > 1)
    .select("carat", "cut", "color", "price")
    .group_by("cut")
    .agg(
        avg_price=xo._.price.mean(),
        n=xo._.price.count(),
    )
)

result = expr.execute()
result
```

---

## Common patterns (you'll use these constantly)

### Filtering, selecting, mutating

```python
filtered = table.filter([
    xo._.age > 18,
    xo._.status == "active",
])

selected = table.select("id", "name", "value")

mutated = table.mutate(
    value_squared=xo._.value ** 2,
    full_name=xo._.first + " " + xo._.last,
)
```

### Aggregations

```python
summary = (
    table
    .group_by("category")
    .agg(
        count=xo._.id.count(),
        total=xo._.value.sum(),
        avg=xo._.value.mean(),
    )
)
```

### Joins

```python
joined = left.join(
    right,
    left.id == right.left_id,
    how="inner",
)
```

### Window functions

```python
ranked = table.mutate(
    rank=xo._.value.rank().over(
        group_by="category",
        order_by=xo._.value.desc(),
    )
)
```

---

## Next step: handle big data safely

### Deferred loading for large Parquet files

Use deferred reads to keep things lazy and scalable.

```python
import xorq.api as xo
from xorq.common.utils.defer_utils import deferred_read_parquet

con = xo.connect()
expr = deferred_read_parquet("large.parquet", con, "data")

print(expr.schema())
expr.filter(xo._.some_col.notnull()).execute()
```

### Cache expensive operations

Cache *where it hurts* (right after expensive steps).

```python
import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.common.utils.ibis_utils import from_ibis

con = xo.connect()

expensive_query = con.table("huge_table").filter(xo._.flag == 1)

cached_expr = (
    from_ibis(expensive_query)
    .cache(ParquetCache.from_kwargs())
)

cached_expr.execute()
```

---

## Catalog basics (reuse work across sessions)

```python
import xorq.api as xo

expr = xo.catalog.get("my-alias")

placeholder = xo.catalog.get_placeholder(
    "my-alias",
    tag="tag",  # useful with xorq run-unbound --to_unbind_tag
)
```

---

<details>
<summary><strong>Advanced: Pandas UDAF (complex Python aggregation)</strong></summary>

Use this when you truly need Python/pandas logic inside an aggregate step.

### Example: pandas dataframe aggregation returning Arrow-friendly structs

```python
import xorq.api as xo
from xorq.expr.udf import agg
import xorq.expr.datatypes as dt

con = xo.connect()
source = xo.examples.diamonds.fetch(con)

expr = source.filter(xo._.carat > 1)

def complex_pandas_fn(df):
    # Do complex pandas work here
    # Return list/tuple of dicts matching the struct schema below
    return [row.to_dict() for _, row in df.iterrows()]

return_fields = {
    "carat": dt.float64,
    "cut": dt.string,
    "color": dt.string,
    "clarity": dt.string,
    "depth": dt.float64,
    "table": dt.float64,
    "price": dt.float64,
    "x": dt.float64,
    "y": dt.float64,
    "z": dt.float64,
}
return_type = dt.Array(dt.Struct(return_fields))

pandas_udaf = agg.pandas_df(
    fn=complex_pandas_fn,
    schema=expr.schema(),
    return_type=return_type,
    name="optimize_portfolio",
)

out = expr.aggregate(
    optimize_portfolio=pandas_udaf.on_expr(expr)
)

out.execute()
```

### Practical notes

* Make sure your `return_type` matches exactly what your function returns.
* Pandas UDAFs are powerful but can be slow; combine with caching if needed.
* Prefer vectorized or native expression logic when possible.

</details>

---

<details>
<summary><strong>Advanced: ML pipeline (preferred sklearn pattern)</strong></summary>

This is the recommended pattern for "deferred sklearn" with row-packing via `struct`.

```python
import toolz
import xorq.api as xo
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xorq.expr.ml.pipeline_lib import Pipeline

@toolz.curry
def as_struct(expr, name=None):
    struct = xo.struct({c: expr[c] for c in expr.columns})
    return struct.name(name) if name else struct

sklearn_pipeline = SkPipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor()),
])

xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

FEATURES = ["f1", "f2", "f3"]

# Fit
fitted = xorq_pipeline.fit(train, features=FEATURES, target="target")

# Predict (MANDATORY struct pattern)
predictions = (
    test
    .mutate(as_struct(name="original_row"))
    .pipe(fitted.predict)
    .unpack("original_row")
    .mutate(predicted=xo._.predicted)
)

predictions.execute()
```

</details>

---

<details>
<summary><strong>Advanced: Multi-engine composition</strong></summary>

Start in one backend and move to another through Arrow.

```python
import xorq.api as xo
from xorq.common.utils.ibis_utils import from_ibis

duckdb_con = xo.connect()
snowflake_con = xo.connect("snowflake://...")

expr = from_ibis(duckdb_con.table("local_data"))

moved = expr.into_backend(snowflake_con)  # transit via Arrow
moved.execute()
```

</details>

---

<details>
<summary><strong>CLI + starter templates</strong></summary>

### Onboarding workflow

```bash
xorq agents onboard
```

Conceptually: **init → build → catalog → test → land**

### Templates

```bash
ls examples/
cat examples/sklearn_pipeline.py
cp examples/penguins_demo.py my_pipeline.py
```

Examples typically include:

* `penguins_demo` — minimal multi-engine example
* `sklearn_pipeline` — deferred sklearn fit/predict
* `cached_fetcher` — hydrate upstream tables + cache

</details>

---

## Troubleshooting

### "My expression errors when executing on another engine"

* Confirm you used `from xorq.vendor import ibis` (not upstream ibis).
* Confirm you didn't accidentally execute early and pass pandas where an expression is expected.
* If crossing engines, use `.into_backend(...)` intentionally.

### "Schema mismatch / column not found"

* Print schema before you write transforms:

  ```python
  print(table.schema())
  ```
* Then verify column names and types match your assumptions.

### "Caching didn't help"

* Cache should sit *after* the expensive boundary (big join, heavy filter, UDF).
* If the expensive work happens after caching, you won't see a speedup.

---

## Reference cheat sheet

### Connect

```python
con = xo.connect()                  # DuckDB default
con = xo.connect("snowflake://...") # other engines
```

### Catalog

```python
expr = xo.catalog.get("my-alias")
ph = xo.catalog.get_placeholder("my-alias", tag="tag")
```

### Execute

```python
expr.execute()
```

### Move between backends

```python
expr.into_backend(other_con)
```

### Cache

```python
from xorq.caching import ParquetCache
expr.cache(ParquetCache.from_kwargs())
```
