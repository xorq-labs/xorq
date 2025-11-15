<div align="center">

![Xorq Logo](docs/images/Xorq_WordMark_RGB_Midnight.png#gh-light-mode-only)
![Xorq Logo](docs/images/Xorq_WordMark_RGB_BlueSky.png#gh-dark-mode-only)

![License](https://img.shields.io/github/license/xorq-labs/xorq)
![PyPI - Version](https://img.shields.io/pypi/v/xorq)
![CI Status](https://img.shields.io/github/actions/workflow/status/xorq-labs/xorq/ci-test.yml)

**Build ML pipelines once. Run anywhere.**

Write Python → Execute as optimized SQL on DuckDB, Snowflake, BigQuery, or
any engine.

[Documentation](https://docs.xorq.dev) • [Discord](https://discord.gg/8Kma9DhcJG) • [Website](https://www.xorq.dev)

</div>

---

## What is Xorq?

The compute catalog that makes your ML processing portable and reusable. Think
**dbt for ML** - your Python transformations become versioned, cached, and
executable anywhere.

## Installation
```bash
pip install xorq[examples]
xorq init -t penguins
```
- [Tutorial](https://docs.xorq.dev/tutorials/getting_started/quickstart)

## Quick Start
```python
import xorq.api as xo
from sklearn.ensemble import RandomForestClassifier

data = xo.read_parquet('s3://bucket/penguins.parquet')
train, test = xo.test_train_splits(data, test_size=0.2)

model = xo.Pipeline.from_instance(RandomForestClassifier())
fitted = model.fit(train, features=['bill_length_mm', 'bill_depth_mm'],
                   target='species')

predictions = fitted.predict(test).cache(storage=ParquetStorage()) # deferred
predictions.execute() # do work

# predictions.build()
```

**CLI**:

```bash
xorq build expr.py -e predictions
```

```bash
xorq run builds/<build_hash>
```

## How it works

Xorq captures your ML computation as an **input-addressed manifest** - a YAML
representation where each node is identified by the hash of its computation
specification (not its results). This enables:

1. **Version by intent**: Same computation = same hash, regardless of input data
2. **Precise caching**: Cache results based on what you're computing
3. **Lineage tracking**: Full provenance from raw data through transformations to predictions
4. **Portable execution**: The manifest compiles to optimized SQL for any supported engine

```yaml
# Example manifest snippet showing fit → predict lineage
predicted:
  op: ExprScalarUDF            # Model inference
  kwargs:
    bill_length_mm: ...        # Feature inputs
    bill_depth_mm: ...
  meta:
    __config__:
      computed_kwargs_expr:    # Training lineage preserved
        op: AggUDF             # Model training
        kwargs:
          species: ...         # Original training target
```

### Input-addressing

Every computation gets a unique hash based on its logic:
- Same feature engineering on different days = **same hash** (reusable)
- Different feature logic = **different hash** (new version)

This enables perfect caching - if anyone on your team has run this exact
computation before, Xorq reuses it automatically.

## The Compute Catalog

Your team's shared repository of ML compute - versioned, discoverable, and
reusable:

```bash
❯ xorq catalog add builds/7061dd65ff3c --alias fraud-model

❯ xorq catalog ls
Aliases:
fraud-model                  7061dd65ff3c     r2
customer-features            dbf90860-88b3    r1
recommendation-pipeline      52f987594254     r1

❯ xorq lineage hn_classifier_v3

❯ xorq serve-unbound  --port 8815
```

## Learn More

- [Quickstart Tutorial](https://docs.xorq.dev/tutorials/getting_started/quickstart)
- [Why Xorq?](https://docs.xorq.dev/#why-xorq)
- [Scikit-learn Template](https://github.com/xorq-labs/xorq-template-sklearn)

## Status

Pre-1.0. Expect breaking changes with migration guides.
