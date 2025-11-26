<div align="center">

![Xorq Logo](docs/images/Xorq_WordMark_RGB_Midnight.png#gh-light-mode-only)
![Xorq Logo](docs/images/Xorq_WordMark_RGB_BlueSky.png#gh-dark-mode-only)

![License](https://img.shields.io/github/license/xorq-labs/xorq)
![PyPI - Version](https://img.shields.io/pypi/v/xorq)
![CI Status](https://img.shields.io/github/actions/workflow/status/xorq-labs/xorq/ci-test.yml)


**The periodic table for ML computation.**

Everything is an expression. Addressable. Composable. Portable.

Write high-level expression. Execute as SQL on DuckDB, Snowflake, BigQuery, or
any engine. Every computation addressable, versioned, and reusable.

[Documentation](https://docs.xorq.dev) • [Discord](https://discord.gg/8Kma9DhcJG) • [Website](https://www.xorq.dev)

</div>

---

## What is Xorq?

ML infrastructure is fragmented—features in one system, models in another,
lineage reconstructed through archaeology.

What if features, models, and pipelines aren't different things?

A feature is a computation. A model is a computation. A pipeline is
computations composed. The vendor categories aren't computational
truths—they're commercial territories. Strip away the product boundaries and
everything reduces to the same primitive: the expression.

Xorq is the composability layer for compute expressed as relational plans.

## Installation

```bash
pip install xorq[examples]
xorq init -t penguins
```

[Full Tutorial →](https://docs.xorq.dev/tutorials/getting_started/quickstart)

## Quick Start

```python
import xorq.api as xo
from sklearn.ensemble import RandomForestClassifier

data = xo.read_parquet('s3://bucket/penguins.parquet')
train, test = xo.test_train_splits(data, test_size=0.2)

model = xo.Pipeline.from_instance(RandomForestClassifier())
fitted = model.fit(train, features=['bill_length_mm', 'bill_depth_mm'],
                   target='species')

predictions = fitted.predict(test).cache(storage=ParquetStorage())  # deferred
predictions.execute()  # do work
```

**CLI:**

```bash
xorq build expr.py -e predictions
xorq run builds/
```

## How It Works

Xorq captures your ML computation as an **input-addressed manifest**—a
declarative representation where each node is identified by the hash of its
computation specification, not its results.

```yaml
# Manifest snippet: fit → predict lineage
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

### What This Enables

| Capability | How |
|------------|-----|
| **Version by intent** | Same computation = same hash, regardless of input data |
| **Precise caching** | Cache based on what you're computing, not when |
| **Structural lineage** | Provenance is the graph itself, not reconstructed logs |
| **Portable execution** | Manifest compiles to optimized SQL for any engine |

### Input-Addressing

Every computation gets a unique hash based on its logic:

- Same feature engineering on different days → **same hash** (reusable)
- Different feature logic → **different hash** (new version)

If anyone on your team has run this exact computation before, Xorq reuses it
automatically. The hash is the truth.

## The Catalog

Your team's shared ledger of ML compute—versioned, discoverable, composable:

```bash
# Register a build with an alias
❯ xorq catalog add builds/7061dd65ff3c --alias fraud-model

# Discover what exists
❯ xorq catalog ls
Aliases:
fraud-model                  7061dd65ff3c     r2
customer-features            dbf90860-88b3    r1
recommendation-pipeline      52f987594254     r1

# Trace lineage
❯ xorq lineage fraud-model

# Serve for inference
xorq serve-unbound  fraud-model --port 8001 405154f690d20f4adbcc375252628b75
```

The catalog isn't a database. It's an addressing system—discoverable by humans,
navigable by agents.

## The Architecture
![Architecture](docs/images/architecture-light.png#gh-light-mode-only)
![Architecture](docs/images/architecture-dark.png#gh-dark-mode-only)


## Learn More

- [Quickstart Tutorial](https://docs.xorq.dev/tutorials/getting_started/quickstart)
- [Why Xorq?](https://docs.xorq.dev/#why-xorq)
- [Scikit-learn Template](https://github.com/xorq-labs/xorq-template-sklearn)

## Status

Pre-1.0. Expect breaking changes with migration guides.
