<div align="center">

![Xorq Logo](docs/images/Xorq_WordMark_RGB_Midnight.png#gh-light-mode-only)
![Xorq Logo](docs/images/Xorq_WordMark_RGB_BlueSky.png#gh-dark-mode-only)

![License](https://img.shields.io/github/license/xorq-labs/xorq)
![PyPI - Version](https://img.shields.io/pypi/v/xorq)
![CI Status](https://img.shields.io/github/actions/workflow/status/xorq-labs/xorq/ci-test.yml)

**A compute manifest and tools for ML.**

[Documentation](https://docs.xorq.dev) • [Discord](https://discord.gg/8Kma9DhcJG) • [Website](https://www.xorq.dev)

</div>

---

# The Problem

Feature stores. Model registries. Orchestrators. Vertical silos that don't
serve agentic AI—which needs context and skills, not categories.

# Xorq

![intro](docs/images/intro-light.svg#gh-light-mode-only)
![intro](docs/images/intro-dark.svg#gh-dark-mode-only)
**Manifest = Context.** Every ML computation becomes a structured, input addressed YAML manifest.

**Tools = Skills.** A catalog to discover. A build system to deterministically cache and execute anywhere.

## Quick Start
```bash
pip install xorq[examples]
xorq init -t penguins
```

## The Manifest

Write [Ibis](https://ibis-project.org) expressions, get human-diffable manifests.

```python
# expr.py
import ibis
from xorq.common.utils.ibis_utils import from_ibis
from xorq.caching import ParquetCache


penguins = ibis.examples.penguins.fetch()

penguins_agg = (
    penguins
    .filter(ibis._.species.notnull())
    .group_by("species")
    .agg(avg_bill_length=ibis._.bill_length_mm.mean())
)

expr = (
    from_ibis(penguins_agg)
    .cache(ParquetCache.from_kwargs())
)
```

```bash
xorq build expr.py
```
```bash
❯ lt builds/28ecab08754e/
builds/28ecab08754e
├── database_tables
│   └── f2ac274df56894cb1505bfe8cb03940e.parquet
├── expr.yaml
├── metadata.json
└── profiles.yaml
```

Reproducible build artifacts with `uv` based environments.

And roundtrippable and machine-readable.

```expr.yaml
# Input addressed, composable, portable
# Abridged expr.yaml
nodes:
  '@read_31f0a5be3771':
    op: Read
    name: penguins
    source: builds/28ecab08754e/.../f2ac274df56894cb1505bfe8cb03940e.parquet

  '@filter_23e7692b7128':
    op: Filter
    parent: '@read_31f0a5be3771'
    predicates:
      - NotNull(species)

  '@remotetable_9a92039564d4':
    op: RemoteTable
    remote_expr:
      op: Aggregate
      parent: '@filter_23e7692b7128'
      by: [species]
      metrics:
        avg_bill_length: Mean(bill_length_mm)

  '@cachednode_e7b5fd7cd0a9':
    op: CachedNode
    parent: '@remotetable_9a92039564d4'
    cache:
      type: ParquetCache
      path: parquet
```

Same computation = same hash. "Input addressing" means the address the
expression by the way it was made rather than what it is. The manifest *is* the
version. The hash *is* the address.

#### Portable UDFs

```
import pandas as pd
import xorq.api as xo

schema_in = xo.schema({"bill_length_mm": float, "bill_depth_mm": float})
schema_out = xo.schema({"bill_ratio": float})

def compute_ratio(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"bill_ratio": df["bill_length_mm"] / df["bill_depth_mm"]})

bill_ratio_udxf = xo.expr.relations.flight_udxf(
    process_df=compute_ratio,
    maybe_schema_in=schema_in,
    maybe_schema_out=schema_out,
    name="BillRatio",
)

penguins = xo.memtable({"bill_length_mm": [39.1, 46.5], "bill_depth_mm": [18.7, 17.4]})
expr = bill_ratio_udxf(penguins)

```
#### Multi-Engine
One manifest, many engines. Execute on DuckDB locally, translate to Snowflake
for production, run Python UDFs on Xorq's embedded [DataFusion](https://datafusion.apache.org) engine.

```profiles.yaml
2eca7579af9a9d8e315faf6af1ddb59a_2:
  con_name: duckdb
  idx: 2
  kwargs_tuple:
    database: ':memory:'
    extensions: null
    read_only: false
    temp_directory: null
feda6956a9ca4d2bda0fbc8e775042c3_3:
  con_name: let
  idx: 3
  kwargs_tuple:
    config: null
```

## The Tools
```bash
# Add
❯ xorq catalog add builds/28ecab08754e/ --alias penguins-agg
Added build 28ecab08754e as entry a498016e-5bea-4036-aec0-a6393d1b7c0f revision r1

# List
❯ xorq catalog ls
Aliases:
penguins-agg    a498016e-5bea-4036-aec0-a6393d1b7c0f    r1
Entries:
a498016e-5bea-4036-aec0-a6393d1b7c0f    r1      28ecab08754e

# Introspect

hussainsultan in 🌐 lets-pop in xorq hussain/docs/readme-updates*​​ ⇕≡
❯ xorq lineage penguins-agg

Lineage for column 'avg_bill_length':
Field:avg_bill_length #1
└── Cache xorq_cached_node_name_placeholder #2
    └── RemoteTable:236af67d399a4caaf17e0bf5e1ac4c0f #3
        └── Aggregate #4
            ├── Filter #5
            │   ├── Read #6
            │   └── NotNull #7
            │       └── Field:species #8
            │           └── ↻ see #6
            ├── Field:species #9
            │   └── ↻ see #5
            └── Mean #10
                └── Field:bill_length_mm #11
                    └── ↻ see #5

# Run
❯ xorq run builds/28ecab08754e -o out.parquet

```
### xorq-template-sklearn

Xorq provides utilities to translate `scikit-learn`'s `Pipeline` objects to a
deferred Xorq objects.

```
from xorq.expr.ml.pipeline_lib import (
    Pipeline,
)
sklearn_pipeline = ...
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
```

## Templates

Templates provide ready to start projects that can be customized for your ML use-case:

1. **Penguins** template
```
xorq init -t penguins
```

2. **Sklearn Digits** template

```
xorq init -t sklearn
```

## The Horizontal Stack

Write in Python. Catalog as YAML. Compose anywhere via Ibis. Portable compute
engine built on DataFusion. Universal UDFs via Arrow Flight.

![Architecture](docs/images/architecture-light.svg#gh-light-mode-only)
![Architecture](docs/images/architecture-dark.svg#gh-dark-mode-only)

Lineage, caching, and versioning travel with the manifest—cataloged, not
locked in a vendor's database.


### Integrations

Ibis • scikit-learn • Feast • dbt

# Learn more

- [Quickstart tutorial](https://docs.xorq.dev/tutorials/getting_started/quickstart)
- [Why Xorq?](https://docs.xorq.dev/#why-xorq)
- [Scikit-learn template](https://github.com/xorq-labs/xorq-template-sklearn)

---

Pre-1.0. Expect breaking changes with migration guides.
