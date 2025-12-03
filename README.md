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

## The Problem

Feature stores. Model registries. Orchestrators. Vertical silos that don't
serve agentic AI—which needs context and skills, not categories.

## Xorq

**Manifest = Context.** Every ML computation becomes a structured, addressable YAML manifest.

**Tools = Skills.** A catalog to discover. A build system to execute anywhere.

## Quick Start
```bash
pip install xorq[examples]
xorq init -t penguins
```

## Manifest

Write [Ibis](https://ibis-project.org) expressions, get
addressable manifests.

```python

import ibis

from xorq.common.utils.ibis_utils import from_ibis
from xorq.caching import ParquetStorage

penguins_tbl = ibis.examples.penguins.fetch()

penguins_expr = (
    penguins_tbl
    .filter(ibis._.species.notnull())
    .group_by("species")
    .agg(avg_bill_length=ibis._.bill_length_mm.mean())
)

xo_expr = from_ibis(penguins_expr)
```

```bash
xorq build expr.py -e xo_expr
```
```bash
❯ lt builds/3c840882fa37
builds/3c840882fa37
├── database_tables
│   └── 7a325162f40d8ac265d94b025b9d37a3.parquet
├── expr.yaml
├── metadata.json
├── profiles.yaml
└── sdist.tar.gz
```
Reproducible build artifacts with `uv` based environments.

And roundtrippable, human-diffable YAML.

```yaml
# Addressable, composable, portable
definitions:
  nodes:
    '@filter_68655050':
      op: Filter
      parent:
        node_ref: '@read_e1fcd64e'
      predicates:
      - args:
        - name: species
          op: Field
          relation:
            node_ref: '@read_e1fcd64e'
          type:
            type_ref: type_1
        op: NotNull
        type:
          type_ref: type_0
      snapshot_hash: 68655050dd9e691abb7b31308eed1f3f
    '@read_e1fcd64e':
      method_name: read_parquet
      name: penguins
      normalize_method: gAWVWAAAAAAAAACMNXhvcnEuY29tbW9uLnV0aWxzLmRhc2tfbm9ybWFsaXplLmRhc2tfbm9ybWFsaXplX3V0aWxzlIwabm9ybWFsaXplX3JlYWRfcGF0aF9tZDVzdW2Uk5Qu
      op: Read
      profile: 2eca7579af9a9d8e315faf6af1ddb59a_2
      read_kwargs:
      - - source_list
        - builds/3c840882fa37/database_tables/7a325162f40d8ac265d94b025b9d37a3.parquet
      - - table_name
        - penguins
      schema_ref: schema_1
      snapshot_hash: e1fcd64eb0e8c9d39aa07787ed7523ca
```

Same computation = same hash. The manifest *is* the version. The hash *is*
the address.

## Tools
```bash
# Discover

xorq catalog ls

# Trace lineage
xorq lineage fraud-model

# Register
xorq catalog add builds/7061dd65ff3c --alias fraud-model
```

## The Architecture

Write in Python. Catalog as YAML. Execute anywhere via Ibis.

![Architecture](docs/images/architecture-light.png#gh-light-mode-only)
![Architecture](docs/images/architecture-dark.png#gh-dark-mode-only)

Lineage, caching, and versioning travel with the manifest—cataloged, not
locked in a vendor's database.


## Multi-Engine

Manifests are portable. Execute on DuckDB locally, compile to multi-engine
Snowflake for production, and use Python  with Xorq's embedded engine, based
on DataFusion.

One manifest, many engines.

## Integrations

Ibis • scikit-learn • Feast • dbt

## Learn more

- [Quickstart tutorial](https://docs.xorq.dev/tutorials/getting_started/quickstart)
- [Why Xorq?](https://docs.xorq.dev/#why-xorq)
- [Scikit-learn template](https://github.com/xorq-labs/xorq-template-sklearn)

---

Pre-1.0. Expect breaking changes with migration guides.
