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
import xorq as xo

expr = (
    ibis.read_parquet("penguins.parquet")
    .filter(ibis._.species.notnull())
    .group_by("species")
    .agg(avg_bill_length=ibis._.bill_length_mm.mean())
)

xo_expr = xo.from_ibis(expr)
```

```bash
xorq build pipeline.py -e xo_expr
```

```yaml
# Addressable, composable, portable
xo_expr:
  hash: 7061dd65ff3c
  op: Aggregate
  inputs:
    - species
    - bill_length_mm
  source: penguins.parquet
```

Same computation = same hash. The manifest *is* the version. The hash *is* the address.

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

Manifests are portable. Execute on DuckDB locally, compile to Snowflake for production.

One manifest, any engine.

## Integrations

Ibis • scikit-learn • Feast • dbt

## Learn more

- [Quickstart tutorial](https://docs.xorq.dev/tutorials/getting_started/quickstart)
- [Why Xorq?](https://docs.xorq.dev/#why-xorq)
- [Scikit-learn template](https://github.com/xorq-labs/xorq-template-sklearn)

---

Pre-1.0. Expect breaking changes with migration guides.
