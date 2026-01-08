<div align="center">

![Xorq Logo](docs/images/Xorq_WordMark_RGB_Midnight.png#gh-light-mode-only)
![Xorq Logo](docs/images/Xorq_WordMark_RGB_BlueSky.png#gh-dark-mode-only)

![License](https://img.shields.io/github/license/xorq-labs/xorq)
![PyPI - Version](https://img.shields.io/pypi/v/xorq)
![CI Status](https://img.shields.io/github/actions/workflow/status/xorq-labs/xorq/ci-test.yml)

**A compute manifest and composable tools for ML.**

[Documentation](https://docs.xorq.dev) • [Website](https://www.xorq.dev)

</div>

---

# The Problem

You write a feature pipeline. It works on your laptop with DuckDB. Deploying
it to Snowflake ends up in a rewrite. Intermediate results should be cached so you add infrastructure and a result naming system. A requirement to track pipeline changes is introduced, so you add a metadata store. Congrats, you're going to production! It's time to add a serving layer ...

Six months later: five tools that don't talk to each other and a pipeline only one person understands

| Pain | Symptom |
|------|---------|
| **Glue code everywhere** | Each engine is a silo. Moving between them means rewriting, not composing. |
| **Runtime Feedback** | Imperative Python code where you can only tell if something will fail while running the job.
| **Unnecessary recomputations** | No shared understanding of what changed. Everything runs from scratch. |
| **Opaque Lineages** | Feature logic, metadata, lineage. All in different systems. Debugging means archaeology. |
| **"Works on my machine"** | Environments drift. Reproducing results means reverse engineering someone's setup and interrogating your own. |
| **Stateful orchestrators** | Retry logic, task states, failure recovery. Another system to manage, another thing that breaks.

Feature stores, Model registries, Orchestrators: Vertical silos that don't
serve agentic processes, which needs context and skills, not categories.

# Xorq

![intro](docs/images/intro-light.svg#gh-light-mode-only)
![intro](docs/images/intro-dark.svg#gh-dark-mode-only)

**Manifest = Context.** Every ML computation becomes a structured,
input-addressed YAML manifest.

**Tools = Skills.** A catalog to discover. A build system to deterministically
execute anywhere with user directed caching.

```bash
pip install xorq[examples]
xorq init -t penguins
```

---

# The Expression

Write declarative [Ibis](https://ibis-project.org) expressions. Xorq extends
Ibis with caching, multi-engine execution, and UDFs.

```python
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

Declare `.cache()` on any node. Xorq handles the rest. No cache keys to generate or manage,
no invalidation logic to write.

## Compose across engines

One expression, many engines. Part of your pipeline runs on DuckDB, part on
Xorq's embedded [DataFusion](https://datafusion.apache.org) engine, UDFs

[comment]: # not all udfs are via arrow flight
via Arrow Flight. Xorq systematically handles data transit with low overhead. Bye byte glue code.

```python
expr = from_ibis(penguins).into_backend(xo.sqlite.connect())
expr.ls.backends
```
```
(<xorq.backends.sqlite.Backend at 0x7926a815caa0>,
 <xorq.backends.duckdb.Backend at 0x7926b409faa0>)
```

## Translate Python to many SQLs

Expressions are declarative i.e. you describe what, not how. When bound to a
backend, Xorq invokes that backend to generate an arrow record batch stream.
Errors surface at definition time, not during execution time.
Custom Python logic runs as UDFs, but the relational core is always SQL.
One expression, many dialects, early feedback.

---

# The Manifest

Build an expression, get a manifest.

```bash
xorq build expr.py
```

```
builds/28ecab08754e
├── database_tables
│   └── f2ac274df56894cb1505bfe8cb03940e.parquet
├── expr.yaml
├── metadata.json
└── profiles.yaml
└── sdist.tar.gz
```

No external metadata store. No separate lineage tool. The build directory *is*
the versioned, cached, portable artifact.

```yaml
# Input-addressed, composable, portable
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

## Reproducible builds

The manifest is roundtrippable—machine-readable and machine-writable. Git-diff
your pipelines. Code review your features. Track python dependencies. Rebuild from YAML alone.

```bash
xorq uv-build builds/28ecab08754e/

builds/28ecab08754e/dist/xorq_build-0.1.0.tar.gz
```

The build captures everything: expression graph, dependencies, memory tables.
Share the build that has sdist, get identical results. No "works on my machine."

## Only recompute what changed

The manifest is input-addressed: it describes *how* the computation was made,
not just what it is. Same inputs = same hash. Change an input, get a new hash.

```python
expr.ls.get_cache_paths()
```
```
(PosixPath('/home/user/.cache/xorq/parquet/letsql_cache-7c3df7ccce5ed4b64c02fbf8af462e70.parquet'),)
```

The hash *is* the cache key. No invalidation logic to debug.
If the expression is the same, the hash is the same, and the cache is valid.
Change an input, get a new hash, trigger recomputation.

Traditional caching asks "has this expired?" Input-addressed caching asks "is
this the same computation?" The second question has a deterministic answer.


---

# The Tools

The manifest provides context. The tools provide skills: catalog, introspect,
serve, execute.

## Catalog

```bash
# Add to catalog
xorq catalog add builds/28ecab08754e/ --alias penguins-agg
Added build 28ecab08754e as entry a498016e-5bea-4036-aec0-a6393d1b7c0f revision r1

# List entries
xorq catalog ls
Aliases:
penguins-agg    a498016e-5bea-4036-aec0-a6393d1b7c0f    r1
Entries:
a498016e-5bea-4036-aec0-a6393d1b7c0f    r1      28ecab08754e
```

## Run

```bash
xorq run builds/28ecab08754e -o out.parquet
```

## Serve

Serve expressions anywhere via Arrow Flight:

```bash
xorq serve-unbound builds/28ecab08754e/ \
  --to_unbind_hash 31f0a5be37713fe2c1a2d8ad8fdea69f \
  --host localhost --port 9002
```

```python
import xorq.api as xo

backend = xo.flight.connect(host="localhost", port=9002)
f = backend.get_exchange("default")

data = {
    "species": ["Adelie", "Gentoo", "Chinstrap"],
    "island": ["Torgersen", "Biscoe", "Dream"],
    "bill_length_mm": [39.1, 47.5, 49.0],
    "bill_depth_mm": [18.7, 14.2, 18.5],
    "flipper_length_mm": [181, 217, 195],
    "body_mass_g": [3750, 5500, 4200],
    "sex": ["male", "female", "male"],
    "year": [2007, 2008, 2009],
}

xo.memtable(data).pipe(f).execute()
```

```
     species  avg_bill_length
0     Adelie             39.1
1  Chinstrap             49.0
2     Gentoo             47.5
```

## Debug with confidence

No more archaeology. Lineage is encoded in the manifest—not scattered across
tools—and queryable from the CLI.

```bash
xorq lineage penguins-agg

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
```

## Workflows, without state

No task states. Just retry on failure.

Xorq executes expressions as Arrow RecordBatch streams. There's no DAG of tasks
to checkpoint, just data flowing through operators. If something fails, rerun
from the manifest. Cached nodes resolve instantly; the rest recomputes.

---

## Templates

Ready-to-start projects:

```bash
# Penguins aggregation
xorq init -t penguins

# Sklearn digits classification
xorq init -t sklearn
```

### Scikit-learn Integration

Xorq translates `scikit-learn` Pipeline objects to deferred expressions:

```python
from xorq.expr.ml.pipeline_lib import Pipeline

sklearn_pipeline = ...
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
```
---

## The Horizontal Stack

Write in Python. Catalog as YAML. Compose anywhere via Ibis. Portable compute
engine built on DataFusion. Universal UDFs via Arrow Flight.

![Architecture](docs/images/architecture-light.svg#gh-light-mode-only)
![Architecture](docs/images/architecture-dark.svg#gh-dark-mode-only)

Lineage, caching, and versioning travel with the manifest; cataloged, not locked
in a vendor's database.

**Integrations:** Ibis • scikit-learn • Feast(wip) • dbt (upcoming)

---

## Learn More

- [Quickstart tutorial](https://docs.xorq.dev/tutorials/getting_started/quickstart)
- [Why Xorq?](https://docs.xorq.dev/#why-xorq)
- [Scikit-learn template](https://github.com/xorq-labs/xorq-template-sklearn)

---

Pre-1.0. Expect breaking changes with migration guides.
