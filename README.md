<div align="center">

![Xorq Logo](docs/images/Xorq_WordMark_RGB_Midnight.png#gh-light-mode-only)
![Xorq Logo](docs/images/Xorq_WordMark_RGB_BlueSky.png#gh-dark-mode-only)

![License](https://img.shields.io/github/license/xorq-labs/xorq)
![PyPI - Version](https://img.shields.io/pypi/v/xorq)
![CI Status](https://img.shields.io/github/actions/workflow/status/xorq-labs/xorq/ci-test.yml)

**Expression-based context engine with git-native versioning.**

[Documentation](https://docs.xorq.dev) • [Website](https://www.xorq.dev) • [Claude Code plugin](https://github.com/xorq-labs/claude-plugins)

</div>

---

Xorq is a CLI and a TUI application for building data pipelines as
content-addressed [Ibis](https://ibis-project.org) expressions, with a
git-native catalog for publishing and reusing them.
Additionally, Xorq context engine comes with:
1. Embedded [DataFusion](https://datafusion.apache.org) based engine
2. Deterministic Caching
3. [Arrow](https://arrow.apache.org) Flight-based serving

![xorq catalog TUI](docs/images/catalog-tui.png)

# The Problem

You ask a coding agent to build a dashboard. A few hours later you have one,
along with a folder of one-off Python scripts that import each other in
non-obvious ways, an embedded JSON holding intermediate state, and a
`requirements.txt` that was last regenerated two sessions ago. It runs
end-to-end on your laptop. Reproducing it on another machine, or
productionizing any of it, means rewriting most of it.

| Pain | Symptom |
|------|---------|
| **Scripts as deliverables** | The output of an agent run is a folder of `.py`, `.json`, and `.html`. Reproducing it means re-running scripts in the right order with the right state. |
| **Redundant compute** | Agents on the same task can't see each other's caches. The same join gets recomputed every session. |
| **Opaque runs** | Agents report what they did in prose. There's no versioned artifact to point at; supervising means reading transcripts. |
| **Lineage in chat history** | An upstream column rename breaks a downstream model. The dependency was never captured outside the chat that produced it. |
| **No way to publish** | An agent produces something reusable, but there's no shared store the next agent can discover it from. |
| **"Works on my sandbox"** | An environment that worked in one agent session doesn't work in the next. Reproducing means rebuilding the setup from scratch. |


# Two ways to start

**With an agent.** Install the xorq plugin in Claude Code and let it build
catalogs for you:

```
/plugin marketplace add xorq-labs/claude-plugins
/plugin install xorq@xorq-plugins
```

Four slash commands cover the lifecycle: `/xorq:init` ingests CSV or Parquet
files into a catalog, `/xorq:composer` joins catalog entries into new aliased
expressions, `/xorq:builder` constructs ML pipelines and semantic-layer
entries, and `/xorq:catalog-explore` discovers what's already there. The agent
does the building; you keep the catalog.

**Manually.** Install the library and start composing expressions in Python:

```bash
$ pip install xorq[examples]
$ xorq init -t penguins
```
---

# The Expression

Write declarative Ibis expressions that run like a tool. Xorq extends Ibis with
caching, multi-engine execution, and UDFs.

```python
import xorq.api as xo
from xorq.caching import ParquetCache

penguins = xo.examples.penguins.fetch()

penguins_agg = (
    penguins
    .filter(xo._.species.notnull())
    .group_by("species")
    .agg(avg_bill_length=xo._.bill_length_mm.mean())
)

expr = (
    penguins_agg
    .cache(ParquetCache.from_kwargs())
)
```


## One expression, many engines


```python
expr = penguins.into_backend(xo.sqlite.connect())
expr.ls.backends
```
```
(<xorq.backends.sqlite.Backend at 0x107debda0>,
 <xorq.backends.xorq_datafusion.Backend at 0x1669002c0>)
```

## Expressions are tools, Arrow is the pipe

Unix gave us small programs that compose via stdout. Xorq gives you expressions
that compose via Arrow.

```
In [6]: expr.to_pyarrow_batches()
Out[6]: <pyarrow.lib.RecordBatchReader at 0x15dc3f570>
```

## Workflows, without state

Xorq executes expressions as Arrow RecordBatch streams — no DAG of tasks to
checkpoint, just data flowing through operators.

## Scikit-learn pipelines

Xorq translates `scikit-learn` Pipeline objects to deferred expressions:

```python
from xorq.expr.ml.pipeline_lib import Pipeline

sklearn_pipeline = ...
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
```

---

# The Catalog

The catalog is a git repo of build artifacts on filesystem — `xorq catalog add`
packages a build into a zip with python environment and source using `uv`.

## Build and add

```bash
$ xorq uv build expr.p❯ xorq uv build expr2.py
Building wheel...
Successfully built ...
builds/fa2122f6a9e9

❯ xorq catalog -p git-catalogs/penguins init
Initialized catalog at /git-catalogs/penguins

❯ xorq catalog add builds/fa2122f6a9e9/ -a penguins-agg
Added fa2122f6a9e9
```

## Git history

Every catalog operation is a commit you can read:

```
❯ git -C git-catalogs/penguins reflog
17dd4e9 (HEAD -> main) HEAD@{0}: add: fa2122f6a9e9 (aliases penguins-agg)
9f5d242 HEAD@{1}: add catalog.yaml
9915df3 HEAD@{2}: commit: Switching to main
```

## Inside an entry

What gets zipped into each entry is human-readable:

```
$ tree builds/fa2122f6a9e9
├── build_metadata.json
├── expr.yaml
├── expr_metadata.json
├── profiles.yaml
├── requirements.txt
└── xorq-0.3.24-py3-none-any.whl
```

The manifest *is* the versioned, cached, portable artifact.

```yaml
# Input-addressed, composable, portable
# Abridged expr.yaml
definitions:
  nodes:
    '@read_b5f228c91f16':
      op: Read
      method_name: read_parquet
      name: penguins
      read_kwargs:
        - [hash_path, .../penguins/20250703T145709Z-c3cde/penguins.parquet]
        - [table_name, penguins]
      schema_ref: schema_f11dda6745cc

    '@filter_fa4a3fde7765':
      op: Filter
      parent: { node_ref: '@read_b5f228c91f16' }
      predicates:
        - { op: NotNull, arg: { op: Field, name: species, ... } }

    '@aggregate_eb3109707390':
      op: Aggregate
      parent: { node_ref: '@filter_fa4a3fde7765' }
      by:
        species: { op: Field, name: species, ... }
      metrics:
        avg_bill_length:
          op: Mean
          arg: { op: Field, name: bill_length_mm, ... }

    '@cachednode_fa2122f6a9e9':
      op: CachedNode
      parent: { node_ref: '@aggregate_eb3109707390' }
      cache:
        type: ParquetCache
        relative_path: parquet
      schema_ref: schema_9271d5e9d443

expression:
  node_ref: '@cachednode_fa2122f6a9e9'
  schema_ref: { schema_ref: schema_9271d5e9d443 }
```

---

# The Tools

The manifest is the unit of executable context. The tools — catalog, run, serve
— are how agents and humans compose with it.

## Catalog

Once a build is published, agents discover it by alias or by hash; humans
browse it like git refs — or open the TUI to preview data, schema, lineage, and
git history side-by-side.

```bash
❯ xorq catalog list-aliases
penguins-agg

❯ xorq catalog list
fa2122f6a9e9
```

## Run

```bash
$ xorq run builds/fa2122f6a9e9 -o out.parquet
```
Additionally, you can serve an unbound expression over Arrow Flight. with `xorq
serve-*` commands.

---

# How is this different from…

- **dbt** versions SQL bound to a single warehouse. xorq versions Ibis expressions that run across engines, with the manifest itself as the unit of versioning and lineage.
- **Dagster / Airflow** orchestrate task DAGs with retry state and checkpoints. xorq has no task graph — expressions stream over Arrow; failures just rerun from the manifest, hitting caches where they exist.
- **DVC / MLflow** track data and experiments alongside the work. xorq versions the *recipe*: inputs are content-addressed, the artifact is the executable expression, and provides a git-native catalog.

---

# Learn more

- [Quickstart](https://docs.xorq.dev/getting_started/quickstart)
- [Why xorq?](https://docs.xorq.dev/#why-xorq)
- [Claude Code plugin](https://github.com/xorq-labs/claude-plugins)
- [Scikit-learn ](https://github.com/xorq-labs/xorq-template-sklearn)

---

Pre-1.0. Expect breaking changes with migration guides.
