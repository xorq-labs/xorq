<div align="center">

![Xorq Logo](docs/images/Xorq_WordMark_RGB_Midnight.png#gh-light-mode-only)
![Xorq Logo](docs/images/Xorq_WordMark_RGB_BlueSky.png#gh-dark-mode-only)

![License](https://img.shields.io/github/license/xorq-labs/xorq)
![PyPI - Version](https://img.shields.io/pypi/v/xorq)
![CI Status](https://img.shields.io/github/actions/workflow/status/xorq-labs/xorq/ci-test.yml)


[Documentation](https://docs.xorq.dev) • [Website](https://www.xorq.dev)
</div>

---
Xorq is an executable, composable catalog for tabular data work. It gives
agents runnable, content-addressed pipelines instead of markdown notes: memory
you can execute, not prose you have to reread. Entries compose, so new
pipelines build on cataloged ones instead of starting from scratch. It turns
ephemeral agent work (pandas scripts, sklearn pipelines, ad-hoc tables) into
durable artifacts that any future agent or human can discover, reproduce and
reuse. The catalog is a ledger: an append-only record where each entry is named
by what it computes and carries its inputs and schema.

It comes with a CLI for agents and a TUI for humans with a git-native catalog.
Below, a catalog of ten entries for an insurance pricing pipeline. The selected
entry is the scored test set: its lineage runs through three fitted models down
to the two source reads, next to its schema and a git log with one commit per
entry.

![xorq catalog TUI: ten entries on the left; the lineage of the scored test set down to the two source reads, its schema, and the git log with one commit per entry](docs/images/ledger-tui.png)

A fitted model is an entry too. The classifier below is a one-column table
named after the features and target it was fit on, with its training slice as
its lineage.

![The claim-model entry in the xorq catalog TUI: a one-column table whose lineage is the fit slice, the split, the join and the two source reads](docs/images/ledger-claim-model.png)

## The Problem

Coding agents finish closed-loop tasks and leave tech debt behind. Ask one for
a dashboard and you get a folder of one-off scripts that import each other, a
JSON file of intermediate state, and a `requirements.txt` last regenerated two
sessions ago. It runs on your laptop. Reproducing it on another machine or
putting it in production means rewriting it, and every rewrite adds complexity.

| Pain | Symptom |
|------|---------|
| **Imperative, stateful artifacts** | An agent run leaves you with a folder of `.py`, `.json`, and `.html` files. Reproducing the result means re-running them in the right order without a declarative spec |
| **No discoverable, shared index** | "Team memory" today is `~/.claude/memory/*.md`, with a `MEMORY.md` index of one-liners pointing to the notes. There's no executable catalog two agents can both pull into context |
| **No lineage graph** | Rename a column upstream and a downstream model breaks at runtime. The dependency lived only in chat history, not in a graph that could have flagged it before it shipped. |
| **No portable environment** | A pipeline that ran in one agent session has no path to another sandbox, your machine, or production.|

## Benchmark

On [DABStep](https://huggingface.co/spaces/adyen/DABstep), 450 data-analysis
questions over payment transaction data, a Xorq semantic catalog of 33 named
expressions takes Haiku from 50% to 84%, 9 points above the Sonnet baseline.

![DABStep accuracy: Haiku 4.5 50%, Sonnet 4.6 75%, Haiku 4.5 + Semantic Catalog
84%](docs/images/dabstep-benchmark.png)

Where the agent looks for context mattered more than which base model it
used. Full write-up:
[Orientation Over Reasoning](https://xorq.dev/blog/orientation-over-reasoning/).

A second experiment ran one metric question 900 times with Haiku. Deriving the
definition from the source files, the agent was never right. Selecting it from
a reviewed catalog entry, it was always right, in a fifth of the turns at a
fifth of the cost.

| | Derived from source files | Selected from a reviewed entry |
|---|---|---|
| Correct | 0 of 100 | 100 of 100 |
| Median turns | 21 | 4 |
| Cost per run | $0.09 | $0.02 |

Full write-up:
[Faithful, Reproducible, Wrong](https://xorq.dev/blog/faithful-reproducible-wrong/).

## Getting started

Install the library and scaffold a project:

```bash
❯ pip install "xorq[examples]"
❯ xorq init -t penguins
```

The walkthrough below runs inside the project this creates.

## Xorq in 60 seconds

Think of Xorq as giving a dataframe pipeline a **reproducible identity**.

### 1. Define the pipeline

You start with a lazy dataframe expression:

```python
import xorq.api as xo

penguins = xo.examples.penguins.fetch()
expr = (
    penguins
    .filter(xo._.species.notnull())
    .group_by("species")
    .agg(avg_bill_length=xo._.bill_length_mm.mean())
)
```

This describes the transformation: filter the penguins data, group it by
species, and calculate the average bill length. Unlike a sequence of scripts
that must be executed in the right order, the expression gives Xorq a
structured representation of the computation.

### 2. Build it

Save it as the project's `expr.py`, then build it into a reproducible artifact:

```bash
xorq uv build expr.py
```

Xorq produces a hashed build directory:

```text
builds/b45d9db8189a/
├── build_metadata.json
├── expr.yaml
├── expr_metadata.json
├── profiles.yaml
├── requirements.txt
└── xorq_template_penguins-0.1.1-py3-none-any.whl
```

The manifest (`expr.yaml` + `*_metadata.json`) is the content-addressed
specification of the pipeline. It captures the computation, including its
operations, schema, lineage, and inputs. The build packages that
specification with the project's wheel and pinned requirements, the
environment needed to reproduce and execute it. The directory is named after
the hash of the expression's root node.

### 3. Add it to a catalog

Add the build to a catalog:

```bash
xorq catalog add builds/b45d9db8189a/ -a penguins-agg
```

The catalog is a Git repository of these entries. An entry can be discovered
by its hash or by a human-readable alias such as `penguins-agg`.

The flow is:

```text
Dataframe expression
        ↓
    xorq uv build
        ↓
Content-addressed specification
        +
Reproducible environment
        ↓
   Catalog entry
        ↓
     Git repo
        ↓
 Humans / CI / agents
```

Another human or agent can now discover, reproduce, run, or compose that
pipeline without reconstructing the original work from scripts, environment
setup, and chat history.

That is the core idea behind Xorq's executable catalog: instead of remembering
a description of what happened, the catalog keeps the computation itself.


## Design choices

| Choice | What it enables |
|--------|-----------------|
| **[Ibis](https://ibis-project.org/) as expression system** | Declarative dataframe expressions that compile to many engines. |
| **[Git](https://git-scm.com/) for state and storage** | The catalog is a git repo of entries with git-annex support for large files  |
| **[uv](https://docs.astral.sh/uv/) for reproducible environments** | Each entry ships with a wheel and pinned `requirements.txt`. |
| **[DataFusion](https://datafusion.apache.org/) for embedded compute** | In-process SQL and UDF execution, no cluster to stand up |
| **[Arrow](https://arrow.apache.org) for IPC and network** | Operators exchange Arrow RecordBatches |

## Supported engines

The same expression can run against any of these backends, and `into_backend`
moves data between them.

| Category | Engines |
|----------|---------|
| **Embedded** | DataFusion, DuckDB, SQLite, pandas |
| **Warehouses** | Snowflake, Databricks, BigQuery, Trino, Postgres |
| **Lakehouse** | PyIceberg |
| **Arrow Flight** | GizmoSQL |

## Comparison

Xorq is complementary to general-purpose data catalogs. Atlan, Collibra, and
Unity Catalog inventory and govern the assets you already have: descriptions,
owners, glossaries, policies, permissions, harvested lineage. Xorq catalogs the
computation, so an entry is something you run, not something you read about.

| Dimension | Xorq | Atlan, Collibra, Unity Catalog |
|-----------|------|--------------------------------|
| Cataloged unit | Content-addressed expression plus pinned environment | Metadata record over an existing table |
| Result of a lookup | Arrow stream from running the entry | A governed table you query with external code |
| Lineage | Read off the manifest, exact by construction | Harvested from executed queries, best effort |
| Operating model | Git repo of build artifacts on disk, no service to call | Hosted metastore and control plane |
| Portability | Same expression runs on DataFusion, DuckDB, Snowflake, Databricks | Bound to the host platform |
| Primary contributors | Agents and engineers composing new pipelines | Stewards, analysts, compliance teams |
| Out of scope | Glossaries, ownership workflows, policy, access control | Packaging the pipeline and its environment |

A governance catalog tells you what data exists, who owns it, and who may read
it. Xorq tells an agent what to run to get the same answer twice and humans how
it arrived at that answer. Xorq reads governed tables through the Databricks
and Snowflake backends, and its entries can be registered there as assets.

## Under the hood

<details open>
<summary><b>The Expression</b>: declarative Ibis, multi-engine, Arrow-native</summary>

Write declarative Ibis expressions that run like a tool. Xorq extends Ibis with
caching, multi-engine execution, and UDFs. Below, `xo._` is the Ibis row
reference, so `xo._.species` refers to the `species` column of the current table.

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

### One expression, many engines

```python
expr = penguins.into_backend(xo.sqlite.connect())
expr.ls.backends
```
```
(<xorq.backends.sqlite.Backend at 0x107debda0>,
 <xorq.backends.xorq_datafusion.Backend at 0x1669002c0>)
```

### Expressions are tools, Arrow is the pipe

Unix pipes text streams between small programs. Xorq pipes Arrow streams
between expressions, with no DAG of tasks to checkpoint: just data flowing
through transforms.

`unix : programs :: xorq : arrow-transforms`

```
In [6]: expr.to_pyarrow_batches()
Out[6]: <pyarrow.lib.RecordBatchReader at 0x15dc3f570>
```

### Scikit-learn pipelines

Xorq translates `scikit-learn` Pipeline objects to deferred expressions via
`Pipeline.from_instance(sklearn_pipeline)`. End-to-end sklearn examples live in
[xorq-labs/xorq-gallery](https://github.com/xorq-labs/xorq-gallery).

</details>

<details>
<summary><b>The Catalog</b>: a git repo of build artifacts on the filesystem</summary>

The catalog is a git repo of build artifacts on the filesystem. `xorq catalog
add` packages a build directory (the manifest plus the Python environment
pinned by `uv`) into an entry.

### Build and add

```bash
❯ xorq uv build expr.py
Building wheel...
Successfully built .../xorq_template_penguins-0.1.1-py3-none-any.whl
builds/b45d9db8189a

❯ xorq catalog -p git-catalogs/penguins init
Initialized catalog at .../git-catalogs/penguins

❯ xorq catalog -p git-catalogs/penguins add builds/b45d9db8189a/ -a penguins-agg
Added b45d9db8189a
```

### Git history

Every catalog operation is a commit you can read:

```
❯ git -C git-catalogs/penguins reflog
81a694a HEAD@{0}: add: b45d9db8189a (aliases penguins-agg)
e00f12c HEAD@{1}: add catalog.yaml
ead7911 HEAD@{2}: commit: Switching to main
```

### Catalog layout

```
❯ tree git-catalogs/penguins
git-catalogs/penguins
├── aliases
│   └── penguins-agg.zip -> ../entries/b45d9db8189a.zip
├── catalog.yaml
├── entries
│   └── b45d9db8189a.zip
└── metadata
    └── b45d9db8189a.zip.metadata.yaml
```

Aliases are symlinks, entries are zipped builds, and metadata sidecars are
plain YAML. An agent that clones the repo can discover everything with file
operations: no service to call, no API to learn.

```bash
# List aliased entries
❯ ls git-catalogs/penguins/aliases/

# Find entries that emit an 'avg_bill_length' column
❯ grep -l 'avg_bill_length' git-catalogs/penguins/metadata/*.yaml

# Find entries running on DataFusion
❯ grep -l 'xorq_datafusion' git-catalogs/penguins/metadata/*.yaml

# Find source entries (other kinds: expr, unbound_expr, expr_builder)
❯ grep -l 'kind: source' git-catalogs/penguins/metadata/*.yaml
```

### Inside an entry

A build directory contains the manifest plus everything needed to reproduce
it. The zipped build is the entry stored in the catalog.

```
❯ tree builds/b45d9db8189a
builds/b45d9db8189a
├── build_metadata.json
├── expr.yaml
├── expr_metadata.json
├── profiles.yaml
├── requirements.txt
└── xorq_template_penguins-0.1.1-py3-none-any.whl
```

The manifest (`expr.yaml` + `*_metadata.json`) is the content-addressed
specification of the pipeline. The **entry** packages it with deps and source
for reproducible execution.

```yaml
# Input-addressed, composable, portable
# Abridged expr.yaml
definitions:
  nodes:
    '@read_dbd8d81198f1':
      op: Read
      method_name: read_parquet
      name: penguins
      read_kwargs:
        - [hash_path, .../penguins/20250703T145709Z-c3cde/penguins.parquet]
        - [table_name, penguins]
      schema_ref: schema_11039d461f5b

    '@filter_f7eedc0c6f85':
      op: Filter
      parent: { node_ref: '@read_dbd8d81198f1' }
      predicates:
        - { op: NotNull, arg: { op: Field, name: species, ... } }

    '@aggregate_b45d9db8189a':
      op: Aggregate
      parent: { node_ref: '@filter_f7eedc0c6f85' }
      by:
        species: { op: Field, name: species, ... }
      metrics:
        avg_bill_length:
          op: Mean
          arg: { op: Field, name: bill_length_mm, ... }

expression:
  node_ref: '@aggregate_b45d9db8189a'
  schema_ref: { schema_ref: schema_cdf21e46eb8a }
```

</details>

<details>
<summary><b>The Tools</b>: catalog, run, serve</summary>

The entry is the unit of the catalog: the manifest plus the environment to run
it. The tools (catalog, run, serve) are how agents and humans compose with
it.

### Catalog

Once an entry is published, agents discover it straight from the catalog
filesystem: `metadata/*.yaml` sidecars sit next to the zipped entries, so
listing, filtering, and lookup-by-alias/hash all work with plain file reads
and `git` (no service required). Humans open the TUI to preview data,
schema, lineage, and git history side-by-side.

```bash
❯ xorq catalog list-aliases
penguins-agg

❯ xorq catalog list
b45d9db8189a
```

### Run

```bash
❯ xorq run builds/b45d9db8189a -o out.parquet
```
You can also serve an unbound expression over Arrow Flight with `xorq
serve-unbound`, or a UDXF with `xorq serve-flight-udxf`.

</details>


---

## Learn more

- [Quickstart](https://docs.xorq.dev/getting_started/quickstart)
- [Why Xorq?](https://docs.xorq.dev/#why-xorq)
- [Scikit-learn template](https://github.com/xorq-labs/xorq-template-sklearn)
- [A Git-Native Semantic Layer](https://xorq.dev/blog/bsl-xorq/): building a portable semantic catalog with Xorq
- [Orientation Over Reasoning](https://xorq.dev/blog/orientation-over-reasoning/): Haiku with a Xorq catalog hits 84% on DABStep, above the Sonnet baseline
- [Faithful, Reproducible, Wrong](https://xorq.dev/blog/faithful-reproducible-wrong/): one metric question, 900 runs, and why a reviewed entry beats a bigger model

---

Pre-1.0. Expect breaking changes with migration guides.
