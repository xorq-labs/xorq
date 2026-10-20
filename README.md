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

Coding agents finish closed-loop tasks and leave tech debt behind: a folder of
one-off scripts that import each other, a JSON file of intermediate state, and
a `requirements.txt` last regenerated two sessions ago. Reproducing it on
another machine means rewriting it, and every rewrite adds complexity.

| Pain | Symptom |
|------|---------|
| **Imperative, stateful artifacts** | Reproducing a result means re-running `.py`, `.json`, and `.html` files in the right order, with no declarative spec |
| **No shared index** | Team memory is a folder of markdown notes. Two agents have no executable catalog to pull into context |
| **No lineage graph** | Rename a column upstream and a downstream model breaks at runtime. The dependency lived only in chat history |
| **No portable environment** | A pipeline that ran in one agent session has no path to another sandbox, your machine, or production |

## Benchmark

On [DABStep](https://huggingface.co/spaces/adyen/DABstep), 450 data-analysis
questions over payment transaction data, a Xorq semantic catalog of 33 named
expressions takes Haiku from 50% to 84%, 9 points above the Sonnet baseline
([Orientation Over Reasoning](https://xorq.dev/blog/orientation-over-reasoning/)).

![DABStep accuracy: Haiku 4.5 50%, Sonnet 4.6 75%, Haiku 4.5 + Semantic Catalog
84%](docs/images/dabstep-benchmark.png)

A second experiment ran one metric question 900 times with Haiku
([Faithful, Reproducible, Wrong](https://xorq.dev/blog/faithful-reproducible-wrong/)).
Where the agent looked for the definition mattered more than which model it
used:

| | Derived from source files | Selected from a reviewed entry |
|---|---|---|
| Correct | 0 of 100 | 100 of 100 |
| Median turns | 21 | 4 |
| Cost per run | $0.09 | $0.02 |

## Getting started

```bash
❯ pip install "xorq[examples]"
❯ xorq init -t penguins && cd xorq-template
❯ xorq uv build expr.py
builds/4c915297a711
❯ xorq catalog init
Initialized catalog at ~/.local/share/xorq/catalogs/default
❯ xorq catalog add builds/4c915297a711/ -a penguins-predict
Added 4c915297a711
❯ xorq uv run builds/4c915297a711 -o out.parquet
```

Build turns an expression into a directory named by its hash, holding the
manifest, the project wheel, and pinned requirements. Add puts that directory
in a git catalog under its hash and an alias. Run executes it in the
environment the entry pins.

## Design choices

| Choice | What it enables |
|--------|-----------------|
| **[Ibis](https://ibis-project.org/) as expression system** | Declarative dataframe expressions that compile to many engines |
| **[Git](https://git-scm.com/) for state and storage** | The catalog is a git repo of entries, with git-annex for large files |
| **[uv](https://docs.astral.sh/uv/) for reproducible environments** | Each entry ships with a wheel and pinned `requirements.txt` |
| **[DataFusion](https://datafusion.apache.org/) for embedded compute** | In-process SQL and UDF execution, no cluster to stand up |
| **[Arrow](https://arrow.apache.org) for IPC and network** | Operators exchange Arrow RecordBatches |

The same expression runs on DataFusion, DuckDB, SQLite, pandas, Snowflake,
Databricks, BigQuery, Trino, Postgres, PyIceberg, and GizmoSQL over Arrow
Flight. `into_backend` moves data between them.

## Comparison

Xorq is complementary to governance catalogs. Atlan, Collibra, and Unity
Catalog inventory the assets you already have: owners, glossaries, policies,
harvested lineage. Xorq catalogs the computation, so an entry is something you
run, not something you read about. It reads governed tables through the
Databricks and Snowflake backends, and its entries can be registered there as
assets.

| Dimension | Xorq | Atlan, Collibra, Unity Catalog |
|-----------|------|--------------------------------|
| Cataloged unit | Content-addressed expression plus pinned environment | Metadata record over an existing table |
| Result of a lookup | Arrow stream from running the entry | A governed table you query with external code |
| Lineage | Read off the manifest, exact by construction | Harvested from executed queries, best effort |
| Operating model | Git repo of build artifacts on disk, no service to call | Hosted metastore and control plane |
| Portability | Same expression runs on DataFusion, DuckDB, Snowflake, Databricks | Bound to the host platform |
| Out of scope | Glossaries, ownership workflows, policy, access control | Packaging the pipeline and its environment |

## Under the hood

<details open>
<summary><b>The Expression</b>: declarative Ibis, multi-engine, Arrow-native</summary>

Xorq extends Ibis with caching, multi-engine execution, and UDFs. `xo._` is
the Ibis row reference, so `xo._.species` is the `species` column of the
current table.

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

Call `.cache(ParquetCache.from_kwargs())` on any node to materialize it, and
`.into_backend(con)` to move its data to another engine:

```python
expr.into_backend(xo.sqlite.connect()).ls.backends
```
```
(<xorq.backends.sqlite.Backend at 0x107debda0>,
 <xorq.backends.xorq_datafusion.Backend at 0x1669002c0>)
```

Unix pipes text between small programs. Xorq pipes Arrow RecordBatches between
expressions, with no DAG of tasks to checkpoint:

```
In [6]: expr.to_pyarrow_batches()
Out[6]: <pyarrow.lib.RecordBatchReader at 0x15dc3f570>
```

`Pipeline.from_instance(sklearn_pipeline)` turns a scikit-learn pipeline into
a deferred expression. End-to-end examples live in
[xorq-labs/xorq-gallery](https://github.com/xorq-labs/xorq-gallery).

</details>

<details>
<summary><b>The Catalog</b>: a git repo of build artifacts on the filesystem</summary>

Building the expression above and adding it to a catalog:

```bash
❯ xorq uv build expr.py
builds/b45d9db8189a
❯ xorq catalog -p git-catalogs/penguins init
Initialized catalog at .../git-catalogs/penguins
❯ xorq catalog -p git-catalogs/penguins add builds/b45d9db8189a/ -a penguins-agg
Added b45d9db8189a
```

Every catalog operation is a commit you can read:

```
❯ git -C git-catalogs/penguins reflog
81a694a HEAD@{0}: add: b45d9db8189a (aliases penguins-agg)
e00f12c HEAD@{1}: add catalog.yaml
ead7911 HEAD@{2}: commit: Switching to main
```

Aliases are symlinks, entries are zipped builds, and metadata sidecars are
plain YAML, so an agent that clones the repo can discover everything with file
operations:

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

❯ grep -l 'avg_bill_length' git-catalogs/penguins/metadata/*.yaml  # entries that emit a column
❯ grep -l 'xorq_datafusion' git-catalogs/penguins/metadata/*.yaml  # entries that run on DataFusion
```

Humans get the same view in `xorq catalog tui`: data preview, schema, lineage,
and git history side by side. Agents use the CLI:

```bash
❯ xorq catalog -p git-catalogs/penguins list-aliases
penguins-agg
❯ xorq catalog -p git-catalogs/penguins list
b45d9db8189a
❯ xorq uv run builds/b45d9db8189a -o out.parquet
```

`xorq serve-unbound` serves an unbound expression over Arrow Flight, and
`xorq serve-flight-udxf` serves a UDXF.

</details>

<details>
<summary><b>The Entry</b>: a manifest plus the environment to run it</summary>

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
specification of the pipeline: its operations, schema, lineage, and inputs.
The directory is named after the hash of the expression's root node. The
wheel and `requirements.txt` pin the environment that runs it.

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
