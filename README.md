<div align="center">

![Xorq Logo](docs/images/Xorq_WordMark_RGB_Midnight.png#gh-light-mode-only)
![Xorq Logo](docs/images/Xorq_WordMark_RGB_BlueSky.png#gh-dark-mode-only)

![License](https://img.shields.io/github/license/xorq-labs/xorq)
![PyPI - Version](https://img.shields.io/pypi/v/xorq)
![CI Status](https://img.shields.io/github/actions/workflow/status/xorq-labs/xorq/ci-test.yml)

</div>

> **Xorq is a multi‑engine batch transformation framework built on Ibis,
> DataFusion and Arrow.**
> It ships a compute catalog and a multi-engine manifest you can run
> across DuckDB, Snowflake, DataFusion, and more.

---

## What Xorq gives you

- **Multi-engine manifest:** A single, typed plan captured as a YAML artifact
that can execute in DuckDB, Snowflake, DataFusion, etc.
- **Deterministic builds & caching:** Content hashes of the plan power
reproducible runs and cheap replays.
- **Lineage & Schemas:** Compile-time schema checks and end-to-end to end
column-level lineage.
- **Compute catalog:** Versioned registry that stores and operates on manifests
(run, cache, diff, serve-unbound).
- **Portable UDxFs:** Arbitrary python logic with schema-in/out contracts
portable via Arrow Flight.
- **Scikit-learn integration:** Model fitting pipeline captured in the predict
pipeline manifest for portable batch scoring and model training lineage

> **Not an orchestrator.** Use Xorq from Airflow, Dagster, GitHub Actions, etc.

> **Not streaming/online.** Xorq focuses on **batch**,**out-of-core**
> transformations.


## Quickstart

```bash
pip install xorq[examples]
xorq init -t penguins
```

Then follow the [Quickstart
Tutorial](https://docs.xorq.dev/tutorials/getting_started/quickstart) for a
full walk-through using the Penguins dataset.

## From `scikit-learn` to multi-engine manifest

The manifest is a collection of YAML files that captures the expression graph
and supporting files like memtables serialized to disk.

Once you xorq build your pipeline, you get:

- expr.yaml: a reproducible expression graph
- deferred_reads.yaml: source metadata
- SQL and metadata files for inspection and CI

Xorq makes it easy to bring your scikit-learn Pipeline and automatically
converts it into a deferred Xorq expression.

```python
import xorq.api as xo
from xorq.expr.ml.pipeline_lib import Pipeline


(train, test) = xo.test_train_splits(...)
sklearn_pipeline = make_pipeline(...)
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
# still no work done: deferred fit expression
fitted_pipeline = xorq_pipeline.fit(train, features=features, target=target)
expr = fitted_pipeline.predict(test[features])
```

Here's a commented snippet from a YAML manifest

```bash
predicted:
  op: ExprScalarUDF            # predict(...)
  kwargs:
    bill_length_mm: ...        # features
    bill_depth_mm: ...
    flipper_length_mm: ...
    body_mass_g: ...
  meta:
    __config__:
      computed_kwargs_expr:
        op: AggUDF             # fit(...)
        kwargs:
          bill_length_mm: ...
          bill_depth_mm: ...
          flipper_length_mm: ...
          body_mass_g: ...
          species: ...         # target
```
The YAML format serializes the Expression graph and all its nodes, including
UDFs as pickled entries.

## From manifest to catalog

Once an expression is built, we can then catalog it and share across teams.

The compute catalog is a versioned registry of compute manifests. It can be
stored in Git, S3, GCS, or a database.

```bash
❯ xorq catalog add builds/{build-hash} --alias penguins-model
```

```
❯ xorq catalog ls
Aliases:
mortgage-test-predicted dbf90860-88b3-4b6c-830a-8518b3296e7c    r1
Entries:
dbf90860-88b3-4b6c-830a-8518b3296e7c    r1      52f987594254
```

You can then run, serve or cache the catalog entry, including unbinding nodes
that depend on external state (e.g. source tables). This is useful to serve a
trained pipeline with new data.

### Serve the same expression with new inputs (serve-unbound)

We can rerun an expression with new inputs by replacing an arbitrary node in
the expression defined by its node-hash.

```bash
xorq serve-unbound builds/7061dd65ff3c --host localhost --port 8001 --cache-dir penguins_example b2370a29c19df8e1e639c63252dacd0e
```
- `builds/7061dd65ff3c`: Your built expression manifest
- `--host localhost --port 8001`: Where to serve the UDxF from
- `--cache-dir penguins_example`: Directory for caching results
- `b2370a29c19df8e1e639c63252dacd0e`: The node-hash that represents the expression input to replace

To learn more on how to find the node hash, check out the [Serve Unbound](https://docs.xorq.dev/tutorials/getting_started/quickstart#finding-the-node-hash).

### Compose with the served expression:

```python
import xorq.api as xo

client = xo.flight.connect("localhost", 8001)
f = client.get_exchange("default") # currently all expressions get the default name in addition to their hash

new_expr = expr.pipe(f)

new_expr.execute()
```

## How Xorq works

Xorq uses Apache Arrow Flight RPC for zero-copy data transfer and leverages Ibis and
DataFusion under the hood for efficient computation.

![Xorq Architecture](docs/images/how-xorq-works-2.png)

## Use cases

A generic catalog that can be used to build new workloads:

- Lineage‑preserving, multi-engine feature stores (offline, reproducible)
- Composable data products (ship datasets as compute artifacts)
- Governed sharing of compute (catalog entries as the contract between teams)
- ML/data pipeline development (deterministic builds)


Also great for:

- Generating SQL from high-level DSLs (e.g. Semantic Layers)
- Batch model scoring across engines (same expr, different backends)
- Cross‑warehouse migrations (portability via Ibis + UDxFs)
- Data CI (compile‑time schema/lineage checks in PRs)


## Learn More

* [Why Xorq?](https://docs.xorq.dev/#why-xorq)
* [Caching Guide](https://docs.xorq.dev/core_concepts/caching)
* [Backend Profiles](https://docs.xorq.dev/api_reference/backend_configuration/profiles_api)
* [Scikit-learn Template](https://github.com/xorq-labs/xorq-template-sklearn)

## Status

Xorq is pre-1.0 and evolving fast. Expect breaking changes.

## Get Involved

* [Website](https://www.xorq.dev)
* [Discord](https://discord.gg/8Kma9DhcJG)
* [Contribute on GitHub](https://github.com/xorq-labs/xorq)
