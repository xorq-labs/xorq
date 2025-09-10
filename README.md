<div align="center">

![Xorq Logo](docs/images/Xorq_WordMark_RGB_Midnight.png)
![License](https://img.shields.io/github/license/xorq-labs/xorq)
![PyPI - Version](https://img.shields.io/pypi/v/xorq)
![CI Status](https://img.shields.io/github/actions/workflow/status/xorq-labs/xorq/ci-test.yml)

</div>

> **Xorq is a multi‑engine batch transformation framework.**
> It ships a **compute catalog**—versioned Python/Ibis expressions you can run across
> DuckDB, Snowflake, DataFusion, and more.

---

## What Xorq is

- 🧠 **Compute catalog:** A registry of declarative transformations as
  diffable, addressable manifest (`expr.yaml`).
- 🔁 **Deterministic builds & caching:** Content‑addressed (expr‑hash) naming
  for repeatable runs and cheap replays/backfills.
- 🧩 **Portable UDXFs:** User‑Defined (Aggregate) Functions serialized once,
  reused across engines.
- 🔬 **Lineage & schema checks:** Column‑level lineage and compile‑time
  relational integrity.
- 🤖 **Scikit‑learn integration:** Treat estimators/pipelines as compute—**fit** as an aggregate step, **predict** as a scalar step—parameters are serialized into the catalog for portable batch scoring.

> **Not an orchestrator.** Use Xorq from Airflow, Dagster, Prefect, GitHub
> Actions, etc.
> **Not streaming/online.** Xorq focuses on **batch** transformations.


## 🔧 Quickstart

```bash
pip install xorq[examples]
xorq init -t penguins
```

Then follow the [Quickstart Tutorial](https://docs.xorq.dev/tutorials/getting_started/quickstart) for a full walk-through using the Penguins dataset.

## Core concepts

- Expression Format: Python expressions captured as YAML (expr.yaml) for reproducible, engine‑portable compute.
- Deferred reads: Source metadata captured in deferred_reads.yaml.
- Profiles: Pluggable backends (e.g., DuckDB, Snowflake, DataFusion) selected at run time.
- UDxFs: User‑Defined Exchange Functions (UDF/UDAF) packaged for cross‑engine reuse.

## 📸 Expression Format for Compute

Xorq makes it easy to bring your scikit-learn Pipeline and automatically
converts it into a deferred Xorq expression.

```python
(train, test) = xo.test_train_split(...)
sklearn_pipeline = make_pipeline(...)
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
# still no work done: deferred fit expression
fitted_pipeline = xorq_pipeline.fit(train, features=features, target=target)
expr = test_predicted = fitted_pipeline.predict(test[features])
```

Once you `xorq build` your pipeline, you get:

* `expr.yaml`: a reproducible expression graph
* `deferred_reads.yaml`: source metadata
* SQL and metadata files for inspection and CI

Conceptual shape (what the catalog captures) when converting to a YAML manifest:

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
Please note that this is still in beta and the spec is subject to change.

## How Xorq works

Xorq uses Apache Arrow for zero-copy data transfer and leverages Ibis and
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


## 📌 Learn More

* [Why Xorq?](https://docs.xorq.dev/#why-xorq)
* [Caching Guide](https://docs.xorq.dev/core_concepts/caching)
* [Backend Profiles](https://docs.xorq.dev/api_reference/backend_configuration/profiles_api)
* [Scikit-learn Template](https://github.com/xorq-labs/xorq-template-sklearn)

## 🧪 Status

Xorq is pre-1.0 and evolving fast. Expect breaking changes.

## 🤝 Get Involved

* [Website](https://www.xorq.dev)
* [Discord](https://discord.gg/8Kma9DhcJG)
* [Contribute on GitHub](https://github.com/xorq-labs/xorq)
