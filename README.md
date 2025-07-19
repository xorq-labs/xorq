<div align="center">

![Xorq Logo](docs/images/Xorq_WordMark_RGB_Midnight.png)
![License](https://img.shields.io/github/license/xorq-labs/xorq)
![PyPI - Version](https://img.shields.io/pypi/v/xorq)
![CI Status](https://img.shields.io/github/actions/workflow/status/xorq-labs/xorq/ci-test.yml)

</div>

> **✨ Xorq is an opinionated framework for cataloging composable compute
> expressions for your data in flight. ✨**

Xorq helps teams build **declarative, reusable ML pipelines** across Python and
SQL engines like DuckDB, Snowflake, and DataFusion. It offers:

* 🧠 **Multi-engine, declarative expressions** using pandas-style syntax and Ibis.
* 📦 **Expression Format** for Python in YAML, enabling repeatable compute.
* ⚡ **Portable UDFs and UDAFs** with automatic serialization.
* 🔁 **Cached, shift-left** with hash-based expression tokenization.
* 🔍 **Column-level lineage and observability** out of the box.

## 🔧 Quickstart

```bash
pip install xorq[examples]
xorq init -t penguins
```

Then follow the [Quickstart Tutorial](https://docs.xorq.dev/tutorials/getting_started/quickstart) for a full walk-through using the Penguins dataset.

## 🚀 Why Xorq?

ML pipelines are brittle, inconsistent, and hard to reuse. Xorq gives you:

| Pain                  | How Xorq Helps          |
| --------------------- | ----------------------- |
| Mixing pandas and SQL | Unified declarative API |
| Wasted computation    | Transparent caching     |
| Manual deployment     | Xorq serve any expr     |
| Debugging lineage     | Visual lineage trees    |
| Engine lock-in        | Portable UDxFs          |
| Repro issues          | Compile-time schema and relational integrity validation |

## 📸 Example Output

Once you `xorq build` your pipeline, you get:

* `expr.yaml`: a reproducible expression graph
* `deferred_reads.yaml`: source metadata
* SQL and metadata files for inspection and CI

## 📌 Learn More

* [Why Xorq?](https://docs.xorq.dev/intro/why_xorq)
* [Caching Guide](https://docs.xorq.dev/core_concepts/caching)
* [Profiles + Remote Backends](https://docs.xorq.dev/core_concepts/profiles_guide)
* [Scikit-learn Pipelines](examples/pipelines_example.py)

## 🧪 Status

Xorq is pre-1.0 and evolving fast. Expect breaking changes.

## 🤝 Get Involved

* [Website](https://www.xorq.dev)
* [Discord](https://discord.gg/8Kma9DhcJG)
* [Contribute on GitHub](https://github.com/xorq-labs/xorq)
