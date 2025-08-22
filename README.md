<div align="center">

![Xorq Logo](docs/images/Xorq_WordMark_RGB_Midnight.png)
![License](https://img.shields.io/github/license/xorq-labs/xorq)
![PyPI - Version](https://img.shields.io/pypi/v/xorq)
![CI Status](https://img.shields.io/github/actions/workflow/status/xorq-labs/xorq/ci-test.yml)

</div>

> **✨ Xorq is an opinionated framework for cataloging, sharing, and shipping
> multi-engine compute as diffable artifacts for your data in flight. ✨**

Xorq helps teams build **declarative, reusable ML pipelines** across Python and
SQL engines like DuckDB, Snowflake, and DataFusion. It offers:

* 🧠 **Multi-engine, declarative expressions** using pandas-style syntax and Ibis.
* 📦 **Expression Format** for Python in YAML, enabling repeatable compute.
* ⚡ **Portable UDFs and UDAFs** with automatic serialization.
* 🔁 **Shift-left with caching** using expr hash for naming things.
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

Here is a sample (abbreviated) output:

```bash
❯ cat deferred_reads.yaml
reads:
 penguins-36877e5b81573dffe4e988965ce3950b:
   engine: pandas
   profile_name: 08f39a9ca2742d208a09d0ee9c7756c0_1
   relations:
   - penguins-36877e5b81573dffe4e988965ce3950b
   options:
     method_name: read_csv
     name: penguins
     read_kwargs:
     - source: /Users/hussainsultan/Library/Caches/pins-py/gs_d3037fb8920d01eb3b262ab08d52335c89ba62aa41299e5236f01807aa8b726d/penguins/20250206T212843Z-8f28a/penguins.csv
     - table_name: penguins
   sql_file: 8b5f90115b97.sql
and similarly expr.yaml (just a snippet):

predicted:
  op: ExprScalarUDF
  class_name: _predicted_e1d43fe620d0175d76276
  kwargs:
    op: dict
    bill_length_mm:
      node_ref: ecb7ceed7bab79d4e96ed0ce037f4dbd
    bill_depth_mm:
      node_ref: 26ca5f78d58daed6adf20dd2eba92d41
    flipper_length_mm:
      node_ref: 916dc998f8de70812099b2191256f4c1
    body_mass_g:
      node_ref: e094d235b0c1b297da5c194a5c4c331f
  meta:
    op: dict
    dtype:
      op: DataType
      type: String
      nullable:
        op: bool
        value: true
    __input_type__:
      op: InputType
      name: PYARROW
    __config__:
      op: dict
      computed_kwargs_expr:
        op: AggUDF
        class_name: _fit_predicted_e1d43fe620d0175d7
        kwargs:
          op: dict
          bill_length_mm:
            node_ref: ecb7ceed7bab79d4e96ed0ce037f4dbd
          bill_depth_mm:
            node_ref: 26ca5f78d58daed6adf20dd2eba92d41
          flipper_length_mm:
            node_ref: 916dc998f8de70812099b2191256f4c1
          body_mass_g:
            node_ref: e094d235b0c1b297da5c194a5c4c331f
          species:
            node_ref: a9fa43a2d8772c7eca4a7e2067107bfc
```
Please note that this is still in beta and the spec is subject to change.

## How Xorq works

![Xorq Architecture](docs/images/how-xorq-works.png)

Xorq uses Apache Arrow for zero-copy data transfer and leverages Ibis and
DataFusion under the hood for efficient computation.

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
