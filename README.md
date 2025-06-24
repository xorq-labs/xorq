![Xorq Logo](https://raw.githubusercontent.com/xorq-labs/xorq/main/docs/images/Xorq_PrimaryLogo_RGB_Midnight.png)

![GitHub License](https://img.shields.io/github/license/xorq-labs/xorq)
![PyPI - Status](https://img.shields.io/pypi/status/xorq)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/xorq-labs/xorq/ci-test.yml)

Popular Python tools like pandas and Ibis make data exploration enjoyable—but
when it's time to build reliable ML pipelines across multiple engines, things
quickly become complex.

Here's the challenge we faced:

* **SQL engines** like Snowflake or DuckDB excel at heavy computation but often feel disconnected from Python workflows.
* **Python libraries** like pandas and scikit-learn are fantastic for complex transformations but struggle with scale.
* **Python UDFs** handle custom logic beautifully, yet orchestrating them across engines can get cumbersome.
* **Caching intermediate results** should save precious compute resources but isn't always automatic.
* **Automated column-level lineage** is crucial for reproducibility but usually an afterthought.
* **Fail-fast pipelines** should give feedback at compile time, not runtime—but current solutions rarely achieve this.

Stitching these elements together into a reliable pipeline? It's still painful.
Each step often speaks a different language, needs constant babysitting, and
quickly becomes fragile.

That's exactly why we built **Xorq**.

## How Xorq works

[Xorq Architecture](https://raw.githubusercontent.com/xorq-labs/xorq/main/docs/images/how-xorq-works.png)

Xorq lets you:

* **Write expressive, pandas-style transformations** without memory constraints.
* **Seamlessly move between SQL engines and Python** within a single declarative pipeline.
* **Automatically cache intermediate results**, so no computation is wasted.
* **Serialize entire pipelines** for reproducibility and CI/CD.

Xorq uses Apache Arrow for zero-copy data transfer and leverages Ibis and
DataFusion under the hood for efficient computation.

## Why Xorq?

We built Xorq because existing tools fall short:

* **Ibis** is great for SQL but struggles with Python UDFs and caching.
* **PySpark** is complex and heavyweight for many use cases, especially when you just need a simple pipeline.
* **Airflow** is powerful but overkill for many ML workflows, and it lacks native support for multiple engines.
* **Feast** provides feature management and serving but lacks batch transformations.
* **dbt** lets you compose SQL models but not Python functions.

Xorq’s key differentiators are:

* **Multi-engine workflows**: Combine Snowflake, DuckDB, and Python effortlessly.
* **Built-in caching**: No repeated expensive joins or wasted resources.
* **Serializable pipelines**: YAML and SQL artifacts for reproducibility and easy deployment.
* **Portable UDxFs**: Write your logic once and run it anywhere supported by DataFusion.

## Demo Time!

Let's see Xorq in action.

### Step 1: Install Xorq

```bash
pip install xorq  # Note: xorq is still pre-1.0!
```

### Step 2: Create a pipeline

```python
import xorq as xo
import xorq.expr.datatypes as dt

@xo.udf.make_pandas_udf(
    schema=xo.schema({"title": str, "url": str}),
    return_type=dt.bool,
    name="url_in_title",
)
def url_in_title(df):
    return df.apply(lambda s: (s.url or "") in (s.title or ""), axis=1)

con = xo.connect()
expr = (
    xo.deferred_read_parquet(con, "hn-data.parquet", "hn")
    .mutate(url_in_title=url_in_title.on_expr)
)

print(expr.execute().head())
```

### Step 3: Serialize your pipeline

```bash
xorq build pipeline.py -e expr --target-dir builds/
```
The CLI creates reproducible build artifacts:

```
builds/
└── fce90c2d4bb8/
    ├── expr.yaml
    ├── deferred_reads.yaml
    ├── *.sql
    └── metadata.json
```

Ship these anywhere—CI systems, teammates, or other engines—and results remain consistent.

## Current Limitations

We're upfront about what’s not quite there yet:

* **API Stability**: Xorq is rapidly evolving, and breaking changes are common until v1.0.
* **Single-Machine Only**: We don't have distributed scheduling yet—so Xorq currently scales vertically.
* **Documentation Gaps**: Docs are improving but still thin in areas.

### Out of Scope (for now)

* Real-time sources (Kafka, Pulsar, etc.)
* Rust-based UDFs
* R, Javascript, or other language support


## What’s Next?

We're genuinely excited about the upcoming features:

* **Engine plugins** for BigQuery and Polars.
* **Native lineage** tracking with OpenLineage.
* **Enhanced Jupyter integration**.

We'd love your feedback! Your ⭐, issues, and contributions help us shape Xorq's future.

## Getting Involved

Interested? Dive deeper:

* Read the [full article](https://docs.xorq.dev/blog/intro).
* Join the discussion on Slack: [#xorq](link).
* Contribute via [GitHub](https://github.com/xorq-labs/xorq).

## Installation Requirements

```bash
pip install xorq  # or pip install "xorq[examples]"
```
* Python 3.9+
* Apache Arrow 10.0+

## License & Acknowledgements

Xorq is licensed under [Apache 2.0](https://github.com/xorq-labs/xorq/blob/main/LICENSE).

This project heavily relies on [Ibis](https://github.com/ibis-project/ibis) and [DataFusion](https://github.com/apache/datafusion).
