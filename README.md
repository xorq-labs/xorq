# xorq: Multi-engine ML pipelines made simple

[![PyPI Downloads](https://static.pepy.tech/badge/xorq)](https://pepy.tech/projects/xorq)
![PyPI - Version](https://img.shields.io/pypi/v/xorq)
![GitHub License](https://img.shields.io/github/license/xorq-labs/xorq)
![PyPI - Status](https://img.shields.io/pypi/status/xorq)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/xorq-labs/xorq/ci-test.yml)
![Codecov](https://img.shields.io/codecov/c/github/xorq-labs/xorq)

xorq is a deferred computational framework that brings the replicability and
performance of declarative pipelines to the Python ML ecosystem. It enables us
to write pandas-style transformations that never run out of memory,
automatically cache intermediate results, and seamlessly move between SQL
engines and Python UDFs—all while maintaining replicability. xorq is built on
top of Ibis and DataFusion.

| Feature                                                                       | Description |
|-------------------------------------------------------------------------------|-------------|
| **Declarative expressions**                                                   | Express and execute complex data processing logic via declarative functions. Define transformations as Ibis expressions so that you are not tied to a specific execution engine. |
| **[Multi-engine](https://docs.xorq.dev/core_concepts#multi-engine-system)**   | Create unified ML workflows that leverage the strengths of different data engines in a single pipeline. xorq orchestrates data movement between engines (e.g., Snowflake for initial extraction, DuckDB for transformations, and Python for ML model training). |
| **[Built-in caching](https://docs.xorq.dev/core_concepts#caching-system)**    | xorq automatically caches intermediate pipeline results, minimizing repeated work. |
| **Serializable pipelines**                                                    | All pipeline definitions, including UDFs, are serialized to YAML, enabling version control, reproducibility, and CI/CD integration. Ensures consistent results across environments and makes it easy to track changes over time. |
| **[Portable UDFs](https://docs.xorq.dev/core_concepts#custom-ud-x-f-system)** | Build pipelines as  UDxFs- aggregates, windows, and transformations. The DataFusion-based xorq engine provides a portable runtime for UDF execution. |
| **Arrow-native architecture**                                                 | Built on the Apache Arrow columnar memory format and Arrow Flight transport layer, xorq achieves high-performance data transfer without cumbersome serialization overhead. |


## Getting Started
xorq functions as both an interactive library for building expressions and a
command-line interface. This dual nature enables seamless transition
from exploratory research to production-ready artifacts. The steps below will
guide through using both the CLI and library components to get started.

> [!CAUTION] 
> This library does not currently have a stable release. Both the
> API and implementation are subject to change, and future updates may not be
> backward compatible.

### Installation

xorq is available as [`xorq`](https://pypi.org/project/xorq/) on PyPI:

```shell
pip install xorq
```

> [!NOTE]
> We are changing the name from LETSQL to xorq.

### Usage

```python
# your_pipeline.py
import xorq as xo
import xorq.expr.datatypes as dt

@xo.udf.make_pandas_udf(
    schema=xo.schema({"title": str, "url": str}),
    return_type=dt.bool,
    name="url_in_title",
)
def url_in_title(df):
    return df.apply(
        lambda s: (s.url or "") in (s.title or ""),
        axis=1,
    )

# Connect to xorq's embedded engine
con = xo.connect()

# Reference to the parquet file
name = "hn-data-small.parquet"

expr = xo.deferred_read_parquet(
    con,
    xo.options.pins.get_path(name),
    name,
).mutate(**{"url_in_title": url_in_title.on_expr})

expr.execute().head()
```

xorq provides a CLI that enables you to build serialized artifacts from expressions, making your pipelines reproducible and deployable:

```shell
# Build an expression from a Python script
xorq build your_pipeline.py -e "expr" --target-dir builds
```
This will create a build artifact directory named by its expression hash:
```
builds
└── fce90c2d4bb8
   ├── abe2c934f4fe.sql
   ├── cec2eb9706bc.sql
   ├── deferred_reads.yaml
   ├── expr.yaml
   ├── metadata.json
   ├── profiles.yaml
   └── sql.yaml
```

The CLI converts Ibis expressions into serialized artifacts that capture the complete execution graph, ensuring consistent results across environments.
More info can be found in the tutorial [Building with xorq](https://docs.xorq.dev/core_concepts/build).

For more examples on how to use xorq, check the
[examples](https://github.com/xorq-labs/xorq/tree/main/examples) directory, note
that in order to run some of the scripts in there, you need to install the
library with `examples` extra:

```shell
pip install 'xorq[examples]'
```

## Contributing

Contributions are welcome and highly appreciated. To get started, check out the [contributing guidelines](https://github.com/xorq-labs/xorq/blob/main/CONTRIBUTING.md).

## Acknowledgements

This project heavily relies on [Ibis](https://github.com/ibis-project/ibis) and [DataFusion](https://github.com/apache/datafusion).   

## License

This repository is licensed under the [Apache License](https://github.com/xorq-labs/xorq/blob/main/LICENSE)
