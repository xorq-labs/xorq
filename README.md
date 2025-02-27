# xorq: do-anything, run-anywhere pandas-style pipelines

[![Downloads](https://static.pepy.tech/badge/letsql)](https://pepy.tech/project/letsql)
![PyPI - Version](https://img.shields.io/pypi/v/letsql)
![GitHub License](https://img.shields.io/github/license/letsql/letsql)
![PyPI - Status](https://img.shields.io/pypi/status/letsql)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/letsql/letsql/ci-test.yml)
![Codecov](https://img.shields.io/codecov/c/github/letsql/letsql)

xorq is a deferred computation toolchain that brings the replicability and
performance of declarative pipelines to the Python ML ecosystem. It enables us
to write pandas-style transformations that never run out of memory,
automatically cache intermediate results, and seamlessly move between SQL
engines and Python UDFs—all while maintaining replicability. xorq is built on
top of Ibis and DataFusion.

| Feature | Description |
|---------|-------------|
| **Declarative expressions** | xorq lets you define transformations as Ibis expressions so that you are not tiedd to a specific execution engine. The `.into_backend()` method in xorq enables seamless transitions between engines within a single pipeline. |
| **[Built-in caching](https://docs.xorq.dev/core_concepts#caching-system)** | xorq automatically tracks the computational graph of your pipeline and caches intermediate results when `cache` operator is invoked, minimizing repeated work.  |
| **[Multi-engine](https://docs.xorq.dev/core_concepts#multi-engine-system)** | Create unified ML workflows that leverage the strengths of different data engines in a single pipeline. xorq orchestrates data movement between engines (e.g., Snowflake for initial extraction, DuckDB for transformations, and Python for ML model training). |
| **Serializable pipelines** | All pipeline definitions, including UDFs, are serialized to YAML format, enabling robust version control, reproducibility, and CI/CD integration. This serialization captures the complete execution graph, ensuring consistent results across environments and making it easy to track changes over time. |
| **Portable UDFs** | xorq support user-defined functions and its variants like aggregates, window functions, and transformations. The DataFusion based embedded engine provides a portable runtime for UDF execution. |
| **Arrow-native architecture** | Built on Apache Arrow's columnar memory format and Arrow Flight transport layer, xorq achieves high-performance data transfer without cumbersome serialization overhead. |


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


pg = xo.postgres.connect_env()
db = xo.duckdb.connect()

batting = pg.table("batting")
awards_players = xo.examples.awards_players.fetch(backend=db)

left = batting.filter(batting.yearID == 2015)

right = (awards_players.filter(awards_players.lgID == "NL")
                       .drop("yearID", "lgID")
                       .into_backend(pg, "filtered"))

expr = (left.join(right, ["playerID"], how="semi")
            .cache()
            .select(["yearID", "stint"]))

result = expr.execute()
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
More info can be found in the tutorial [Building with xorq](https://docs.xorq.dev/tutorials/build).

For more examples on how to use xorq, check the
[examples](https://github.com/letsql/xorq/tree/main/examples) directory, note
that in order to run some of the scripts in there, you need to install the
library with `examples` extra:

```shell
pip install 'xorq[examples]'
```

## Contributing

Contributions are welcome and highly appreciated. To get started, check out the [contributing guidelines](https://github.com/letsql/xorq/blob/main/CONTRIBUTING.md).

## Acknowledgements

This project heavily relies on [Ibis](https://github.com/ibis-project/ibis) and [DataFusion](https://github.com/apache/datafusion).   

## License

This repository is licensed under the [Apache License](https://github.com/letsql/xorq/blob/main/LICENSE)
