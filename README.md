# LETSQL

[![Downloads](https://static.pepy.tech/badge/letsql)](https://pepy.tech/project/letsql)
![PyPI - Version](https://img.shields.io/pypi/v/letsql)
![GitHub License](https://img.shields.io/github/license/letsql/letsql)
![PyPI - Status](https://img.shields.io/pypi/status/letsql)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/letsql/letsql/ci-test.yml)
![Codecov](https://img.shields.io/codecov/c/github/letsql/letsql)

Data processing library built on top of **Ibis** and **DataFusion** to write multi-engine data workflows.

## Getting Started

### Installation

LETSQL is available as [`letsql`](https://pypi.org/project/letsql/) on PyPI:

```shell
pip install letsql
```

### Usage

```python
import letsql as ls

con = ls.connect()

iris = con.read_csv("data/iris.csv", "iris")

res = (
    iris.filter([iris.sepal_length > 5])
    .group_by("species")
    .agg(iris.sepal_width.sum())
    .execute()
)
```

## Contributing

Contributions are welcome and highly appreciated. To get started, check out the [contributing guidelines](https://github.com/letsql/letsql/blob/main/CONTRIBUTING.md).

## Support

If you have any issues with this repository, please don't hesitate to [raise them](https://github.com/letsql/letsql/issues/new).
It is actively maintained, and we will do our best to help you.

## Acknowledgements

This project heavily relies on [Ibis](https://github.com/ibis-project/ibis) and [DataFusion](https://github.com/apache/datafusion).   

## Liked the work?

If you've found this repository helpful, why not give it a star? It's an easy way to show your appreciation and support for the project.
Plus, it helps others discover it too!

## License

This repository is licensed under the [Apache License](https://github.com/letsql/letsql/blob/main/LICENSE)
