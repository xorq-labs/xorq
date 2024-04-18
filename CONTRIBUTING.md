## Contributing Guide

### Setting up a development environment

This assumes that you have rust and cargo installed. We use the workflow recommended by [pyo3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).

```bash
# fetch this repo
git clone git@github.com:letsql/letsql.git
# prepare development environment (used to build wheel / install in development)
python3 -m venv venv
# activate the venv
source venv/bin/activate
# update pip itself if necessary
python -m pip install -U pip
# install dependencies (for Python 3.8+)
python -m pip install -r requirements-dev.txt
# set up the git hook scripts
pre-commit install
```

### Running the test suite
Install the [just](https://github.com/casey/just#installation) command runner, if needed.
Download example data to run the tests successfully.

```bash
just download-data
```

To test the code:
```bash
# make sure you activate the venv using "source venv/bin/activate" first
maturin develop
just up postgres # some of the tests use postgres
python -m pytest # or pytest
```