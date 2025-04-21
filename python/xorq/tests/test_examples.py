import pathlib
import runpy

import pytest
from pytest import param

import xorq as xo


KEY_PREFIX = xo.config.options.cache.key_prefix
LIBRARY_SCRIPTS = ("pandas_example",)
GCS_SCRIPTS = ("gcstorage_example",)
NON_TESTABLE = (
    "mcp_flight_server.py",
    "duckdb_flight_example.py",
    "complex_cached_expr.py",
)

file_path = pathlib.Path(__file__).absolute()
root = file_path.parent
examples_dir = file_path.parents[3] / "examples"
scripts = (p for p in examples_dir.glob("*.py") if p.name not in NON_TESTABLE)


def teardown_function():
    """Remove any generated parquet cache files"""
    for path in root.glob(f"{KEY_PREFIX}*"):
        path.unlink(missing_ok=True)

    for path in pathlib.Path.cwd().glob(f"{KEY_PREFIX}*"):
        path.unlink(missing_ok=True)


def maybe_library(name: str):
    return pytest.mark.library if name in LIBRARY_SCRIPTS else ()


def maybe_gcs(name: str):
    return pytest.mark.gcs if name in GCS_SCRIPTS else ()


def maybe_marks(name: str):
    fs = (maybe_library, maybe_gcs)
    return tuple(filter(None, (f(name) for f in fs)))


@pytest.mark.parametrize(
    "script",
    [
        param(script, id=script.stem, marks=maybe_marks(script.stem))
        for script in scripts
    ],
)
def test_script_execution(script):
    dct = runpy.run_path(str(script), run_name="__pytest_main__")
    assert dct.get("pytest_examples_passed")
