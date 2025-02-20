import pathlib
import runpy

import pytest
from pytest import param

import xorq as xo


KEY_PREFIX = xo.config.options.cache.key_prefix
LIBRARY_SCRIPTS = ("pandas_example",)

file_path = pathlib.Path(__file__).absolute()
root = file_path.parent
examples_dir = file_path.parents[3] / "examples"
scripts = examples_dir.glob("*.py")


def teardown_function():
    """Remove any generated parquet cache files"""
    for path in root.glob(f"{KEY_PREFIX}*"):
        path.unlink(missing_ok=True)

    for path in pathlib.Path.cwd().glob(f"{KEY_PREFIX}*"):
        path.unlink(missing_ok=True)


def maybe_library(name: str):
    return pytest.mark.library if name in LIBRARY_SCRIPTS else ()


@pytest.mark.parametrize(
    "script",
    [
        param(script, id=script.stem, marks=maybe_library(script.stem))
        for script in scripts
    ],
)
def test_script_execution(script):
    runpy.run_path(str(script))
