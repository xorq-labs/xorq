import pathlib

import runpy
import pytest
from pytest import param

import letsql


KEY_PREFIX = letsql.config.options.cache.key_prefix


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


@pytest.mark.parametrize(
    "script", [param(script, id=script.stem) for script in scripts]
)
def test_script_execution(script):
    runpy.run_path(str(script))
