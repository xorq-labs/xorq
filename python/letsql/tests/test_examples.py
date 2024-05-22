import pathlib
import runpy
import pytest

from letsql.common.caching import KEY_PREFIX

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


@pytest.mark.parametrize("script", scripts)
def test_script_execution(script):
    runpy.run_path(str(script))
