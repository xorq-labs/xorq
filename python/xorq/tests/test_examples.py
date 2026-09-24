import importlib.util
import pathlib
import runpy
import types

import fsspec.implementations.memory
import pytest
from pytest import param

import xorq.api as xo
from xorq.tests.test_cli import build_run_examples_expr_names


KEY_PREFIX = xo.config.options.cache.key_prefix
LIBRARY_SCRIPTS = (
    "pandas_example",
    "flight_dummy_exchanger",
)
GCS_SCRIPTS = ("gcstorage_example",)
S3_SCRIPTS = ("s3_content_store_catalog",)
NON_TESTABLE = (
    "mcp_flight_server.py",
    "duckdb_flight_example.py",
    "complex_cached_expr.py",
    "xorq_build_and_run.py",
    "weather_flight.py",
    "gizmosql_demo.py",
    "semantic_builder_example.py",
    "databricks_into_backend_example.py",
)
TESTED_IN_BUILD_AND_RUN = tuple(name for name, *_ in build_run_examples_expr_names)

file_path = pathlib.Path(__file__).absolute()
root = file_path.parent
examples_dir = file_path.parents[3] / "examples"
scripts = (
    p
    for p in examples_dir.glob("*.py")
    if p.name not in (NON_TESTABLE + TESTED_IN_BUILD_AND_RUN)
)


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


def maybe_s3(name: str) -> pytest.MarkDecorator | tuple[()]:
    return pytest.mark.s3 if name in S3_SCRIPTS else ()


def maybe_marks(name: str):
    fs = (maybe_library, maybe_gcs, maybe_s3)
    return tuple(filter(None, (f(name) for f in fs)))


@pytest.fixture(autouse=True)
def gcs_cache_stays_in_memory(script, monkeypatch):
    """Keep the GCS examples off the shared ``expr-cache`` bucket.

    CI authenticates as a read-only service account, so the example's cache
    writes cannot run against real GCS at all; locally they would mutate a
    bucket shared with every other developer, and a run that dies between the
    write and the drop strands an object under a key every later run reuses.

    Only ``gcloud_utils``' ``gcsfs`` reference is swapped, which is the single
    place ``GCStorage`` constructs its filesystem. The pins board is itself
    GCS-backed and keeps the real ``gcsfs`` it needs to download the example's
    input. ``monkeypatch.setattr`` raises if that attribute ever goes away, so
    a refactor cannot silently route these writes back to the real bucket.
    """
    if script.stem not in GCS_SCRIPTS:
        return
    if importlib.util.find_spec("gcsfs") is None:
        pytest.skip("gcsfs is not installed")
    monkeypatch.setattr(
        "xorq.common.utils.gcloud_utils.gcsfs",
        types.SimpleNamespace(
            GCSFileSystem=fsspec.implementations.memory.MemoryFileSystem
        ),
    )


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
