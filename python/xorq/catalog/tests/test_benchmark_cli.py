"""Subprocess-based benchmarks for the catalog CLI.

Each benchmark spawns a fresh Python process so that import/startup time
is included in the measurement — this is the number that matters for
interactive use.
"""

import subprocess
import sys

import pytest

from xorq.catalog.catalog import Catalog
from xorq.catalog.tests.conftest import make_build_tgz


CATALOG_CMD = [sys.executable, "-m", "xorq.cli", "catalog"]


def run(args, catalog_path=None, check=True):
    cmd = CATALOG_CMD[:]
    if catalog_path is not None:
        cmd += ["--path", str(catalog_path)]
    cmd += args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog_path(tmp_path):
    repo = Catalog.init_repo_path(tmp_path / "catalog")
    return str(repo.working_dir)


@pytest.fixture
def catalog_path_with_entry(catalog_path, tmp_path):
    tgz = make_build_tgz(tmp_path, "bench-entry")
    run(["add", str(tgz)], catalog_path=catalog_path)
    return catalog_path, tgz.stem


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_benchmark_catalog_help(benchmark):
    benchmark(run, ["--help"])


@pytest.mark.benchmark
def test_benchmark_catalog_init(benchmark, tmp_path):
    paths = iter(tmp_path / f"cat-{i}" for i in range(10_000))

    def init():
        benchmark.extra_info["last_path"] = p = str(next(paths))
        run(["--path", p, "init"])

    benchmark(init)


@pytest.mark.benchmark
def test_benchmark_catalog_add(benchmark, catalog_path, tmp_path):
    tgzs = iter(make_build_tgz(tmp_path, f"e{i}") for i in range(10_000))

    def add():
        run(["add", str(next(tgzs))], catalog_path=catalog_path)

    benchmark(add)


@pytest.mark.benchmark
def test_benchmark_catalog_list(benchmark, catalog_path_with_entry):
    catalog_path, _ = catalog_path_with_entry
    benchmark(run, ["list"], catalog_path=catalog_path)


@pytest.mark.benchmark
def test_benchmark_catalog_info(benchmark, catalog_path_with_entry):
    catalog_path, _ = catalog_path_with_entry
    benchmark(run, ["info"], catalog_path=catalog_path)


@pytest.mark.benchmark
def test_benchmark_catalog_check(benchmark, catalog_path_with_entry):
    catalog_path, _ = catalog_path_with_entry
    benchmark(run, ["check"], catalog_path=catalog_path)
