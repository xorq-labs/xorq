"""Subprocess-based import-time benchmarks.

Each benchmark spawns a fresh Python process so the measurement
captures real cold-start cost — the same latency a user or the CLI
pays on first import.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


MODULES = [
    "xorq",
    "xorq.cli",
    "xorq.ibis_yaml.packager",
    "xorq.internal",
    "xorq.common.utils.logging_utils",
    "xorq.config",
    "xorq.catalog.catalog",
    "xorq.backends.xorq_datafusion",
    "xorq.expr.datatypes",
    "xorq.common.utils.defer_utils",
    "xorq.expr.relations",
    "xorq.expr.api",
    "xorq.flight",
    "xorq.api",
    "xorq.backends.pyiceberg",
]


def _import_module(module: str) -> None:
    subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        check=True,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("module", MODULES)
def test_benchmark_import(benchmark: pytest.fixture, module: str) -> None:
    benchmark(_import_module, module)
