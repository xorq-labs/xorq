"""Which ``-m`` expressions reach a backend's test directory.

``pytest_ignore_collect`` decides per backend directory before any test in it
is imported, so it can only see backend names; every other marker is unknown
at that point and must not rule the directory out on its own.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from xorq.backends.conftest import pytest_ignore_collect


# under backends/ but no backend's, so nothing marks it core automatically
pytestmark = pytest.mark.core

FLIGHT_TESTS = Path(__file__).parents[1] / "xorq_flight" / "tests"


def ignored(mark_expr: str) -> bool:
    config = SimpleNamespace(getoption=lambda _name: mark_expr)
    return pytest_ignore_collect(FLIGHT_TESTS, config)


@pytest.mark.parametrize(
    "mark_expr",
    (
        "xorq_flight",
        "backend",
        "xorq_flight and dep_floor",
        "(core or (xorq_flight and dep_floor)) and not uv_export and not slow",
        "xorq_flight and not slow",
        "not core",
    ),
)
def test_expression_that_can_select_the_backend_collects_it(mark_expr):
    assert not ignored(mark_expr)


@pytest.mark.parametrize(
    "mark_expr",
    (
        "duckdb",
        "core",
        "core and not uv_export",
        "not xorq_flight",
        "duckdb and dep_floor",
        # selections that need no backend keep skipping backend directories
        "slow(level=1)",
        "gcs",
        "bare_install",
    ),
)
def test_expression_that_cannot_select_the_backend_skips_it(mark_expr):
    assert ignored(mark_expr)
