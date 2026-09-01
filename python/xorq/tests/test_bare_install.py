"""Drive the CLI against an environment holding only the declared dependencies.

Covers what the static scan in ``test_declared_dependencies`` cannot: imports
deferred inside a function body, which is most of the packager imports in
``xorq/cli.py``.
"""

import sys

import pytest

from xorq.common.utils.process_utils import subprocess_run


pytestmark = pytest.mark.bare_install


PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

EXPR_PY = """
import xorq.api as xo

t = xo.memtable({"a": [1, 2, 3]})
expr = t.filter(t.a > 1)
"""

# Modules on the build, run and catalog paths.
CORE_MODULES = (
    "xorq",
    "xorq.api",
    "xorq.cli",
    "xorq.catalog.cli",
    "xorq.ibis_yaml.compiler",
    "xorq.ibis_yaml.packager",
    "xorq.ibis_yaml.pep723",
)


def run_bare(wheel, args, cwd):
    return subprocess_run(
        (
            "uv",
            "tool",
            "run",
            "--isolated",
            "--python",
            PYTHON_VERSION,
            "--with",
            str(wheel),
            *args,
        ),
        cwd=cwd,
        text=True,
    )


@pytest.fixture(scope="module")
def wheel(project_root, tmp_path_factory):
    dist = tmp_path_factory.mktemp("dist")
    returncode, stdout, stderr = subprocess_run(
        ("uv", "build", "--wheel", "-o", str(dist)),
        cwd=project_root,
        text=True,
    )
    assert returncode == 0, stderr
    (wheel,) = dist.glob("*.whl")
    return wheel


def test_core_modules_import(wheel, tmp_path):
    """The static scan cannot see function-level imports; this can."""
    source = ";".join(f"import {module}" for module in CORE_MODULES)
    returncode, stdout, stderr = run_bare(wheel, ("python", "-c", source), cwd=tmp_path)
    assert returncode == 0, stderr


def test_build_then_run_round_trip(wheel, tmp_path):
    """`xorq build` succeeded even while `xorq run` was broken, so test both."""
    tmp_path.joinpath("expr.py").write_text(EXPR_PY)

    returncode, stdout, stderr = run_bare(
        wheel, ("xorq", "build", "expr.py", "-e", "expr"), cwd=tmp_path
    )
    assert returncode == 0, stderr
    build_path = stdout.strip().splitlines()[-1]
    assert tmp_path.joinpath(build_path).exists()

    returncode, stdout, stderr = run_bare(
        wheel, ("xorq", "run", build_path), cwd=tmp_path
    )
    assert returncode == 0, stderr


@pytest.mark.parametrize(
    "args",
    (
        ("xorq", "--help"),
        ("xorq", "build", "--help"),
        ("xorq", "run", "--help"),
        ("xorq", "catalog", "--help"),
    ),
)
def test_help_is_reachable(wheel, tmp_path, args):
    returncode, stdout, stderr = run_bare(wheel, args, cwd=tmp_path)
    assert returncode == 0, stderr
