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

BARE_IMPORT_CHECK = "python/xorq/tests/check_bare_imports.py"
MODULE_LISTING = "python/xorq/tests/bare_install_modules.txt"


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
def wheel(root_dir, tmp_path_factory):
    dist = tmp_path_factory.mktemp("dist")
    returncode, stdout, stderr = subprocess_run(
        ("uv", "build", "--wheel", "-o", str(dist)),
        cwd=root_dir,
        text=True,
    )
    assert returncode == 0, stderr
    (wheel,) = dist.glob("*.whl")
    return wheel


def test_core_modules_import(wheel, root_dir):
    """The static scan cannot see function-level imports; this can."""
    returncode, stdout, stderr = run_bare(
        wheel, ("python", BARE_IMPORT_CHECK), cwd=root_dir
    )
    assert returncode == 0, stderr


def test_module_listing_covers_the_new_declarations(root_dir):
    """Each declaration needs a module that imports it named in the listing.

    pygments comes from catalog/tui.py, numpy from backends/pandas/executor.py,
    xxhash from common/utils/dasher.  dasher is reached incidentally through
    xorq.api today, so name it explicitly: coverage should not depend on
    xorq.api continuing to import it.

    This exercises those import paths; it does not guard the declarations.
    Removing numpy, pygments or xxhash from pyproject.toml leaves these tests
    green, because each still arrives transitively via pandas, rich/textual and
    xorq-dasher.  test_declared_dependencies is what fails on removal.
    """
    listing = root_dir.joinpath(MODULE_LISTING).read_text()
    for module in (
        "xorq.catalog.tui",
        "xorq.backends.pandas.executor",
        "xorq.common.utils.dasher",
    ):
        assert module in listing


def test_build_then_run_round_trip(wheel, tmp_path):
    """`xorq build` succeeded even while `xorq run` was broken, so test both."""
    tmp_path.joinpath("expr.py").write_text(EXPR_PY)

    returncode, stdout, stderr = run_bare(
        wheel,
        # not stdout: OTel's ConsoleSpanExporter flushes there at shutdown, after
        # the path is printed, whenever OTEL_EXPORTER_CONSOLE_FALLBACK is set.
        ("xorq", "build", "expr.py", "-e", "expr", "--emit-build-path-to", "path.txt"),
        cwd=tmp_path,
    )
    assert returncode == 0, stderr
    build_path = tmp_path.joinpath("path.txt").read_text().strip()
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
