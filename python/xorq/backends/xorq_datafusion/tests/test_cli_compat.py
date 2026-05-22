import subprocess
import sys

import pytest
from click.testing import CliRunner

from xorq.cli import cli, uv_group
from xorq.cli_constants import DEFAULT_OUTPUT_FORMAT, OutputFormats


@pytest.mark.skipif(
    sys.version_info > (3, 10), reason="compatibility test for Python 3.10"
)
@pytest.mark.xorq_datafusion
def test_output_formats_enum():
    for fmt in OutputFormats:
        assert OutputFormats(fmt.value) == fmt
    assert DEFAULT_OUTPUT_FORMAT == OutputFormats.parquet


@pytest.mark.skipif(
    sys.version_info > (3, 10), reason="compatibility test for Python 3.10"
)
@pytest.mark.xorq_datafusion
def test_uv_group_shows_help_without_subcommand():
    runner = CliRunner()
    result = runner.invoke(cli, ["uv"])
    assert result.exit_code == 0, result.output
    assert "Commands that use uv" in result.output
    for name in uv_group.commands:
        assert name in result.output


@pytest.mark.skipif(
    sys.version_info > (3, 10), reason="compatibility test for Python 3.10"
)
@pytest.mark.xorq_datafusion
@pytest.mark.parametrize(
    "cmd",
    [
        ("run",),
        ("run-cached",),
        (
            "uv",
            "run",
        ),
        ("run-unbound",),
    ],
)
def test_output_format_choices_in_help(cmd):
    runner = CliRunner()
    result = runner.invoke(cli, [*cmd, "--help"])
    assert result.exit_code == 0, result.output
    for fmt in OutputFormats:
        assert fmt.value in result.output


@pytest.mark.skipif(
    sys.version_info > (3, 10), reason="compatibility test for Python 3.10"
)
@pytest.mark.xorq_datafusion
@pytest.mark.parametrize("fmt", OutputFormats)
def test_output_format_accepted_by_cli(tmp_path, fmt):
    runner = CliRunner()
    result = runner.invoke(cli, ["run", str(tmp_path), "--format", fmt.value])
    assert "Invalid value for '-f' / '--format'" not in (result.output or "")


@pytest.mark.xorq_datafusion
def test_python_udf_process_exits_cleanly(tmp_path):
    """Regression XOR-357: tokio blocking-pool must not race Py_Finalize at exit."""
    sentinel = tmp_path / "atexit_ran"
    script = (
        "import atexit, sys\n"
        "from pathlib import Path\n"
        f"atexit.register(lambda: Path({str(sentinel)!r}).touch())\n"
        "import xorq.api as xo\n"
        "import xorq.expr.datatypes as dt\n"
        "import pyarrow.compute as pc\n"
        "@xo.udf.scalar.pyarrow\n"
        "def double_it(arr: dt.int64) -> dt.int64:\n"
        "    return pc.multiply(arr, 2)\n"
        "con = xo.connect()\n"
        "t = con.create_table('t', {'x': [1, 2, 3]})\n"
        "assert double_it(t.x).execute().tolist() == [2, 4, 6]\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        timeout=30,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert sentinel.exists(), (
        "atexit did not run — process likely hung or called os._exit"
    )
