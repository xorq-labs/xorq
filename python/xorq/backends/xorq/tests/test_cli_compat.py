import sys

import pytest
from click.testing import CliRunner

from xorq.cli import OutputFormats, cli


@pytest.mark.skipif(
    sys.version_info > (3, 10), reason="compatibility test for Python 3.10"
)
@pytest.mark.xorq
def test_output_formats_enum():
    for fmt in OutputFormats:
        assert OutputFormats(fmt.value) == fmt
    assert OutputFormats.default == OutputFormats.parquet


@pytest.mark.skipif(
    sys.version_info > (3, 10), reason="compatibility test for Python 3.10"
)
@pytest.mark.xorq
@pytest.mark.parametrize("cmd", ["run", "run-cached", "uv-run", "run-unbound"])
def test_output_format_choices_in_help(cmd):
    runner = CliRunner()
    result = runner.invoke(cli, [cmd, "--help"])
    assert result.exit_code == 0, result.output
    for fmt in OutputFormats:
        assert fmt.value in result.output


@pytest.mark.skipif(
    sys.version_info > (3, 10), reason="compatibility test for Python 3.10"
)
@pytest.mark.xorq
@pytest.mark.parametrize("fmt", OutputFormats)
def test_output_format_accepted_by_cli(tmp_path, fmt):
    runner = CliRunner()
    result = runner.invoke(cli, ["run", str(tmp_path), "--format", fmt.value])
    assert "Invalid value for '-f' / '--format'" not in (result.output or "")
