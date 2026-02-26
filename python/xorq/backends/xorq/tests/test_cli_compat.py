"""
Compatibility tests for the CLI on Python 3.10.

In Python 3.10, StrEnum is not available in the stdlib and falls back to the
`strenum` package.  These tests verify that OutputFormats and the click
commands that expose it work correctly without requiring any external services.
"""

import pytest
from click.testing import CliRunner

from xorq.cli import OutputFormats, cli


def test_output_formats_enum():
    """OutputFormats StrEnum must be constructable from string values."""
    for fmt in OutputFormats:
        assert OutputFormats(fmt.value) == fmt
    assert OutputFormats.default == OutputFormats.parquet


@pytest.mark.parametrize("cmd", ["run", "run-cached", "uv-run", "run-unbound"])
def test_output_format_choices_in_help(cmd):
    """All OutputFormats values must appear as valid choices in each command's help."""
    runner = CliRunner()
    result = runner.invoke(cli, [cmd, "--help"])
    assert result.exit_code == 0, result.output
    for fmt in OutputFormats:
        assert fmt.value in result.output


@pytest.mark.parametrize("fmt", OutputFormats)
def test_output_format_accepted_by_cli(tmp_path, fmt):
    """Each OutputFormats value must be accepted by click without an 'Invalid choice' error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", str(tmp_path), "--format", fmt.value])
    assert "Invalid value for '-f' / '--format'" not in (result.output or "")
