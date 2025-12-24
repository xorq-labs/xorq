"""Tests for Arrow IPC streaming in CLI."""

import io
import re

import pyarrow as pa
import pytest

from xorq.cli.io import read_arrow_stream, write_arrow_stream
from xorq.common.utils.process_utils import subprocess_run


def test_write_arrow_stream():
    """Test writing Arrow IPC stream to a buffer."""
    # Create test data
    table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    batches = table.to_batches()

    # Write to buffer
    buf = io.BytesIO()
    write_arrow_stream(iter(batches), out=buf)

    # Read back
    buf.seek(0)
    result = pa.ipc.open_stream(buf).read_all()

    assert result.equals(table)


def test_read_arrow_stream():
    """Test reading Arrow IPC stream from a buffer."""
    # Create test data
    table = pa.table({"x": [10, 20, 30], "y": [40, 50, 60]})

    # Write to buffer
    buf = io.BytesIO()
    writer = pa.ipc.new_stream(buf, table.schema)
    for batch in table.to_batches():
        writer.write_batch(batch)
    writer.close()

    # Read back using our function
    buf.seek(0)
    result = read_arrow_stream(buf)

    assert result.equals(table)


def test_write_arrow_stream_empty():
    """Test writing empty iterator."""
    buf = io.BytesIO()
    write_arrow_stream(iter([]), out=buf)

    # Should write nothing
    assert buf.tell() == 0


@pytest.mark.slow(level=1)
def test_run_command_arrow_output(tmp_path, fixture_dir):
    """Test running a build with Arrow output format."""
    target_dir = tmp_path / "build"
    script_path = fixture_dir / "pipeline.py"

    # Build the expression
    build_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(target_dir),
    ]
    (returncode, stdout, stderr) = subprocess_run(build_args)
    assert returncode == 0, stderr

    if match := re.search(f"{target_dir}/([0-9a-f]+)", stdout.decode("ascii")):
        output_path = tmp_path / "test.arrow"
        expression_path = match.group()

        # Run with arrow output
        test_args = [
            "xorq",
            "run",
            expression_path,
            "--output-path",
            str(output_path),
            "--format",
            "arrow",
        ]
        (returncode, _, stderr) = subprocess_run(test_args)
        assert returncode == 0, stderr
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify we can read the arrow file
        with open(output_path, "rb") as f:
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()
            assert len(table) > 0

    else:
        raise AssertionError("No expression hash")


@pytest.mark.slow(level=1)
def test_run_command_arrow_output_stdout(tmp_path, fixture_dir):
    """Test running a build with Arrow output to stdout."""
    target_dir = tmp_path / "build"
    script_path = fixture_dir / "pipeline.py"

    # Build the expression
    build_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(target_dir),
    ]
    (returncode, stdout, stderr) = subprocess_run(build_args)
    assert returncode == 0, stderr

    if match := re.search(f"{target_dir}/([0-9a-f]+)", stdout.decode("ascii")):
        expression_path = match.group()

        # Run with arrow output to stdout
        test_args = [
            "xorq",
            "run",
            expression_path,
            "--output-path",
            "-",
            "--format",
            "arrow",
        ]
        (returncode, stdout, stderr) = subprocess_run(test_args)
        assert returncode == 0, stderr
        assert len(stdout) > 0

        # Verify the stdout contains valid Arrow IPC data
        buf = io.BytesIO(stdout)
        reader = pa.ipc.open_stream(buf)
        table = reader.read_all()
        assert len(table) > 0

    else:
        raise AssertionError("No expression hash")


@pytest.mark.slow(level=1)
def test_run_command_with_batch_size(tmp_path, fixture_dir):
    """Test running a build with custom batch size."""
    target_dir = tmp_path / "build"
    script_path = fixture_dir / "pipeline.py"

    # Build the expression
    build_args = [
        "xorq",
        "build",
        str(script_path),
        "--expr-name",
        "expr",
        "--builds-dir",
        str(target_dir),
    ]
    (returncode, stdout, stderr) = subprocess_run(build_args)
    assert returncode == 0, stderr

    if match := re.search(f"{target_dir}/([0-9a-f]+)", stdout.decode("ascii")):
        output_path = tmp_path / "test.arrow"
        expression_path = match.group()

        # Run with arrow output and custom batch size
        test_args = [
            "xorq",
            "run",
            expression_path,
            "--output-path",
            str(output_path),
            "--format",
            "arrow",
            "--batch-size",
            "10",
        ]
        (returncode, _, stderr) = subprocess_run(test_args)
        assert returncode == 0, stderr
        assert output_path.exists()

        # Verify we can read the arrow file
        with open(output_path, "rb") as f:
            reader = pa.ipc.open_stream(f)
            table = reader.read_all()
            assert len(table) > 0

    else:
        raise AssertionError("No expression hash")
