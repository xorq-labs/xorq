import re
import sys
from pathlib import Path

import pytest

from xorq.cli import build_command, main


def test_build_command(monkeypatch, tmp_path, capsys):
    builds_dir = tmp_path / "builds"
    script_path = Path(__file__).absolute().parent / "fixtures" / "pipeline.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "-e",
        "expr",
        "--builds-dir",
        str(builds_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run the CLI (with try/except to prevent SystemExit)
    try:
        main()
    except SystemExit:
        pass

    # Check output
    captured = capsys.readouterr()
    assert "Building expr" in captured.out

    assert builds_dir.exists()


def test_build_command_with_udtf(monkeypatch, tmp_path, capsys):
    builds_dir = tmp_path / "builds"
    script_path = Path(__file__).absolute().parent / "fixtures" / "udxf_expr.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "-e",
        "expr",
        "--builds-dir",
        str(builds_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    try:
        main()
    except SystemExit:
        pass

    captured = capsys.readouterr()
    assert "Building expr" in captured.out

    assert builds_dir.exists()


def test_build_command_on_notebook(monkeypatch, tmp_path, capsys):
    builds_dir = tmp_path / "builds"
    script_path = Path(__file__).absolute().parent / "fixtures" / "pipeline.ipynb"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "-e",
        "expr",
        "--builds-dir",
        str(builds_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run the CLI (with try/except to prevent SystemExit)
    try:
        main()
    except SystemExit:
        pass

    # Check output
    captured = capsys.readouterr()
    assert "Building expr" in captured.out

    assert builds_dir.exists()


def test_run_command_default(monkeypatch, tmp_path, capsys):
    target_dir = tmp_path / "build"
    script_path = Path(__file__).absolute().parent / "fixtures" / "pipeline.py"

    build_command(str(script_path), "expr", builds_dir=str(target_dir))
    capture = capsys.readouterr()

    if match := re.search(f"{target_dir}/([0-9a-f]+)", str(capture.out)):
        expression_path = match.group()
        test_args = [
            "xorq",
            "run",
            expression_path,
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        try:
            main()
        except SystemExit:
            pass
    else:
        raise AssertionError("No expression hash")


@pytest.mark.parametrize("output_format", ["csv", "json", "parquet"])
def test_run_command(monkeypatch, tmp_path, capsys, output_format):
    target_dir = tmp_path / "build"
    script_path = Path(__file__).absolute().parent / "fixtures" / "pipeline.py"

    build_command(str(script_path), "expr", builds_dir=str(target_dir))
    capture = capsys.readouterr()

    if match := re.search(f"{target_dir}/([0-9a-f]+)", str(capture.out)):
        output_path = tmp_path / f"test.{output_format}"
        expression_path = match.group()
        test_args = [
            "xorq",
            "run",
            expression_path,
            "--output-path",
            str(output_path),
            "--format",
            output_format,
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        try:
            main()
        except SystemExit:
            pass
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    else:
        raise AssertionError("No expression hash")


@pytest.mark.parametrize(
    "expression,message",
    [
        ("integer", "The object integer must be an instance of"),
        ("missing", "Expression missing not found"),
    ],
)
def test_build_command_bad_expr_name(
    monkeypatch, tmp_path, capsys, expression, message
):
    builds_dir = tmp_path / "builds"
    script_path = Path(__file__).absolute().parent / "fixtures" / "pipeline.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "-e",
        expression,
        "--builds-dir",
        str(builds_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run the CLI (with try/except to prevent SystemExit)
    try:
        main()
    except SystemExit:
        pass

    # Check output
    captured = capsys.readouterr()
    assert message in captured.err
