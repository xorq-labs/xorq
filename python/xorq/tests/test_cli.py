import re
import sys
from pathlib import Path

import pytest
import toolz

from xorq.cli import build_command, main


build_run_examples_expr_names = (
    ("local_cache.py", "expr"),
    ("multi_engine.py", "expr"),
    ("remote_caching.py", "expr"),
    ("iris_example.py", "expr"),
    ("simple_example.py", "expr"),
    ("deferred_read_csv.py", "pg_expr_replace"),
    ("train_test_splits.py", "train_table"),
    ("train_test_splits.py", "split_column"),
    ("postgres_caching.py", "expr"),
)


# Run the CLI (with try/except to prevent SystemExit)
main_no_exit = toolz.excepts(SystemExit, main)


def test_build_command(monkeypatch, tmp_path, fixture_dir, capsys):
    builds_dir = tmp_path / "builds"
    script_path = fixture_dir / "pipeline.py"

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
    main_no_exit()

    # Check output
    captured = capsys.readouterr()
    assert "Building expr" in captured.err

    assert builds_dir.exists()


def test_build_command_with_udtf(monkeypatch, tmp_path, fixture_dir, capsys):
    builds_dir = tmp_path / "builds"
    script_path = fixture_dir / "udxf_expr.py"

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

    main_no_exit()

    captured = capsys.readouterr()
    assert "Building expr" in captured.err

    assert builds_dir.exists()


def test_build_command_on_notebook(monkeypatch, tmp_path, fixture_dir, capsys):
    builds_dir = tmp_path / "builds"
    script_path = fixture_dir / "pipeline.ipynb"

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

    main_no_exit()

    # Check output
    captured = capsys.readouterr()
    assert "Building expr" in captured.err

    assert builds_dir.exists()


def test_run_command_default(monkeypatch, tmp_path, fixture_dir, capsys):
    target_dir = tmp_path / "build"
    script_path = fixture_dir / "pipeline.py"

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

        main_no_exit()
    else:
        raise AssertionError("No expression hash")


@pytest.mark.parametrize("output_format", ["csv", "json", "parquet"])
def test_run_command(monkeypatch, tmp_path, fixture_dir, capsys, output_format):
    target_dir = tmp_path / "build"
    script_path = fixture_dir / "pipeline.py"

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

        main_no_exit()
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    else:
        raise AssertionError("No expression hash")


@pytest.mark.parametrize("output_format", ["csv", "json", "parquet"])
def test_run_command_stdout(
    monkeypatch, tmp_path, fixture_dir, capsysbinary, output_format
):
    target_dir = tmp_path / "build"
    script_path = fixture_dir / "pipeline.py"

    build_command(str(script_path), "expr", builds_dir=str(target_dir))
    capture = capsysbinary.readouterr()

    if match := re.search(f"{target_dir}/([0-9a-f]+)", str(capture.out)):
        expression_path = match.group()
        test_args = [
            "xorq",
            "run",
            expression_path,
            "--output-path",
            "-",
            "--format",
            output_format,
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        main_no_exit()
        capture = capsysbinary.readouterr()
        assert capture.out

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
    monkeypatch, tmp_path, fixture_dir, capsys, expression, message
):
    builds_dir = tmp_path / "builds"
    script_path = fixture_dir / "pipeline.py"

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

    main_no_exit()

    # Check output
    captured = capsys.readouterr()
    assert message in captured.err


@pytest.mark.parametrize(
    ("example", "expr_name"),
    build_run_examples_expr_names,
)
def test_examples(
    example,
    expr_name,
    examples_dir,
    tmp_path,
    monkeypatch,
    capsys,
):
    # build
    builds_dir = tmp_path / "builds"
    example_path = examples_dir / example
    assert example_path.exists()
    build_args = (
        "xorq",
        "build",
        str(example_path),
        "--expr-name",
        expr_name,
        "--builds-dir",
        str(builds_dir),
    )
    monkeypatch.setattr(sys, "argv", build_args)
    main_no_exit()
    captured = capsys.readouterr()
    print(captured.err, file=sys.stderr)
    expression_path = Path(captured.out.strip())
    assert expression_path.exists()

    # run
    output_format = "parquet"
    output_path = expression_path / f"test.{output_format}"
    assert not output_path.exists()
    run_args = (
        "xorq",
        "run",
        str(expression_path),
        "--format",
        output_format,
        "--output-path",
        str(output_path),
    )
    monkeypatch.setattr(sys, "argv", run_args)
    main_no_exit()
    captured = capsys.readouterr()
    print(captured.err, file=sys.stderr)
    assert output_path.exists()
