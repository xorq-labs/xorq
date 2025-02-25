import sys
from pathlib import Path

from xorq.cli import main


def test_build_command(monkeypatch, tmp_path, capsys):
    target_dir = tmp_path / "build"
    script_path = Path(__file__).absolute().parent / "fixtures" / "pipeline.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "-e",
        "expr",
        "--target-dir",
        str(target_dir),
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

    assert target_dir.exists()


def test_build_command_not_implemented(monkeypatch, capsys):
    script_path = Path(__file__).absolute().parent / "fixtures" / "pipeline.py"

    test_args = [
        "xorq",
        "build",
        str(script_path),
        "--target-dir",
        str(Path.cwd()),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run the CLI (with try/except to prevent SystemExit)
    try:
        main()
    except SystemExit:
        pass

    # Check output
    captured = capsys.readouterr()
    assert "Expected one, and only one expression" in captured.out
