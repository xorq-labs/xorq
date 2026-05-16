import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import click
import pytest
import tomlkit

from xorq.cli import uv_build_command
from xorq.ibis_yaml.packager import (
    PYPROJECT_NAME,
    PackagedBuilder,
    WheelPackager,
    _cap_requires_python,
)
from xorq.ibis_yaml.pep723 import read_inline_metadata, synthesize_project


pytestmark = pytest.mark.core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_SCRIPT = textwrap.dedent("""\
    # /// script
    # requires-python = ">=3.10"
    # dependencies = ["pandas>=2.0"]
    # ///
    import pandas as pd
    print(pd.__version__)
""")

SCRIPT_NO_METADATA = textwrap.dedent("""\
    import sys
    print(sys.version)
""")

SCRIPT_MULTIPLE_BLOCKS = textwrap.dedent("""\
    # /// script
    # dependencies = ["pandas"]
    # ///

    x = 1

    # /// script
    # dependencies = ["numpy"]
    # ///
""")

SCRIPT_WITH_XORQ = textwrap.dedent("""\
    # /// script
    # requires-python = ">=3.10"
    # dependencies = ["xorq>=0.3.0"]
    # ///
    import xorq
""")

SCRIPT_XORQ_EXTRAS = textwrap.dedent("""\
    # /// script
    # requires-python = ">=3.10"
    # dependencies = ["xorq[postgres]>=0.3.0"]
    # ///
    import xorq
""")


def _write_script(directory, name, content):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(content)
    return path


def _mock_uv_lock(mock_run):
    """Configure a subprocess.run mock that creates uv.lock in the target dir."""
    real_completed = subprocess.CompletedProcess(args=["uv", "lock"], returncode=0)

    def _side_effect(args, **kwargs):
        for i, arg in enumerate(args):
            if arg == "--directory" and i + 1 < len(args):
                (Path(args[i + 1]) / "uv.lock").write_text("# mock lock")
                break
        return real_completed

    mock_run.side_effect = _side_effect


def _make_pyproject(directory):
    path = Path(directory) / PYPROJECT_NAME
    path.write_text(
        '[build-system]\nrequires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n\n'
        '[project]\nname = "test-pkg"\nversion = "0.0.0"\n'
        'requires-python = ">=3.10"\n'
    )
    return path


# ---------------------------------------------------------------------------
# read_inline_metadata
# ---------------------------------------------------------------------------


def test_read_inline_metadata_valid_script():
    meta = read_inline_metadata(SIMPLE_SCRIPT)
    assert meta is not None
    assert meta["requires-python"] == ">=3.10"
    assert "pandas>=2.0" in meta["dependencies"]


def test_read_inline_metadata_no_metadata():
    assert read_inline_metadata(SCRIPT_NO_METADATA) is None


def test_read_inline_metadata_multiple_blocks_raises():
    with pytest.raises(ValueError, match="multiple"):
        read_inline_metadata(SCRIPT_MULTIPLE_BLOCKS)


def test_read_inline_metadata_empty_string():
    assert read_inline_metadata("") is None


def test_read_inline_metadata_non_script_block_ignored():
    script = textwrap.dedent("""\
        # /// tool
        # name = "ruff"
        # ///
        import sys
    """)
    assert read_inline_metadata(script) is None


# ---------------------------------------------------------------------------
# _cap_requires_python: shared across pyproject and PEP 723 paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected_fragment"),
    [
        (">=3.10", "<3.14"),
        (">=3.11", "<3.14"),
        (">=3.10,<3.13", "<3.13"),
        (None, ">=3.10"),
        (None, "<3.14"),
    ],
)
def test__cap_requires_python(raw, expected_fragment):
    result = _cap_requires_python(raw)
    assert expected_fragment in result


# ---------------------------------------------------------------------------
# synthesize_project
# ---------------------------------------------------------------------------


def test_synthesize_no_inline_metadata_raises(tmp_path):
    script = _write_script(tmp_path, "bare.py", SCRIPT_NO_METADATA)
    with pytest.raises(ValueError, match="no PEP 723 inline metadata"):
        synthesize_project(script)


def test_synthesize_xorq_auto_injected(tmp_path):
    script = _write_script(tmp_path, "test.py", SIMPLE_SCRIPT)
    with patch("xorq.ibis_yaml.pep723.subprocess.run") as mock_run:
        _mock_uv_lock(mock_run)
        with synthesize_project(script, xorq_version="0.3.99") as synth_path:
            pyproject = tomlkit.loads((Path(synth_path) / "pyproject.toml").read_text())
            deps = pyproject["project"]["dependencies"]
            assert any("xorq==0.3.99" in d for d in deps)


def test_synthesize_xorq_not_duplicated_when_present(tmp_path):
    script = _write_script(tmp_path, "test.py", SCRIPT_WITH_XORQ)
    with patch("xorq.ibis_yaml.pep723.subprocess.run") as mock_run:
        _mock_uv_lock(mock_run)
        with synthesize_project(script, xorq_version="0.3.99") as synth_path:
            pyproject = tomlkit.loads((Path(synth_path) / "pyproject.toml").read_text())
            deps = pyproject["project"]["dependencies"]
            xorq_deps = [d for d in deps if "xorq" in d]
            assert len(xorq_deps) == 1
            assert "xorq>=0.3.0" in xorq_deps[0]


def test_synthesize_xorq_with_extras_recognized(tmp_path):
    script = _write_script(tmp_path, "test.py", SCRIPT_XORQ_EXTRAS)
    with patch("xorq.ibis_yaml.pep723.subprocess.run") as mock_run:
        _mock_uv_lock(mock_run)
        with synthesize_project(script, xorq_version="0.3.99") as synth_path:
            pyproject = tomlkit.loads((Path(synth_path) / "pyproject.toml").read_text())
            deps = pyproject["project"]["dependencies"]
            xorq_deps = [d for d in deps if "xorq" in d]
            assert len(xorq_deps) == 1


@pytest.mark.parametrize(
    ("requires_python", "expected_fragment"),
    [
        (">=3.10", "<3.14"),
        (">=3.12", "<3.14"),
        (">=3.10,<3.13", "<3.13"),
    ],
)
def test_synthesize_requires_python_capped(
    tmp_path, requires_python, expected_fragment
):
    script_text = (
        f"# /// script\n"
        f'# requires-python = "{requires_python}"\n'
        f'# dependencies = ["pandas"]\n'
        f"# ///\n"
    )
    script = _write_script(tmp_path, "test.py", script_text)
    with patch("xorq.ibis_yaml.pep723.subprocess.run") as mock_run:
        _mock_uv_lock(mock_run)
        with synthesize_project(script, xorq_version="0.3.99") as synth_path:
            pyproject = tomlkit.loads((Path(synth_path) / "pyproject.toml").read_text())
            raw = pyproject["project"]["requires-python"]
            assert expected_fragment in raw


def test_synthesize_uv_lock_failure_raises(tmp_path):
    script = _write_script(tmp_path, "test.py", SIMPLE_SCRIPT)
    with patch("xorq.ibis_yaml.pep723.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "uv lock", stderr="resolution failed"
        )
        with pytest.raises(RuntimeError, match="failed to resolve"):
            synthesize_project(script, xorq_version="0.3.99")


def test_synthesize_sanitized_project_name(tmp_path):
    script = _write_script(tmp_path, "My Script.v2.py", SIMPLE_SCRIPT)
    with patch("xorq.ibis_yaml.pep723.subprocess.run") as mock_run:
        _mock_uv_lock(mock_run)
        with synthesize_project(script, xorq_version="0.3.99") as synth_path:
            pyproject = tomlkit.loads((Path(synth_path) / "pyproject.toml").read_text())
            name = pyproject["project"]["name"]
            assert name == "xorq-script-my-script-v2"


def test_synthesize_wheel_builds_successfully(tmp_path):
    script = _write_script(tmp_path, "standalone.py", SIMPLE_SCRIPT)
    with patch("xorq.ibis_yaml.pep723.subprocess.run") as mock_run:
        _mock_uv_lock(mock_run)
        synth_dir = synthesize_project(script, xorq_version="0.3.99")
    synth_path = synth_dir.name
    try:
        out_dir = tmp_path / "dist"
        out_dir.mkdir()
        subprocess.run(
            ("uv", "build", "--wheel", "--out-dir", str(out_dir), synth_path),
            check=True,
        )
        wheels = list(out_dir.glob("*.whl"))
        assert len(wheels) == 1
        assert "xorq_script_standalone" in wheels[0].name
    finally:
        synth_dir.cleanup()


# ---------------------------------------------------------------------------
# WheelPackager.from_script_path: dispatch logic
# ---------------------------------------------------------------------------


def test_dispatch_project_only(tmp_path):
    _make_pyproject(tmp_path)
    (tmp_path / "uv.lock").write_text("# lock")
    script = _write_script(tmp_path, "expr.py", SCRIPT_NO_METADATA)
    packager = WheelPackager.from_script_path(str(script))
    assert packager.project_path == tmp_path


def test_dispatch_inline_only(tmp_path):
    subdir = tmp_path / "standalone"
    script = _write_script(subdir, "expr.py", SIMPLE_SCRIPT)
    with patch("xorq.ibis_yaml.pep723.subprocess.run") as mock_run:
        _mock_uv_lock(mock_run)
        packager = WheelPackager.from_script_path(str(script))
        assert packager._synth_dir is not None


def test_dispatch_project_preferred_over_inline(tmp_path):
    _make_pyproject(tmp_path)
    (tmp_path / "uv.lock").write_text("# lock")
    script = _write_script(tmp_path, "expr.py", SIMPLE_SCRIPT)
    packager = WheelPackager.from_script_path(str(script))
    assert packager.project_path == tmp_path


def test_dispatch_neither_raises(tmp_path):
    subdir = tmp_path / "isolated"
    script = _write_script(subdir, "expr.py", SCRIPT_NO_METADATA)
    with pytest.raises(ValueError, match="No pyproject.toml found"):
        WheelPackager.from_script_path(str(script))


def test_dispatch_pep723_flag_forces_inline(tmp_path):
    _make_pyproject(tmp_path)
    (tmp_path / "uv.lock").write_text("# lock")
    script = _write_script(tmp_path, "expr.py", SIMPLE_SCRIPT)
    with patch("xorq.ibis_yaml.pep723.subprocess.run") as mock_run:
        _mock_uv_lock(mock_run)
        packager = WheelPackager.from_script_path(str(script), pep723=True)
        assert packager._synth_dir is not None


def test_dispatch_extras_raises_for_pep723(tmp_path):
    subdir = tmp_path / "isolated"
    script = _write_script(subdir, "expr.py", SIMPLE_SCRIPT)
    with patch("xorq.ibis_yaml.pep723.subprocess.run") as mock_run:
        _mock_uv_lock(mock_run)
        with pytest.raises(ValueError, match="do not support --extra"):
            WheelPackager.from_script_path(str(script), pep723=True, extras=("pg",))


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


def test_packaged_builder_project_path_and_pep723_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        PackagedBuilder.from_script_path(
            "dummy.py", project_path="/some/path", pep723=True
        )


def test_cli_project_path_and_pep723_exclusive():
    with pytest.raises(click.UsageError, match="mutually exclusive"):
        uv_build_command("dummy.py", project_path="/some/path", pep723=True)
