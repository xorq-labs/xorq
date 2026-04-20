import functools
import sys
import tempfile
import zipfile
from pathlib import (
    Path,
)

import pytest

from xorq.catalog.catalog import _ensure_wheel_artifacts
from xorq.common.utils.download_utils import (
    download_xorq_template,
)
from xorq.common.utils.zip_utils import (
    ZipProxy,
    append_toplevel,
)
from xorq.ibis_yaml.enums import DumpFiles
from xorq.ibis_yaml.packager import (
    PYPROJECT_NAME,
    UVLOCK_NAME,
    PackagedBuilder,
    PackagedRunner,
    WheelBundle,
    WheelPackager,
    _nix_env,
    _read_requires_python,
    _validate_python_version,
    find_file_upwards,
    uv_export_requirements,
)
from xorq.init_templates import InitTemplates


@functools.cache
def get_template_bytes(template=InitTemplates.default):
    with tempfile.TemporaryDirectory() as td:
        target = Path(td).joinpath("template")
        path = download_xorq_template(template, target=target)
        assert target == path
        template_bytes = path.read_bytes()
        return template_bytes


def prep_template_tmpdir(template, tmpdir):
    tmpdir = Path(tmpdir)
    zip_path = tmpdir.joinpath("template.zip")
    zip_path.write_bytes(get_template_bytes(template))
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        (first, *_) = names
        root_dir = first.rstrip("/").split("/")[0]
        zf.extractall(tmpdir)
    project_path = tmpdir.joinpath(root_dir)
    # Remove pre-committed requirements.txt so the packager regenerates it
    # from uv.lock. The templates' requirements.txt may have been exported
    # with a different uv version than CI uses, causing a sync-check failure.
    stale_reqs = project_path / DumpFiles.requirements
    stale_reqs.unlink(missing_ok=True)
    return (zip_path, project_path)


@pytest.mark.slow(level=1)
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_wheel_packager(template, tmpdir):
    zip_path, project_path = prep_template_tmpdir(template, tmpdir)
    packager = WheelPackager(project_path)
    bundle = packager.build()
    assert bundle.wheel_path.exists()
    assert bundle.wheel_path.suffix == ".whl"
    assert bundle.requirements_path.exists()


@pytest.mark.slow(level=1)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_wheel_builder(template, tmpdir):
    zip_path, project_path = prep_template_tmpdir(template, tmpdir)
    script_path = project_path.joinpath("expr.py")
    packager = WheelPackager(project_path)
    packaged_builder = PackagedBuilder(
        script_path=script_path,
        bundle=packager.build(),
    )
    assert packaged_builder.build_path, packaged_builder.build_result.stderr
    assert list(packaged_builder.build_path.glob("*.whl"))
    assert (packaged_builder.build_path / DumpFiles.requirements).exists()


@pytest.mark.slow(level=2)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_wheel_runner(template, tmpdir):
    tmpdir = Path(tmpdir)
    output_path = tmpdir.joinpath("output")
    zip_path, project_path = prep_template_tmpdir(template, tmpdir)
    script_path = project_path.joinpath("expr.py")
    packager = WheelPackager(project_path)
    packaged_builder = PackagedBuilder(
        script_path=script_path,
        bundle=packager.build(),
    )
    packaged_runner = PackagedRunner(
        packaged_builder.build_path, output_path=str(output_path)
    )
    assert packaged_runner.run_result.returncode == 0
    assert output_path.exists()


# ---------------------------------------------------------------------------
# Unit tests for pure helpers (no subprocess / network)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["3.11", ">=3.10", ">=3.10,<3.14", None],
)
def test_validate_python_version_accepts_valid(value):
    _validate_python_version(None, None, value)


@pytest.mark.parametrize(
    "value",
    ["garbage", "abc", ">>>3.10"],
)
def test_validate_python_version_rejects_invalid(value):
    with pytest.raises(ValueError, match="invalid python version specifier"):
        _validate_python_version(None, None, value)


def test_wheel_packager_rejects_bad_python_version():
    with pytest.raises(ValueError, match="invalid python version specifier"):
        WheelPackager(project_path="/tmp", python_version="garbage")


def test_wheel_packager_rejects_missing_lock_and_requirements(tmp_path):
    _make_pyproject(tmp_path)
    with pytest.raises(FileNotFoundError, match="neither .* nor .* found"):
        WheelPackager(tmp_path)


# ---------------------------------------------------------------------------
# zip_utils: append_toplevel and ZipProxy error paths
# ---------------------------------------------------------------------------


def test_append_toplevel(tmp_path):
    # create a zip with a root dir
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("myroot/existing.txt", "hello")
    # append a file
    new_file = tmp_path / "added.txt"
    new_file.write_text("world")
    append_toplevel(zip_path, new_file)
    # verify it landed under root dir
    zp = ZipProxy(zip_path)
    assert zp.toplevel_name_exists("added.txt")
    with zp.open_toplevel_member("added.txt") as fh:
        assert fh.read() == b"world"


def test_zip_proxy_rejects_non_zip(tmp_path):
    tar_path = tmp_path / "test.tar.gz"
    tar_path.write_bytes(b"")
    with pytest.raises(ValueError, match="expected .zip file"):
        ZipProxy(tar_path)


# ---------------------------------------------------------------------------
# PackagedBuilder / PackagedRunner: validation error paths
# ---------------------------------------------------------------------------


def test_wheel_bundle_rejects_missing_wheel(tmp_path):
    with pytest.raises(FileNotFoundError, match="wheel not found"):
        WheelBundle(
            wheel_path=tmp_path / "missing.whl",
            requirements_path=tmp_path / "requirements.txt",
        )


def test_packaged_runner_rejects_missing_build_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="build path does not exist"):
        PackagedRunner(build_path=tmp_path / "nonexistent")


def test_packaged_runner_rejects_missing_wheel(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    # requirements exists but wheel doesn't
    (build_dir / DumpFiles.requirements).write_text("requests==2.31.0")
    with pytest.raises(FileNotFoundError, match="invalid build path"):
        PackagedRunner(build_path=build_dir)


def test_packaged_runner_rejects_missing_requirements(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    _make_wheel(build_dir)
    with pytest.raises(FileNotFoundError, match="invalid build path"):
        PackagedRunner(build_path=build_dir)


# ---------------------------------------------------------------------------
# Helpers for unit tests below
# ---------------------------------------------------------------------------


def _make_wheel(directory, requires_python=">=3.10"):
    """Create a minimal .whl with valid METADATA."""
    wheel_path = Path(directory) / "pkg-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr(
            "pkg-0.0.0.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: pkg\nVersion: 0.0.0\n"
            f"Requires-Python: {requires_python}\n",
        )
    return wheel_path


def _make_pyproject(directory, requires_python=">=3.10"):
    """Write a minimal pyproject.toml."""
    path = Path(directory) / PYPROJECT_NAME
    path.write_text(
        '[build-system]\nrequires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n\n'
        '[project]\nname = "test-pkg"\nversion = "0.0.0"\n'
        f'requires-python = "{requires_python}"\n'
    )
    return path


# ---------------------------------------------------------------------------
# _read_requires_python: all input types and error paths
# ---------------------------------------------------------------------------


def test_read_requires_python_from_pyproject_file(tmp_path):
    pyproject = _make_pyproject(tmp_path, requires_python=">=3.11")
    result = _read_requires_python(pyproject)
    assert ">=3.11" in result


def test_read_requires_python_from_directory(tmp_path):
    _make_pyproject(tmp_path, requires_python=">=3.11")
    result = _read_requires_python(tmp_path)
    assert ">=3.11" in result


def test_read_requires_python_from_wheel(tmp_path):
    wheel = _make_wheel(tmp_path, requires_python=">=3.10")
    result = _read_requires_python(wheel)
    assert ">=3.10" in result


def test_read_requires_python_wheel_missing_dist_info(tmp_path):
    wheel_path = tmp_path / "bad-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr("some_file.txt", "content")
    with pytest.raises(ValueError, match="no .dist-info/METADATA"):
        _read_requires_python(wheel_path)


def test_read_requires_python_wheel_missing_requires_python(tmp_path):
    wheel_path = tmp_path / "bad-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr(
            "bad-0.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: bad\nVersion: 0.0.0\n",
        )
    with pytest.raises(ValueError, match="no Requires-Python"):
        _read_requires_python(wheel_path)


def test_read_requires_python_rejects_unknown_path(tmp_path):
    txt = tmp_path / "random.txt"
    txt.write_text("hello")
    with pytest.raises(ValueError, match="can only handle"):
        _read_requires_python(txt)


# ---------------------------------------------------------------------------
# find_file_upwards: error path
# ---------------------------------------------------------------------------


def test_find_file_upwards_not_found(tmp_path):
    with pytest.raises(ValueError, match="could not find"):
        find_file_upwards(tmp_path, "nonexistent_file_xyz.toml")


# ---------------------------------------------------------------------------
# _nix_env: conditional branches
# ---------------------------------------------------------------------------


def test_nix_env_returns_none_outside_nix(monkeypatch):
    monkeypatch.setattr("xorq.ibis_yaml.packager.in_nix_shell", lambda: False)
    assert _nix_env() is None


def test_nix_env_uses_override_inside_nix(monkeypatch):
    monkeypatch.setattr("xorq.ibis_yaml.packager.in_nix_shell", lambda: True)
    monkeypatch.setenv("UV_TOOL_RUN_LD_LIBRARY_PATH", "/custom/lib")
    env = _nix_env()
    assert env is not None
    assert env["LD_LIBRARY_PATH"] == "/custom/lib"


def test_nix_env_removes_ld_path_inside_nix(monkeypatch):
    monkeypatch.setattr("xorq.ibis_yaml.packager.in_nix_shell", lambda: True)
    monkeypatch.delenv("UV_TOOL_RUN_LD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/should/be/removed")
    env = _nix_env()
    assert env is not None
    assert "LD_LIBRARY_PATH" not in env


# ---------------------------------------------------------------------------
# WheelBundle.from_build_path
# ---------------------------------------------------------------------------


def test_wheel_bundle_from_build_path(tmp_path):
    wheel = _make_wheel(tmp_path)
    (tmp_path / DumpFiles.requirements).write_text("requests==2.31.0\n")
    bundle = WheelBundle.from_build_path(tmp_path)
    assert bundle.wheel_path == wheel
    assert bundle.requirements_path == tmp_path / DumpFiles.requirements
    assert bundle.python_version is not None


def test_wheel_bundle_from_build_path_no_wheel(tmp_path):
    (tmp_path / DumpFiles.requirements).write_text("requests==2.31.0\n")
    with pytest.raises(RuntimeError, match="expected exactly one"):
        WheelBundle.from_build_path(tmp_path)


# ---------------------------------------------------------------------------
# WheelPackager._write_requirements_path: sync check
# ---------------------------------------------------------------------------


def test_write_requirements_path_rejects_stale_requirements(tmp_path, monkeypatch):
    _make_pyproject(tmp_path)
    (tmp_path / UVLOCK_NAME).write_text("# lockfile")
    (tmp_path / DumpFiles.requirements).write_text("requests==2.31.0\n")
    packager = WheelPackager(tmp_path)
    monkeypatch.setattr(
        "xorq.ibis_yaml.packager.uv_export_requirements",
        lambda *a, **kw: "flask==3.0.0\n",
    )
    with pytest.raises(RuntimeError, match="does not match") as exc_info:
        packager._write_requirements_path()
    message = str(exc_info.value)
    # Error must name the remediation paths explicitly so the user can act.
    assert str(tmp_path / DumpFiles.requirements) in message
    assert "uv export" in message


# ---------------------------------------------------------------------------
# _ensure_wheel_artifacts: branching logic
# ---------------------------------------------------------------------------


def test_ensure_wheel_artifacts_skips_when_present(tmp_path):
    _make_wheel(tmp_path)
    (tmp_path / DumpFiles.requirements).write_text("requests==2.31.0\n")
    _ensure_wheel_artifacts(tmp_path)
    assert len(list(tmp_path.glob("*.whl"))) == 1


def test_ensure_wheel_artifacts_copies_missing(tmp_path, monkeypatch):
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    fake_wheel = _make_wheel(source_dir)
    fake_reqs = source_dir / DumpFiles.requirements
    fake_reqs.write_text("requests==2.31.0\n")

    class FakeBundle:
        wheel_path = fake_wheel
        requirements_path = fake_reqs

    class FakePackager:
        def __init__(self, *a, **kw):
            pass

        def build(self):
            return FakeBundle()

    monkeypatch.setattr("xorq.ibis_yaml.packager.WheelPackager", FakePackager)
    _ensure_wheel_artifacts(build_dir, project_path=tmp_path)

    assert len(list(build_dir.glob("*.whl"))) == 1
    assert (build_dir / DumpFiles.requirements).exists()


def test_ensure_wheel_artifacts_raises_clearly_when_cwd_has_no_pyproject(
    tmp_path, monkeypatch
):
    # Simulate a caller (e.g. a Jupyter kernel) whose cwd is outside any
    # project.  The default upward-walk must fail with a message that
    # names the project_path escape hatch.
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    cwd_without_pyproject = tmp_path / "elsewhere"
    cwd_without_pyproject.mkdir()
    monkeypatch.setattr(
        "xorq.ibis_yaml.packager.find_file_upwards",
        lambda *a, **kw: (_ for _ in ()).throw(
            ValueError("could not find 'pyproject.toml' in ...")
        ),
    )
    with pytest.raises(ValueError, match="project_path=") as exc_info:
        _ensure_wheel_artifacts(build_dir, project_path=None)
    assert "catalog.add" in str(exc_info.value)


# ---------------------------------------------------------------------------
# uv_export_requirements: extras arg construction
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_uv_export(monkeypatch, captured, result=None):
    result = result or _FakeResult()

    def _run(args, **kw):
        captured["args"] = tuple(args)
        return result

    monkeypatch.setattr("subprocess.run", _run)
    monkeypatch.setattr("xorq.ibis_yaml.packager._nix_env", lambda: None)


def test_uv_export_requirements_all_extras(monkeypatch):
    captured = {}
    _patch_uv_export(monkeypatch, captured)
    uv_export_requirements("/proj", ">=3.10", all_extras=True)
    assert "--all-extras" in captured["args"]


def test_uv_export_requirements_specific_extras(monkeypatch):
    captured = {}
    _patch_uv_export(monkeypatch, captured)
    uv_export_requirements("/proj", ">=3.10", extras=("pg", "redis"), all_extras=False)
    args = captured["args"]
    assert "--all-extras" not in args
    pg_idx = args.index("pg")
    assert args[pg_idx - 1] == "--extra"
    redis_idx = args.index("redis")
    assert args[redis_idx - 1] == "--extra"


def test_uv_export_requirements_no_extras(monkeypatch):
    captured = {}
    _patch_uv_export(monkeypatch, captured)
    uv_export_requirements("/proj", ">=3.10", all_extras=False)
    args = captured["args"]
    assert "--all-extras" not in args
    assert "--extra" not in args


def test_uv_export_requirements_default_is_all_extras(monkeypatch):
    captured = {}
    _patch_uv_export(monkeypatch, captured)
    uv_export_requirements("/proj", ">=3.10")
    assert "--all-extras" in captured["args"]


def test_uv_export_requirements_surfaces_stderr_on_failure(monkeypatch):
    captured = {}
    _patch_uv_export(
        monkeypatch,
        captured,
        result=_FakeResult(returncode=2, stderr="error: project has no lock"),
    )
    with pytest.raises(RuntimeError, match="uv export failed") as exc_info:
        uv_export_requirements("/proj", ">=3.10")
    assert "project has no lock" in str(exc_info.value)
