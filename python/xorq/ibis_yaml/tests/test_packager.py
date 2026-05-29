import functools
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import (
    Path,
)

import pytest

from xorq.catalog.catalog import _ensure_wheel_artifacts
from xorq.cli_constants import OutputFormats
from xorq.common.utils.download_utils import (
    download_xorq_template,
)
from xorq.common.utils.zip_utils import (
    ZipProxy,
    append_toplevel,
)
from xorq.config import (
    _default_use_hardlink,
    env_config,
    options,
)
from xorq.ibis_yaml.enums import DumpFiles
from xorq.ibis_yaml.packager import (
    PYPROJECT_NAME,
    UVLOCK_NAME,
    JointBundle,
    PackagedBuilder,
    PackagedCachedRunner,
    PackagedRunner,
    PackagedUnboundRunner,
    UvToolRunError,
    WheelBundle,
    WheelPackager,
    _convert_output_format,
    _link_mode_args,
    _nix_env,
    _read_requires_python,
    _validate_python_version,
    find_file_upwards,
    uv_export_requirements,
    uv_tool_run,
    validate_params_early,
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
@pytest.mark.parametrize(
    "template",
    [
        pytest.param(
            t,
            marks=pytest.mark.xfail(
                t == InitTemplates.sklearn,
                reason=(
                    "sklearn template pins released xorq in requirements.txt; "
                    "the isolated uv runner resolves xorq-dasher-unaware xorq "
                    "from PyPI, causing a build-format mismatch"
                ),
                strict=False,
            ),
        )
        for t in InitTemplates
    ],
)
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
    with pytest.raises(FileNotFoundError, match="no .whl files found"):
        PackagedRunner(build_path=build_dir)


def test_packaged_runner_rejects_missing_requirements(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    _make_wheel(build_dir)
    with pytest.raises(FileNotFoundError, match="invalid build path"):
        PackagedRunner(build_path=build_dir)


def test_packaged_cached_runner_rejects_missing_build_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="build path does not exist"):
        PackagedCachedRunner(build_path=tmp_path / "nonexistent")


def test_packaged_cached_runner_rejects_missing_wheel(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / DumpFiles.requirements).write_text("requests==2.31.0")
    with pytest.raises(FileNotFoundError, match="no .whl files found"):
        PackagedCachedRunner(build_path=build_dir)


def test_packaged_unbound_runner_rejects_missing_build_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="build path does not exist"):
        PackagedUnboundRunner(build_path=tmp_path / "nonexistent")


def test_packaged_unbound_runner_rejects_missing_wheel(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / DumpFiles.requirements).write_text("requests==2.31.0")
    with pytest.raises(FileNotFoundError, match="no .whl files found"):
        PackagedUnboundRunner(build_path=build_dir)


_HELP_WITH_EMIT = (
    "Usage: xorq build [OPTIONS] SCRIPT_PATH\n  --emit-build-path-to TEXT\n"
)


def test_packaged_builder_raises_when_emit_file_missing(tmp_path, monkeypatch):
    """If xorq build returns successfully but didn't write the emit file,
    surface a clear error instead of silently falling back to stdout."""
    wheel = _make_wheel(tmp_path)
    requirements = tmp_path / DumpFiles.requirements
    requirements.write_text("requests==2.31.0")
    script = tmp_path / "script.py"
    script.write_text("expr = None\n")

    bundle = WheelBundle(wheel_path=wheel, requirements_path=requirements)

    def fake_uv_tool_run(*args, **kwargs):
        if "--help" in args:
            return subprocess.CompletedProcess(
                args=(), returncode=0, stdout=_HELP_WITH_EMIT, stderr=""
            )
        return subprocess.CompletedProcess(
            args=(), returncode=0, stdout="", stderr="some-stderr"
        )

    monkeypatch.setattr("xorq.ibis_yaml.packager.uv_tool_run", fake_uv_tool_run)

    builder = PackagedBuilder(script_path=script, bundle=bundle)
    with pytest.raises(RuntimeError, match="did not write build path"):
        builder.build()


def test_packaged_builder_raises_when_emit_file_empty(tmp_path, monkeypatch):
    """If xorq build wrote an empty emit file, surface a clear error."""
    wheel = _make_wheel(tmp_path)
    requirements = tmp_path / DumpFiles.requirements
    requirements.write_text("requests==2.31.0")
    script = tmp_path / "script.py"
    script.write_text("expr = None\n")

    bundle = WheelBundle(wheel_path=wheel, requirements_path=requirements)

    def fake_uv_tool_run(*args, **kwargs):
        if "--help" in args:
            return subprocess.CompletedProcess(
                args=(), returncode=0, stdout=_HELP_WITH_EMIT, stderr=""
            )
        emit_path = Path(args[args.index("--emit-build-path-to") + 1])
        emit_path.write_text("")
        return subprocess.CompletedProcess(args=(), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xorq.ibis_yaml.packager.uv_tool_run", fake_uv_tool_run)

    builder = PackagedBuilder(script_path=script, bundle=bundle)
    with pytest.raises(RuntimeError, match="empty build path"):
        builder.build()


def test_packaged_builder_falls_back_when_inner_xorq_lacks_flag(tmp_path, monkeypatch):
    """If the inner xorq (resolved by uv tool run from requirements) predates
    --emit-build-path-to, the packager skips the flag and parses the build
    path from stdout. This is required while the published xorq on PyPI is
    older than the local CLI."""
    wheel = _make_wheel(tmp_path)
    requirements = tmp_path / DumpFiles.requirements
    requirements.write_text("requests==2.31.0")
    script = tmp_path / "script.py"
    script.write_text("expr = None\n")
    target_build = tmp_path / "builds" / "abc"
    target_build.mkdir(parents=True)

    bundle = WheelBundle(wheel_path=wheel, requirements_path=requirements)

    calls = []

    def fake_uv_tool_run(*args, **kwargs):
        calls.append(args)
        if "--help" in args:
            return subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout="Usage: xorq build [OPTIONS] SCRIPT_PATH\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=(), returncode=0, stdout=f"{target_build}\n", stderr=""
        )

    monkeypatch.setattr("xorq.ibis_yaml.packager.uv_tool_run", fake_uv_tool_run)

    builder = PackagedBuilder(script_path=script, bundle=bundle)
    builder.build()
    assert builder.build_path == target_build
    assert len(calls) == 2
    assert "--help" in calls[0]
    assert "--emit-build-path-to" not in calls[1]


def test_packaged_builder_does_not_fall_back_when_flag_supported(tmp_path, monkeypatch):
    """When the --help probe reports --emit-build-path-to, the builder
    uses the flag instead of parsing stdout."""
    wheel = _make_wheel(tmp_path)
    requirements = tmp_path / DumpFiles.requirements
    requirements.write_text("requests==2.31.0")
    script = tmp_path / "script.py"
    script.write_text("expr = None\n")
    target_build = tmp_path / "builds" / "abc"
    target_build.mkdir(parents=True)

    bundle = WheelBundle(wheel_path=wheel, requirements_path=requirements)

    calls = []

    def fake_uv_tool_run(*args, **kwargs):
        calls.append(args)
        if "--help" in args:
            return subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout=(
                    "Usage: xorq build [OPTIONS] SCRIPT_PATH\n"
                    "  --emit-build-path-to TEXT\n"
                ),
                stderr="",
            )
        emit_idx = args.index("--emit-build-path-to")
        Path(args[emit_idx + 1]).write_text(str(target_build))
        return subprocess.CompletedProcess(args=(), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xorq.ibis_yaml.packager.uv_tool_run", fake_uv_tool_run)

    builder = PackagedBuilder(script_path=script, bundle=bundle)
    builder.build()
    assert builder.build_path == target_build
    assert len(calls) == 2
    assert "--help" in calls[0]
    assert "--emit-build-path-to" in calls[1]


def test_packaged_builder_propagates_build_failure(tmp_path, monkeypatch):
    """A build failure must propagate regardless of the capability check."""
    wheel = _make_wheel(tmp_path)
    requirements = tmp_path / DumpFiles.requirements
    requirements.write_text("requests==2.31.0")
    script = tmp_path / "script.py"
    script.write_text("expr = None\n")

    bundle = WheelBundle(wheel_path=wheel, requirements_path=requirements)

    def fake_uv_tool_run(*args, **kwargs):
        if "--help" in args:
            return subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout=(
                    "Usage: xorq build [OPTIONS] SCRIPT_PATH\n"
                    "  --emit-build-path-to TEXT\n"
                ),
                stderr="",
            )
        raise UvToolRunError(
            returncode=2,
            cmd=("uv", "tool", "run", *args),
            output="",
            stderr="Error: expression 'expr' not found\n",
        )

    monkeypatch.setattr("xorq.ibis_yaml.packager.uv_tool_run", fake_uv_tool_run)

    builder = PackagedBuilder(script_path=script, bundle=bundle)
    with pytest.raises(UvToolRunError):
        builder.build()


# ---------------------------------------------------------------------------
# validate_params_early
# ---------------------------------------------------------------------------


def _make_expr_metadata(directory, params=()):
    """Write a minimal expr_metadata.json."""
    metadata = {"params": [{"param_name": p, "type": "str"} for p in params]}
    path = Path(directory) / DumpFiles.expr_metadata
    path.write_text(json.dumps(metadata))
    return path


def test_validate_params_early_passes_valid(tmp_path):
    _make_expr_metadata(tmp_path, params=("threshold", "name"))
    validate_params_early(tmp_path, ("threshold=0.5", "name=foo"))


def test_validate_params_early_skips_when_no_params_supplied(tmp_path):
    _make_expr_metadata(tmp_path, params=("threshold",))
    validate_params_early(tmp_path, ())


def test_validate_params_early_rejects_unknown_param(tmp_path):
    _make_expr_metadata(tmp_path, params=("threshold",))
    with pytest.raises(ValueError, match="Unknown parameter 'bogus'"):
        validate_params_early(tmp_path, ("bogus=1",))


def test_validate_params_early_rejects_when_no_params_declared(tmp_path):
    _make_expr_metadata(tmp_path, params=())
    with pytest.raises(ValueError, match="declares no parameters"):
        validate_params_early(tmp_path, ("x=1",))


def test_validate_params_early_rejects_malformed_kv(tmp_path):
    _make_expr_metadata(tmp_path, params=("threshold",))
    with pytest.raises(ValueError, match="Expected key=value"):
        validate_params_early(tmp_path, ("noequalssign",))


def test_validate_params_early_rejects_missing_metadata(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        validate_params_early(tmp_path, ("x=1",))


def test_validate_params_early_malformed_and_no_params_declared(tmp_path):
    _make_expr_metadata(tmp_path, params=())
    with pytest.raises(ValueError) as exc_info:
        validate_params_early(tmp_path, ("noequalssign", "x=1"))
    msg = str(exc_info.value)
    assert "Expected key=value" in msg
    assert "declares no parameters" in msg


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
# uv --link-mode propagation (issue #1942)
# ---------------------------------------------------------------------------


def _patch_subprocess_run(monkeypatch):
    """Replace packager.subprocess.run with a stub recording the last args."""
    captured = {"args": None, "calls": 0}

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["calls"] += 1
        return _Result()

    monkeypatch.setattr("xorq.ibis_yaml.packager.subprocess.run", fake_run)
    return captured


@pytest.mark.parametrize(
    ("platform", "env_value", "expected_args"),
    [
        ("darwin", "", ("--link-mode", "hardlink")),
        ("linux", "", ()),
        ("darwin", "False", ()),
        ("linux", "True", ("--link-mode", "hardlink")),
    ],
)
def test_link_mode_args_from_platform_and_env(
    monkeypatch, platform, env_value, expected_args
):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(
        "xorq.config.env_config", env_config.clone(XORQ_UV_USE_HARDLINK=env_value)
    )
    monkeypatch.setattr(options.uv, "use_hardlink", _default_use_hardlink())
    assert _link_mode_args() == expected_args


@pytest.mark.parametrize(
    ("platform", "env_value", "expected"),
    [
        # No env override → sys.platform decides.
        ("darwin", "", True),
        ("linux", "", False),
        # Env override beats sys.platform default.
        ("darwin", "False", False),
        ("linux", "True", True),
    ],
)
def test_default_use_hardlink(monkeypatch, platform, env_value, expected):
    """``sys.platform`` and ``env_config.XORQ_UV_USE_HARDLINK`` are
    monkeypatched (pytest restores both on teardown), so each parametrize
    case exercises the function end-to-end with no args."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(
        "xorq.config.env_config", env_config.clone(XORQ_UV_USE_HARDLINK=env_value)
    )
    assert _default_use_hardlink() is expected


@pytest.mark.parametrize("use_hardlink", [True, False])
def test_uv_export_requirements_omits_link_mode(tmp_path, monkeypatch, use_hardlink):
    """uv export reads the lockfile only; ``--link-mode`` would be confused
    intent even when harmless. Guard against future refactors that splice
    a shared args builder into uv_export_requirements."""
    monkeypatch.setattr(options.uv, "use_hardlink", use_hardlink)
    captured = _patch_subprocess_run(monkeypatch)
    uv_export_requirements(tmp_path, "3.12")
    assert "--link-mode" not in captured["args"]


@pytest.mark.parametrize(
    ("use_hardlink", "flag_present"),
    [(True, True), (False, False)],
)
def test_uv_tool_run_propagates_link_mode(monkeypatch, use_hardlink, flag_present):
    monkeypatch.setattr(options.uv, "use_hardlink", use_hardlink)
    captured = _patch_subprocess_run(monkeypatch)
    uv_tool_run("xorq", "--version", capture_output=False)
    args = captured["args"]
    assert ("--link-mode" in args) is flag_present
    if flag_present:
        assert args[args.index("--link-mode") + 1] == "hardlink"


@pytest.mark.parametrize(
    ("use_hardlink", "flag_present"),
    [(True, True), (False, False)],
)
def test_wheel_packager_build_wheel_propagates_link_mode(
    tmp_path, monkeypatch, use_hardlink, flag_present
):
    monkeypatch.setattr(options.uv, "use_hardlink", use_hardlink)
    _make_pyproject(tmp_path)
    (tmp_path / DumpFiles.requirements).write_text("requests==2.31.0\n")
    captured = _patch_subprocess_run(monkeypatch)
    WheelPackager(tmp_path)._build_wheel()
    args = captured["args"]
    assert ("--link-mode" in args) is flag_present
    if flag_present:
        assert args[args.index("--link-mode") + 1] == "hardlink"


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


# ---------------------------------------------------------------------------
# uv_tool_run: error surfacing
# ---------------------------------------------------------------------------


def test_uv_tool_run_surfaces_stderr_and_stdout_on_failure(monkeypatch):
    def _raise(args, **kw):
        raise subprocess.CalledProcessError(
            1, args, output="build output\n", stderr="resolver error\n"
        )

    monkeypatch.setattr("xorq.ibis_yaml.packager.subprocess.run", _raise)
    with pytest.raises(UvToolRunError) as exc_info:
        uv_tool_run("xorq", "--version", capture_output=False)
    err = exc_info.value
    assert isinstance(err, subprocess.CalledProcessError)
    assert err.returncode == 1
    assert err.cmd is not None
    msg = str(err)
    assert "stderr:" in msg
    assert "resolver error" in msg
    assert "stdout:" in msg
    assert "build output" in msg


def test_uv_tool_run_error_omits_empty_streams(monkeypatch):
    def _raise(args, **kw):
        raise subprocess.CalledProcessError(2, args)

    monkeypatch.setattr("xorq.ibis_yaml.packager.subprocess.run", _raise)
    with pytest.raises(UvToolRunError) as exc_info:
        uv_tool_run("xorq", "--version", capture_output=False)
    msg = str(exc_info.value)
    assert "stderr:" not in msg
    assert "stdout:" not in msg
    assert "exit status 2" in msg


# ---------------------------------------------------------------------------
# JointBundle
# ---------------------------------------------------------------------------


def _make_named_wheel(directory, name, version="0.0.0", requires_python=">=3.10"):
    """Create a minimal but uv-acceptable .whl (METADATA + WHEEL + RECORD)."""
    wheel_path = Path(directory) / f"{name}-{version}-py3-none-any.whl"
    dist_info = f"{name}-{version}.dist-info"
    metadata = (
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        f"Requires-Python: {requires_python}\n"
    )
    wheel_meta = "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    record = f"{dist_info}/METADATA,,\n{dist_info}/WHEEL,,\n{dist_info}/RECORD,,\n"
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr(f"{dist_info}/METADATA", metadata)
        zf.writestr(f"{dist_info}/WHEEL", wheel_meta)
        zf.writestr(f"{dist_info}/RECORD", record)
    return wheel_path


def test_joint_bundle_from_build_path_multi(tmp_path):
    a = _make_named_wheel(tmp_path, "alpha")
    b = _make_named_wheel(tmp_path, "beta")
    (tmp_path / DumpFiles.requirements).write_text("requests==2.31.0\n")
    bundle = JointBundle.from_build_path(tmp_path)
    assert set(bundle.wheel_paths) == {a, b}
    assert bundle.requirements_path == tmp_path / DumpFiles.requirements


def test_joint_bundle_from_build_path_no_wheels(tmp_path):
    (tmp_path / DumpFiles.requirements).write_text("requests==2.31.0\n")
    with pytest.raises(RuntimeError, match="no .whl files found"):
        JointBundle.from_build_path(tmp_path)


def test_from_build_path_pins_python_from_build_metadata(tmp_path):
    """Both bundles must surface the build's Python minor as `==X.Y.*`
    when build_metadata.json carries sys-version_info, so cloudpickled
    UDFs don't get unpickled under a newer interpreter than they were
    built on."""
    _make_wheel(tmp_path)
    (tmp_path / DumpFiles.requirements).write_text("requests==2.31.0\n")
    (tmp_path / DumpFiles.build_metadata).write_text(
        json.dumps({"sys-version_info": [3, 12, 11, "final", 0]})
    )
    assert WheelBundle.from_build_path(tmp_path).python_version == "==3.12.*"

    multi_dir = tmp_path / "multi"
    multi_dir.mkdir()
    _make_named_wheel(multi_dir, "alpha")
    _make_named_wheel(multi_dir, "beta")
    (multi_dir / DumpFiles.requirements).write_text("requests==2.31.0\n")
    (multi_dir / DumpFiles.build_metadata).write_text(
        json.dumps({"sys-version_info": [3, 11, 9, "final", 0]})
    )
    assert JointBundle.from_build_path(multi_dir).python_version == "==3.11.*"


def test_from_build_path_falls_back_when_metadata_missing(tmp_path):
    """No build_metadata.json (older archives) → fall back to the
    wheel's Requires-Python intersection. Don't error."""
    _make_wheel(tmp_path)
    (tmp_path / DumpFiles.requirements).write_text("requests==2.31.0\n")
    bundle = WheelBundle.from_build_path(tmp_path)
    assert bundle.python_version is not None
    assert "==" not in bundle.python_version  # Range from Requires-Python.


# ---------------------------------------------------------------------------
# PackagedUnboundRunner unbind flag compatibility
# ---------------------------------------------------------------------------


def test_unbound_runner_uses_hyphenated_flags_when_supported(tmp_path, monkeypatch):
    """When the inner xorq supports --to-unbind-tag (hyphenated), the runner
    should pass the hyphenated form."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / DumpFiles.requirements).write_text("requests==2.31.0")
    _make_wheel(build_dir)

    calls = []

    def fake_uv_tool_run(*args, **kwargs):
        calls.append(args)
        if "--help" in args:
            return subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout="Usage: xorq run-unbound [OPTIONS] BUILD_PATH\n  --to-unbind-tag TEXT\n",
                stderr="",
            )
        return subprocess.CompletedProcess(args=(), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xorq.ibis_yaml.packager.uv_tool_run", fake_uv_tool_run)

    runner = PackagedUnboundRunner(build_path=build_dir, to_unbind_tag="source")
    runner.run()
    run_call = calls[-1]
    assert "--to-unbind-tag" in run_call
    assert "--to_unbind_tag" not in run_call


def test_unbound_runner_falls_back_to_underscore_flags(tmp_path, monkeypatch):
    """When the inner xorq predates the hyphenated flag rename, the runner
    should pass the underscore form."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / DumpFiles.requirements).write_text("requests==2.31.0")
    _make_wheel(build_dir)

    calls = []

    def fake_uv_tool_run(*args, **kwargs):
        calls.append(args)
        if "--help" in args:
            return subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout="Usage: xorq run-unbound [OPTIONS] BUILD_PATH\n  --to_unbind_tag TEXT\n",
                stderr="",
            )
        return subprocess.CompletedProcess(args=(), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("xorq.ibis_yaml.packager.uv_tool_run", fake_uv_tool_run)

    runner = PackagedUnboundRunner(build_path=build_dir, to_unbind_tag="source")
    runner.run()
    run_call = calls[-1]
    assert "--to_unbind_tag" in run_call
    assert "--to-unbind-tag" not in run_call


@pytest.mark.parametrize("value", [f.value for f in OutputFormats])
def test_convert_output_format_valid(value):
    result = _convert_output_format(value)
    assert isinstance(result, OutputFormats)
    assert result.value == value


@pytest.mark.parametrize("value", list(OutputFormats))
def test_convert_output_format_accepts_enum_member(value):
    assert _convert_output_format(value) is value


def test_convert_output_format_invalid_raises_with_choices():
    with pytest.raises(ValueError, match="invalid output_format") as exc_info:
        _convert_output_format("invalid_format")
    assert exc_info.value.__cause__ is not None
    msg = str(exc_info.value)
    for fmt in OutputFormats:
        assert fmt.value in msg
