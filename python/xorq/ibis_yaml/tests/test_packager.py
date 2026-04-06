import functools
import sys
import tempfile
import zipfile
from pathlib import (
    Path,
)

import pytest
import tomlkit

from xorq.common.utils.download_utils import (
    download_xorq_template,
)
from xorq.common.utils.zip_utils import (
    ZipProxy,
    append_toplevel,
)
from xorq.ibis_yaml.enums import DumpFiles
from xorq.ibis_yaml.packager import (
    PackagedBuilder,
    PackagedRunner,
    WheelPackager,
    _validate_python_version,
    generate_pyproject_toml,
    parse_requirements,
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
    return (zip_path, project_path)


@pytest.mark.slow(level=1)
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_wheel_packager(template, tmpdir):
    zip_path, project_path = prep_template_tmpdir(template, tmpdir)
    packager = WheelPackager(project_path)
    assert packager.wheel_path.exists()
    assert packager.wheel_path.suffix == ".whl"
    assert packager.requirements_path.exists()


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
        wheel_path=packager.wheel_path,
        requirements_path=packager.requirements_path,
        python_version=packager.python_version,
        maybe_packager=packager,
    )
    assert packaged_builder.build_path, packaged_builder._uv_tool_run_xorq_build.stderr
    assert (packaged_builder.build_path / DumpFiles.wheel).exists()
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
        wheel_path=packager.wheel_path,
        requirements_path=packager.requirements_path,
        python_version=packager.python_version,
        maybe_packager=packager,
    )
    packaged_runner = PackagedRunner(
        packaged_builder.build_path, output_path=str(output_path)
    )
    assert packaged_runner.popened.returncode == 0
    assert output_path.exists()


# ---------------------------------------------------------------------------
# Unit tests for pure helpers (no subprocess / network)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # simple pinned deps
        ("requests==2.31.0\nflask==3.0.0\n", ["requests==2.31.0", "flask==3.0.0"]),
        # blank lines and comments
        (
            "# this is a comment\nrequests==2.31.0\n\n# another\nflask>=3\n",
            ["requests==2.31.0", "flask>=3"],
        ),
        # inline hashes
        (
            "requests==2.31.0 --hash=sha256:abc123 --hash=sha256:def456\n",
            ["requests==2.31.0"],
        ),
        # option lines (-i, --index-url, etc.)
        (
            "-i https://pypi.org/simple\n--extra-index-url https://foo\nrequests\n",
            ["requests"],
        ),
        # backslash continuations
        ("requests==2.31.0 \\\n", ["requests==2.31.0"]),
        # inline comment after dep
        ("requests==2.31.0  # pinned\n", ["requests==2.31.0"]),
        # empty input
        ("", []),
        ("\n\n# only comments\n", []),
    ],
    ids=[
        "simple",
        "comments_and_blanks",
        "inline_hashes",
        "option_lines",
        "backslash_continuation",
        "inline_comment",
        "empty",
        "only_comments",
    ],
)
def test_parse_requirements(text, expected):
    assert parse_requirements(text) == expected


def test_generate_pyproject_toml_structure():
    deps = ["requests>=2.31", "flask>=3.0"]
    text = generate_pyproject_toml("my-project", deps, requires_python=">=3.11")
    data = tomlkit.loads(text)
    assert data["build-system"]["requires"] == ["hatchling"]
    assert data["build-system"]["build-backend"] == "hatchling.build"
    assert data["project"]["name"] == "my-project"
    assert data["project"]["version"] == "0.0.0"
    assert data["project"]["requires-python"] == ">=3.11"
    assert list(data["project"]["dependencies"]) == deps


def test_generate_pyproject_toml_empty_deps():
    text = generate_pyproject_toml("empty-proj", [])
    data = tomlkit.loads(text)
    assert list(data["project"]["dependencies"]) == []


@pytest.mark.parametrize(
    "value",
    ["3.11", "3.10", "3.13.1"],
)
def test_validate_python_version_accepts_valid(value):
    # should not raise — use a dummy attrs instance
    _validate_python_version(None, None, value)


@pytest.mark.parametrize(
    "value",
    ["not.a.version", "abc", "3.10.x"],
)
def test_validate_python_version_rejects_invalid(value):
    with pytest.raises(ValueError, match="invalid python version"):
        _validate_python_version(None, None, value)


def test_validate_python_version_accepts_none():
    _validate_python_version(None, None, None)


def test_wheel_packager_rejects_bad_python_version():
    with pytest.raises(ValueError, match="invalid python version"):
        WheelPackager(project_path="/tmp", python_version="garbage")


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


def test_packaged_builder_rejects_missing_wheel(tmp_path):
    with pytest.raises(FileNotFoundError, match="wheel not found"):
        PackagedBuilder(
            script_path=tmp_path / "script.py",
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
    with pytest.raises(FileNotFoundError, match="wheel not found"):
        PackagedRunner(build_path=build_dir)
