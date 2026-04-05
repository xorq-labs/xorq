import functools
import shutil
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
from xorq.ibis_yaml.packager import (
    PYPROJECT_NAME,
    REQUIREMENTS_NAME,
    UVLOCK_NAME,
    PackagedBuilder,
    PackagedRunner,
    SdistArchive,
    SdistPackager,
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
@pytest.mark.snapshot_check
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_sdist_path_hexdigest(template, tmpdir, snapshot):
    zip_path, project_path = prep_template_tmpdir(template, tmpdir)
    packager = SdistPackager(project_path)
    actual = packager.sdist_path_hexdigest
    snapshot.assert_match(actual, f"test_sdist_path_hexdigest-{template}")


@pytest.mark.slow(level=1)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_sdist_builder(template, tmpdir):
    # test that we build and inject the requirements.txt
    zip_path, project_path = prep_template_tmpdir(template, tmpdir)
    script_path = project_path.joinpath("expr.py")
    packaged_builder = PackagedBuilder(script_path=script_path, sdist_path=zip_path)
    assert packaged_builder.build_path, packaged_builder._uv_tool_run_xorq_build.stderr


@pytest.mark.slow(level=1)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_catalog_sdist_validation(template, tmpdir):
    # test that SdistArchive validates a well-formed sdist
    zip_path, project_path = prep_template_tmpdir(template, tmpdir)
    packager = SdistPackager(project_path=project_path)
    sdist_archive = SdistArchive(packager.sdist_path)
    assert sdist_archive.python_version


@pytest.mark.slow(level=1)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_catalog_sdist_rejects_incomplete(template, tmpdir):
    # test that SdistArchive raises when required members are missing
    zip_path, project_path = prep_template_tmpdir(template, tmpdir)
    packager = SdistPackager(project_path=project_path)
    # _sdist_path is the zip before ensure members run — missing uv.lock/requirements.txt
    sdist_incomplete = Path(tmpdir).joinpath("sdist_incomplete.zip")
    shutil.copy2(packager._sdist_path, sdist_incomplete)
    with pytest.raises(FileNotFoundError):
        SdistArchive(sdist_incomplete)


@pytest.mark.slow(level=2)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_sdist_runner(template, tmpdir):
    tmpdir = Path(tmpdir)
    output_path = tmpdir.joinpath("output")
    zip_path, project_path = prep_template_tmpdir(template, tmpdir)
    script_path = project_path.joinpath("expr.py")
    packaged_builder = PackagedBuilder(script_path=script_path, sdist_path=zip_path)
    packaged_runner = PackagedRunner(
        packaged_builder.build_path, output_path=str(output_path)
    )
    assert not packaged_runner.popened.popen.wait()
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


def test_sdist_packager_rejects_bad_python_version():
    with pytest.raises(ValueError, match="invalid python version"):
        SdistPackager(project_path="/tmp", python_version="garbage")


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
# SdistArchive: validates members, rejects missing
# ---------------------------------------------------------------------------


def _make_sdist_zip(tmp_path, members):
    """Helper: create a minimal sdist zip with given top-level member names."""
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in members:
            zf.writestr(f"proj-0.0.0/{name}", "content")
    return zip_path


def test_sdist_archive_accepts_complete(tmp_path):
    zip_path = _make_sdist_zip(
        tmp_path, [PYPROJECT_NAME, UVLOCK_NAME, REQUIREMENTS_NAME]
    )
    archive = SdistArchive(zip_path)
    assert archive.path == zip_path


def test_sdist_archive_rejects_missing_uvlock(tmp_path):
    zip_path = _make_sdist_zip(tmp_path, [PYPROJECT_NAME, REQUIREMENTS_NAME])
    with pytest.raises(FileNotFoundError, match=UVLOCK_NAME):
        SdistArchive(zip_path)


def test_sdist_archive_rejects_missing_multiple(tmp_path):
    zip_path = _make_sdist_zip(tmp_path, [PYPROJECT_NAME])
    with pytest.raises(FileNotFoundError, match=UVLOCK_NAME) as exc_info:
        SdistArchive(zip_path)
    # both missing members mentioned in one error
    assert REQUIREMENTS_NAME in str(exc_info.value)


def test_sdist_archive_rejects_nonexistent_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="sdist not found"):
        SdistArchive(tmp_path / "does_not_exist.zip")


def test_sdist_archive_extract_requirements_to(tmp_path):
    zip_path = _make_sdist_zip(
        tmp_path, [PYPROJECT_NAME, UVLOCK_NAME, REQUIREMENTS_NAME]
    )
    archive = SdistArchive(zip_path)
    dest_dir = tmp_path / "extract"
    dest_dir.mkdir()
    result = archive.extract_requirements_to(dest_dir)
    assert result.exists()
    assert result.read_text() == "content"
