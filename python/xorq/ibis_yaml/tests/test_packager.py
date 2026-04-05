import functools
import shutil
import sys
import tempfile
import zipfile
from pathlib import (
    Path,
)

import pytest

from xorq.common.utils.download_utils import (
    download_xorq_template,
)
from xorq.ibis_yaml.packager import (
    PackagedBuilder,
    PackagedRunner,
    SdistArchive,
    SdistPackager,
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
