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
    SdistBuilder,
    Sdister,
    SdistRunner,
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
    sdister = Sdister(project_path)
    actual = sdister.sdist_path_hexdigest
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
    sdist_builder = SdistBuilder(script_path=script_path, sdist_path=zip_path)
    assert sdist_builder.build_path, sdist_builder._uv_tool_run_xorq_build.stderr


@pytest.mark.xfail(reason="depends on release with unique_key optional")
@pytest.mark.slow(level=1)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_sdist_builder_no_requirements(template, tmpdir):
    # test that we build and inject the requirements.txt
    zip_path, project_path = prep_template_tmpdir(template, tmpdir)
    script_path = project_path.joinpath("expr.py")
    requirements_path = project_path.joinpath("requirements.txt")
    requirements_path.unlink()
    sdist_builder = SdistBuilder.from_script_path(
        script_path=script_path, project_path=project_path, require_requirements=False
    )
    assert sdist_builder.build_path, sdist_builder._uv_tool_run_xorq_build.stderr


@pytest.mark.slow(level=1)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_sdist_builder_no_requirements_fails(template, tmpdir):
    # test that SdistBuilder raises when requirements.txt is missing from sdist
    zip_path, project_path = prep_template_tmpdir(template, tmpdir)
    script_path = project_path.joinpath("expr.py")
    requirements_path = project_path.joinpath("requirements.txt")
    requirements_path.unlink()
    #
    sdister = Sdister(project_path=project_path)
    # _sdist_path is the zip before ensure_requirements_member runs;
    # copy it so we have a zip without requirements.txt
    sdist_no_reqs = Path(tmpdir).joinpath("sdist_no_reqs.zip")
    shutil.copy2(sdister._sdist_path, sdist_no_reqs)
    #
    with pytest.raises(AssertionError):
        sdist_builder = SdistBuilder(
            script_path=script_path,
            sdist_path=sdist_no_reqs,
            require_requirements=True,
        )
        sdist_builder


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
    sdist_builder = SdistBuilder(script_path=script_path, sdist_path=zip_path)
    args = (
        "xorq",
        "run",
        "--output-path",
        str(output_path),
        str(sdist_builder.build_path),
    )
    sdist_runner = SdistRunner(sdist_builder.build_path, args=args)
    assert not sdist_runner.popened.popen.wait()
    assert sdist_runner.popened.returncode == 0
    assert output_path.exists()
