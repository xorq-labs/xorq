import functools
import sys
import tempfile
from pathlib import (
    Path,
)

import pytest

from xorq.cli import (
    InitTemplates,
)
from xorq.common.utils.download_utils import (
    download_xorq_template,
)
from xorq.common.utils.process_utils import (
    Popened,
)
from xorq.ibis_yaml.packager import (
    SdistBuilder,
    Sdister,
    SdistRunner,
)


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
    tgz_path = tmpdir.joinpath("template.tar.gz")
    tgz_path.write_bytes(get_template_bytes(template))
    Popened.check_output(f"tar xzvf {tgz_path} --directory {tmpdir}")
    (project_path,) = (el for el in tmpdir.iterdir() if el.name != tgz_path.name)
    return (tgz_path, project_path)


@pytest.mark.slow(level=1)
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_sdist_path_hexdigest(template, tmpdir, snapshot):
    tgz_path, project_path = prep_template_tmpdir(template, tmpdir)
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
    tgz_path, project_path = prep_template_tmpdir(template, tmpdir)
    script_path = project_path.joinpath("expr.py")
    sdist_builder = SdistBuilder(script_path=script_path, sdist_path=tgz_path)
    assert sdist_builder.build_path, sdist_builder._uv_tool_run_xorq_build.stderr


@pytest.mark.xfail(reason="depends on release with unique_key optional")
@pytest.mark.slow(level=1)
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="requirements.txt issues for python3.10"
)
@pytest.mark.parametrize("template", tuple(InitTemplates))
def test_sdist_builder_no_requirements(template, tmpdir):
    # test that we build and inject the requirements.txt
    tgz_path, project_path = prep_template_tmpdir(template, tmpdir)
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
    # test that we build and inject the requirements.txt
    tgz_path, project_path = prep_template_tmpdir(template, tmpdir)
    script_path = project_path.joinpath("expr.py")
    requirements_path = project_path.joinpath("requirements.txt")
    requirements_path.unlink()
    #
    sdister = Sdister(project_path=project_path)
    sdist_path = sdister.sdist_path
    bak_path = sdister.sdist_path.with_name(sdist_path.name + ".bak")
    #
    bak_path.rename(sdist_path)
    with pytest.raises(AssertionError):
        sdist_builder = SdistBuilder(
            script_path=script_path, sdist_path=sdist_path, require_requirements=True
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
    tgz_path, project_path = prep_template_tmpdir(template, tmpdir)
    script_path = project_path.joinpath("expr.py")
    sdist_builder = SdistBuilder(script_path=script_path, sdist_path=tgz_path)
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
