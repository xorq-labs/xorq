import functools
from pathlib import Path
from subprocess import PIPE
from tempfile import TemporaryDirectory

import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    deep_iterable,
    instance_of,
    optional,
)

from xorq.common.utils.process_utils import (
    Popened,
    assert_not_in_nix_shell,
)
from xorq.common.utils.tar_utils import (
    TGZAppender,
    TGZProxy,
    calc_tgz_content_hexdigest,
    copy_path,
)


REQUIREMENTS_NAME = "requirements.txt"
PYPROJECT_NAME = "pyproject.toml"
BUILD_SDIST_NAME = "sdist.tar.gz"


@frozen
class Sdister:
    project_path = field(validator=instance_of(Path), converter=Path)

    def __attrs_post_init__(self):
        assert self.project_path.exists()
        assert self.pyproject_path.exists()

    @property
    def pyproject_path(self):
        return self.project_path.joinpath(PYPROJECT_NAME)

    @property
    @functools.cache
    def _tmpdir(self):
        return TemporaryDirectory()

    @property
    def tmpdir(self):
        return Path(self._tmpdir.name)

    @property
    @functools.cache
    def _uv_build_popened(self):
        args = (
            "uv",
            "build",
            "--sdist",
            "--out-dir",
            str(self.tmpdir),
            str(self.pyproject_path.parent),
        )
        popened = Popened(args)
        return popened

    popened = _uv_build_popened

    @property
    def _sdist_path(self):
        prefix = "Successfully built "
        (_, line) = self._uv_build_popened.stderr.strip().rsplit("\n", 1)
        (first, rest) = (line[: len(prefix)], line[len(prefix) :])
        assert first == prefix
        sdist_path = Path(rest)
        return sdist_path

    def ensure_requirements_member(self):
        sdist_path = self._sdist_path
        if not TGZProxy(sdist_path).toplevel_name_exists(REQUIREMENTS_NAME):
            requirements_path = self.tmpdir.joinpath(REQUIREMENTS_NAME)
            requirements_text = uv_tool_run_uv_pip_freeze_package_path(sdist_path)
            requirements_path.write_text(requirements_text)
            TGZAppender.append_toplevel(sdist_path, requirements_path)

    @property
    @functools.cache
    def sdist_path(self):
        self.ensure_requirements_member()
        return self._sdist_path

    @property
    def sdist_path_hexdigest(self):
        return calc_tgz_content_hexdigest(self.sdist_path)

    @classmethod
    def from_script_path(cls, script_path):
        pyproject_path = find_file_upwards(script_path, PYPROJECT_NAME)
        return cls(pyproject_path.parent)


@frozen
class SdistBuilder:
    script_path = field(validator=instance_of(Path), converter=Path)
    sdist_path = field(validator=instance_of(Path), converter=Path)
    args = field(
        validator=deep_iterable(instance_of(str), instance_of(tuple)), default=()
    )
    maybe_packager = field(
        validator=optional(instance_of(Path)),
        converter=toolz.curried.excepts(Exception, Path),
        default=None,
    )
    require_requirements = field(validator=instance_of(bool), default=True)

    def __attrs_post_init__(self):
        if self.require_requirements:
            assert TGZProxy(self.sdist_path).toplevel_name_exists(REQUIREMENTS_NAME)
        self.ensure_requirements_path()
        assert self.requirements_path.exists()

    @property
    @functools.cache
    def _tmpdir(self):
        return TemporaryDirectory()

    @property
    def tmpdir(self):
        return Path(self._tmpdir.name)

    @property
    def requirements_path(self):
        return self.tmpdir.joinpath(REQUIREMENTS_NAME)

    def ensure_requirements_path(self):
        if not self.requirements_path.exists():
            if TGZProxy(self.sdist_path).toplevel_name_exists(REQUIREMENTS_NAME):
                TGZProxy(self.sdist_path).extract_toplevel_name(
                    REQUIREMENTS_NAME, self.requirements_path
                )
            else:
                requirements_text = uv_tool_run_uv_pip_freeze_package_path(
                    self.sdist_path
                )
                self.requirements_path.write_text(requirements_text)

    @property
    @functools.cache
    def _uv_tool_run_xorq_build(self):
        args = self.args if self.args else ("xorq", "build", str(self.script_path))
        popened = uv_tool_run(
            *args, with_=self.sdist_path, with_requirements=self.requirements_path
        )
        return popened

    popened = _uv_tool_run_xorq_build

    def get_build_path(self):
        # FIXME: don't capture stdout so user can still use --pdb
        return Path(self._uv_tool_run_xorq_build.stdout.strip())

    @functools.cache
    def copy_sdist(self):
        target = self.get_build_path().joinpath(BUILD_SDIST_NAME)
        copy_path(self.sdist_path, target)
        return target

    @property
    def build_path(self):
        self.copy_sdist()
        return self.get_build_path()

    @classmethod
    def from_script_path(cls, script_path, project_path=None, **kwargs):
        packager = (
            Sdister(project_path)
            if project_path
            else Sdister.from_script_path(script_path)
        )
        return cls(
            script_path=script_path,
            sdist_path=packager.sdist_path,
            maybe_packager=packager,
            **kwargs,
        )


@frozen
class SdistRunner:
    build_path = field(validator=instance_of(Path), converter=Path)
    args = field(
        validator=deep_iterable(instance_of(str), instance_of(tuple)), default=()
    )

    def __attrs_post_init__(self):
        assert self.build_path.exists()
        assert self.sdist_path.exists()
        # we ALWAYS require requirements.txt to run
        assert TGZProxy(self.sdist_path).toplevel_name_exists(REQUIREMENTS_NAME)

    @property
    def sdist_path(self):
        return self.build_path.joinpath(BUILD_SDIST_NAME)

    @property
    @functools.cache
    def _tmpdir(self):
        return TemporaryDirectory()

    @property
    def tmpdir(self):
        return Path(self._tmpdir.name)

    @property
    def requirements_path(self):
        return self.tmpdir.joinpath(REQUIREMENTS_NAME)

    def ensure_requirements_path(self):
        if not self.requirements_path.exists():
            if TGZProxy(self.sdist_path).toplevel_name_exists(REQUIREMENTS_NAME):
                TGZProxy(self.sdist_path).extract_toplevel_name(
                    REQUIREMENTS_NAME, self.requirements_path
                )
            else:
                requirements_text = uv_tool_run_uv_pip_freeze_package_path(
                    self.sdist_path
                )
                self.requirements_path.write_text(requirements_text)

    @property
    @functools.cache
    def _uv_tool_run_xorq_run(self):
        self.ensure_requirements_path()
        args = self.args if self.args else ("xorq", "run", str(self.build_path))
        # FIXME: enable streaming output
        popened = uv_tool_run(
            *args,
            with_=self.sdist_path,
            with_requirements=self.requirements_path,
            capturing=False,
        )
        return popened

    popened = _uv_tool_run_xorq_run


def find_file_upwards(start, name):
    path = Path(start).absolute()
    if path.is_file():
        path = path.parent
    paths = (p.joinpath(name) for p in (path, *path.parents))
    found = next((p for p in paths if p.exists()), None)
    if not found:
        raise ValueError
    return found


def uv_tool_run(
    *args, isolated=True, with_=None, with_requirements=None, check=True, capturing=True
):
    assert_not_in_nix_shell()
    command_v_xorq = Popened.check_output("command -v xorq", shell=True).strip()
    args = tuple(el if el != command_v_xorq else "xorq" for el in args)
    popened_args = (
        "uv",
        "tool",
        "run",
        *(("--isolated",) if isolated else ()),
        *(("--with", str(with_)) if with_ else ()),
        *(("--with-requirements", str(with_requirements)) if with_requirements else ()),
        *args,
    )
    kwargs_tuple = (
        (("stdout", PIPE), ("stderr", PIPE))
        if capturing
        else (("stdout", None), ("stderr", None))
    )
    popened = Popened(popened_args, kwargs_tuple=kwargs_tuple)
    if check:
        assert not popened.returncode, popened.stderr
    return popened


def uv_tool_run_uv_pip_freeze(sdist_path):
    # don't use "uv pip freeze": causes issues with nix and enclosing venv
    popened = uv_tool_run("pip", "freeze", with_=sdist_path)
    stdout = popened.stdout
    return stdout


def uv_tool_run_uv_pip_freeze_package_path(package_path):
    stdout = uv_tool_run_uv_pip_freeze(package_path)
    splat = stdout.split("\n")
    filtered = (el for el in splat if "file:///" not in el)
    joined = "\n".join(filtered)
    return joined
