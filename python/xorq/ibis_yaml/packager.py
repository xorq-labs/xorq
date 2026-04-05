"""
Sdist-based build and run pipeline for xorq expressions.

Pipeline: Sdister → SdistBuilder → SdistRunner

Sdister      project directory → sdist zip (via `uv build --sdist`),
             with requirements.txt embedded.

SdistBuilder sdist zip + script → build directory (via `uv tool run xorq build`),
             containing the serialized expression and a copy of the sdist.

SdistRunner  build directory → execution output (via `uv tool run xorq run`),
             in the sdist's isolated environment.
"""

import functools
import subprocess
from pathlib import Path
from subprocess import PIPE
from tempfile import TemporaryDirectory
from typing import Iterable

import tomlkit
import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from xorq.common.utils.process_utils import (
    Popened,
    in_nix_shell,
)
from xorq.common.utils.zip_utils import (
    ZipAppender,
    ZipProxy,
    calc_zip_content_hexdigest,
    copy_path,
    tgz_to_zip,
)


REQUIREMENTS_NAME = "requirements.txt"
PYPROJECT_NAME = "pyproject.toml"
BUILD_SDIST_NAME = "sdist.zip"


def _validate_python_version(instance, attribute, value):
    if value is None:
        return
    try:
        Version(value)
    except Exception as e:
        raise ValueError(f"invalid python version: {value!r}") from e


def resolve_python_version(path):
    """Return the highest Python version acceptable per requires-python in pyproject.toml."""
    return get_acceptable_python_versions(path)[-1]


@frozen
class Sdister:
    project_path = field(validator=instance_of(Path), converter=Path)
    python_version = field(validator=_validate_python_version, default=None)

    def __attrs_post_init__(self):
        if not self.project_path.exists():
            raise FileNotFoundError(f"project path does not exist: {self.project_path}")
        if not self.pyproject_path.exists():
            raise FileNotFoundError(
                f"pyproject.toml not found at: {self.pyproject_path}"
            )
        if self.python_version is None:
            object.__setattr__(
                self, "python_version", resolve_python_version(self.pyproject_path)
            )

    @property
    def pyproject_path(self):
        return self.project_path.joinpath(PYPROJECT_NAME)

    @functools.cached_property
    def _tmpdir(self):
        return TemporaryDirectory()

    @property
    def tmpdir(self):
        return Path(self._tmpdir.name)

    @functools.cached_property
    def _uv_build_popened(self):
        args = (
            "uv",
            "build",
            "--sdist",
            "--python",
            self.python_version,
            "--out-dir",
            str(self.tmpdir),
            str(self.pyproject_path.parent),
        )
        popened = Popened(args)
        return popened

    @property
    def popened(self):
        return self._uv_build_popened

    @functools.cached_property
    def _tgz_sdist_path(self):
        self._uv_build_popened.wait()  # block until build completes
        tgz_paths = list(self.tmpdir.glob("*.tar.gz"))
        if len(tgz_paths) != 1:
            raise RuntimeError(
                f"expected exactly one .tar.gz in {self.tmpdir}, found {len(tgz_paths)}"
            )
        return tgz_paths[0]

    @functools.cached_property
    def _sdist_path(self):
        tgz_path = self._tgz_sdist_path
        zip_path = tgz_to_zip(tgz_path)
        tgz_path.unlink()
        return zip_path

    def ensure_requirements_member(self):
        sdist_path = self._sdist_path
        if not ZipProxy(sdist_path).toplevel_name_exists(REQUIREMENTS_NAME):
            requirements_path = self.tmpdir.joinpath(REQUIREMENTS_NAME)
            requirements_text = uv_tool_run_uv_pip_freeze_package_path(sdist_path)
            requirements_path.write_text(requirements_text)
            ZipAppender.append_toplevel(sdist_path, requirements_path)

    @functools.cached_property
    def sdist_path(self):
        self.ensure_requirements_member()
        return self._sdist_path

    @property
    def sdist_path_hexdigest(self):
        return calc_zip_content_hexdigest(self.sdist_path)

    @classmethod
    def from_script_path(cls, script_path):
        pyproject_path = find_file_upwards(script_path, PYPROJECT_NAME)
        return cls(pyproject_path.parent)


@frozen
class SdistBuilder:
    script_path = field(validator=instance_of(Path), converter=Path)
    sdist_path = field(validator=instance_of(Path), converter=Path)
    expr_name = field(validator=instance_of(str), default="expr")
    builds_dir = field(validator=instance_of(str), default="builds")
    cache_dir = field(validator=optional(instance_of(str)), default=None)
    python_version = field(validator=_validate_python_version, default=None)
    maybe_packager = field(
        validator=optional(instance_of(Sdister)),
        default=None,
    )
    require_requirements = field(validator=instance_of(bool), default=True)

    def __attrs_post_init__(self):
        if self.python_version is None:
            object.__setattr__(
                self, "python_version", resolve_python_version(self.sdist_path)
            )
        if self.require_requirements:
            if not ZipProxy(self.sdist_path).toplevel_name_exists(REQUIREMENTS_NAME):
                raise FileNotFoundError(
                    f"requirements.txt not found in sdist: {self.sdist_path}"
                )
        self.ensure_requirements_path()
        if not self.requirements_path.exists():
            raise FileNotFoundError(
                f"requirements.txt could not be created at: {self.requirements_path}"
            )

    @functools.cached_property
    def _tmpdir(self):
        return TemporaryDirectory()

    @property
    def tmpdir(self):
        return Path(self._tmpdir.name)

    @property
    def requirements_path(self):
        return self.tmpdir.joinpath(REQUIREMENTS_NAME)

    @functools.cached_property
    def unzipped_path(self):
        zp = ZipProxy(self.sdist_path)
        unzipped_path = zp.extract_toplevel(self.tmpdir)
        return unzipped_path

    def ensure_requirements_path(self):
        if not self.requirements_path.exists():
            if ZipProxy(self.sdist_path).toplevel_name_exists(REQUIREMENTS_NAME):
                ZipProxy(self.sdist_path).extract_toplevel_name(
                    REQUIREMENTS_NAME, self.requirements_path
                )
            else:
                requirements_text = uv_tool_run_uv_pip_freeze_package_path(
                    self.sdist_path
                )
                self.requirements_path.write_text(requirements_text)

    @functools.cached_property
    def _uv_tool_run_xorq_build(self):
        args = (
            "xorq",
            "build",
            str(self.script_path),
            "-e",
            self.expr_name,
            "--builds-dir",
            self.builds_dir,
            *(("--cache-dir", self.cache_dir) if self.cache_dir else ()),
        )
        popened = uv_tool_run(
            *args,
            python_version=self.python_version,
            with_=self.sdist_path,
            with_requirements=self.requirements_path,
        )
        return popened

    @property
    def popened(self):
        return self._uv_tool_run_xorq_build

    def get_build_path(self):
        # FIXME: don't capture stdout so user can still use --pdb
        return Path(self._uv_tool_run_xorq_build.stdout.strip())

    @functools.cached_property
    def copy_sdist(self):
        target = self.get_build_path().joinpath(BUILD_SDIST_NAME)
        copy_path(self.sdist_path, target)
        return target

    @property
    def build_path(self):
        self.copy_sdist
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
            python_version=packager.python_version,
            maybe_packager=packager,
            **kwargs,
        )


@frozen
class SdistRunner:
    build_path = field(validator=instance_of(Path), converter=Path)
    cache_dir = field(validator=optional(instance_of(str)), default=None)
    output_path = field(validator=optional(instance_of(str)), default=None)
    output_format = field(validator=instance_of(str), default="parquet")
    python_version = field(validator=_validate_python_version, default=None)

    def __attrs_post_init__(self):
        if not self.build_path.exists():
            raise FileNotFoundError(f"build path does not exist: {self.build_path}")
        if not self.sdist_path.exists():
            raise FileNotFoundError(f"sdist not found at: {self.sdist_path}")
        if not ZipProxy(self.sdist_path).toplevel_name_exists(REQUIREMENTS_NAME):
            raise FileNotFoundError(
                f"requirements.txt not found in sdist: {self.sdist_path}"
            )
        if self.python_version is None:
            object.__setattr__(
                self, "python_version", resolve_python_version(self.sdist_path)
            )

    @property
    def sdist_path(self):
        return self.build_path.joinpath(BUILD_SDIST_NAME)

    @functools.cached_property
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
            if ZipProxy(self.sdist_path).toplevel_name_exists(REQUIREMENTS_NAME):
                ZipProxy(self.sdist_path).extract_toplevel_name(
                    REQUIREMENTS_NAME, self.requirements_path
                )
            else:
                requirements_text = uv_tool_run_uv_pip_freeze_package_path(
                    self.sdist_path
                )
                self.requirements_path.write_text(requirements_text)

    @functools.cached_property
    def _uv_tool_run_xorq_run(self):
        self.ensure_requirements_path()
        args = (
            "xorq",
            "run",
            str(self.build_path),
            *(("--cache-dir", self.cache_dir) if self.cache_dir else ()),
            *(("--output-path", self.output_path) if self.output_path else ()),
            "--format",
            self.output_format,
        )
        # FIXME: enable streaming output
        popened = uv_tool_run(
            *args,
            python_version=self.python_version,
            with_=self.sdist_path,
            with_requirements=self.requirements_path,
            capturing=False,
        )
        return popened

    @property
    def popened(self):
        return self._uv_tool_run_xorq_run


def find_file_upwards(start, name):
    path = Path(start).absolute()
    if path.is_file():
        path = path.parent
    paths = (p.joinpath(name) for p in (path, *path.parents))
    found = next((p for p in paths if p.exists()), None)
    if not found:
        raise ValueError(
            f"could not find {name!r} in {start!r} or any parent directory"
        )
    return found


def uv_tool_run(
    *args,
    isolated=True,
    python_version=None,
    with_=None,
    with_requirements=None,
    check=True,
    capturing=True,
):
    command_v_xorq = Popened.check_output("command -v xorq", shell=True).strip()
    args = tuple(el if el != command_v_xorq else "xorq" for el in args)
    popened_args = (
        "uv",
        "tool",
        "run",
        *(("--python", python_version) if python_version else ()),
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
    if in_nix_shell():
        import os  # noqa: PLC0415

        env = os.environ | {
            "LD_LIBRARY_PATH": os.environ["UV_TOOL_RUN_LD_LIBRARY_PATH"]
        }
        kwargs_tuple = kwargs_tuple + (("env", env),)
    popened = Popened(popened_args, kwargs_tuple=kwargs_tuple)
    if check:
        popened.wait()
        if popened.returncode:
            raise subprocess.CalledProcessError(
                popened.returncode, popened_args, popened.stdout, popened.stderr
            )
    return popened


def get_acceptable_python_versions(
    path: str | Path,
    known_minors: Iterable[int] = range(8, 14),
) -> tuple[Version, ...]:
    if (path := Path(path)).name == PYPROJECT_NAME:
        pass
    elif path.is_dir() and path.joinpath(PYPROJECT_NAME).exists():
        path = path.joinpath(PYPROJECT_NAME)
    elif path.suffix == ".zip":
        td = TemporaryDirectory()
        path = ZipProxy(path).extract_toplevel_name(
            PYPROJECT_NAME, Path(td.name, PYPROJECT_NAME)
        )
    else:
        raise ValueError(
            f"can only handle {PYPROJECT_NAME} or {PYPROJECT_NAME} containing dir / .zip"
        )
    data = tomlkit.loads(Path(path).read_text())
    requires_python = toolz.get_in(("project", "requires-python"), data)
    spec = SpecifierSet(requires_python)
    acceptable_python_versions = tuple(
        str(v) for v in (Version(f"3.{minor}") for minor in known_minors) if v in spec
    )
    if not acceptable_python_versions:
        raise ValueError("No acceptable python versions found")
    return acceptable_python_versions


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
