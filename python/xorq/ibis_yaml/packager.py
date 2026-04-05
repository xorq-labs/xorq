"""
Sdist-based build and run pipeline for xorq expressions.

Pipeline: Sdister → SdistBuilder → SdistRunner

Sdister      project directory → sdist zip (via `uv build --sdist`),
             guaranteeing uv.lock and requirements.txt are embedded.

SdistBuilder sdist zip + script → build directory (via `uv tool run xorq build`),
             containing the serialized expression and a copy of the sdist.

SdistRunner  build directory → execution output (via `uv tool run xorq run`),
             in the sdist's isolated environment.
"""

import functools
import operator
import shutil
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
    ZipProxy,
    append_toplevel,
    calc_zip_content_hexdigest,
    tgz_to_zip,
)


REQUIREMENTS_NAME = "requirements.txt"
PYPROJECT_NAME = "pyproject.toml"
UVLOCK_NAME = "uv.lock"
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

    def ensure_uvlock_member(self):
        sdist_path = self._sdist_path
        if ZipProxy(sdist_path).toplevel_name_exists(UVLOCK_NAME):
            return
        uvlock_path = self.project_path.joinpath(UVLOCK_NAME)
        if not uvlock_path.exists():
            staging = self.tmpdir / "_uvlock_staging"
            staging.mkdir(exist_ok=True)
            shutil.copy2(self.pyproject_path, staging / PYPROJECT_NAME)
            Popened.check_output(
                ("uv", "lock", "--directory", str(staging)), shell=False
            )
            uvlock_path = staging / UVLOCK_NAME
        append_toplevel(sdist_path, uvlock_path)

    def ensure_requirements_member(self):
        sdist_path = self._sdist_path
        zp = ZipProxy(sdist_path)
        requirements_text = uv_export_requirements_from_sdist(sdist_path, self.tmpdir)
        if zp.toplevel_name_exists(REQUIREMENTS_NAME):
            with zp.open_toplevel_member(REQUIREMENTS_NAME) as fh:
                existing = fh.read().decode()
            if existing != requirements_text:
                raise ValueError(
                    f"existing {REQUIREMENTS_NAME} in sdist does not match "
                    f"what uv export generates from uv.lock"
                )
        else:
            requirements_path = self.tmpdir.joinpath(REQUIREMENTS_NAME)
            requirements_path.write_text(requirements_text)
            append_toplevel(sdist_path, requirements_path)

    @functools.cached_property
    def sdist_path(self):
        self.ensure_uvlock_member()
        self.ensure_requirements_member()
        return self._sdist_path

    @property
    def sdist_path_hexdigest(self):
        return calc_zip_content_hexdigest(self.sdist_path)

    @classmethod
    def from_script_path(cls, script_path):
        pyproject_path = find_file_upwards(script_path, PYPROJECT_NAME)
        return cls(pyproject_path.parent)

    @classmethod
    def from_script_and_requirements(
        cls, script_path, requirements_path, requires_python=">=3.10", **kwargs
    ):
        """Create an Sdister from a bare script and requirements.txt.

        Generates a temporary project in a staging dir, then constructs
        normally. The staging dir content is moved into self.tmpdir afterward
        so a single tmpdir owns everything.
        """
        with TemporaryDirectory() as staging:
            generate_project_from_requirements(
                script_path,
                requirements_path,
                staging,
                requires_python=requires_python,
            )
            instance = cls(project_path=staging, **kwargs)
            # move generated project into instance's tmpdir, repoint project_path
            dest = instance.tmpdir / "project"
            shutil.copytree(staging, dest)
        object.__setattr__(instance, "project_path", dest)
        return instance


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
            zp = ZipProxy(self.sdist_path)
            if not (
                zp.toplevel_name_exists(UVLOCK_NAME)
                or zp.toplevel_name_exists(REQUIREMENTS_NAME)
            ):
                raise FileNotFoundError(
                    f"neither {UVLOCK_NAME} nor {REQUIREMENTS_NAME} found in sdist: {self.sdist_path}"
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
            if ZipProxy(self.sdist_path).toplevel_name_exists(UVLOCK_NAME):
                requirements_text = uv_export_requirements_from_sdist(
                    self.sdist_path, self.tmpdir
                )
                self.requirements_path.write_text(requirements_text)
            elif ZipProxy(self.sdist_path).toplevel_name_exists(REQUIREMENTS_NAME):
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
        shutil.copy2(self.sdist_path, target)
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
        zp = ZipProxy(self.sdist_path)
        if not (
            zp.toplevel_name_exists(UVLOCK_NAME)
            or zp.toplevel_name_exists(REQUIREMENTS_NAME)
        ):
            raise FileNotFoundError(
                f"neither {UVLOCK_NAME} nor {REQUIREMENTS_NAME} found in sdist: {self.sdist_path}"
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
            if ZipProxy(self.sdist_path).toplevel_name_exists(UVLOCK_NAME):
                requirements_text = uv_export_requirements_from_sdist(
                    self.sdist_path, self.tmpdir
                )
                self.requirements_path.write_text(requirements_text)
            elif ZipProxy(self.sdist_path).toplevel_name_exists(REQUIREMENTS_NAME):
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


def uv_export_requirements(project_dir):
    """Run uv export in a directory with pyproject.toml + uv.lock."""
    args = (
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--no-header",
        "--no-annotate",
        "--directory",
        str(project_dir),
    )
    return Popened.check_output(args, shell=False)


def uv_export_requirements_from_sdist(sdist_path, tmpdir):
    """Extract uv.lock + pyproject.toml from sdist, run uv export."""
    tmpdir = Path(tmpdir)
    zp = ZipProxy(sdist_path)
    zp.extract_toplevel_name(UVLOCK_NAME, tmpdir / UVLOCK_NAME)
    zp.extract_toplevel_name(PYPROJECT_NAME, tmpdir / PYPROJECT_NAME)
    return uv_export_requirements(tmpdir)


def parse_requirements(requirements_text):
    """Parse requirements.txt lines into dependency specifiers.

    Strips hashes, comments, blank lines, and options (e.g. --hash, -i).
    Returns a list of PEP 508 dependency strings.
    """
    deps = []
    for line in requirements_text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        # strip inline hash options: e.g. "pkg==1.0 --hash=sha256:abc..."
        if " --hash" in line:
            line = line[: line.index(" --hash")].strip()
        # strip backslash continuations
        line = line.rstrip("\\").strip()
        if line:
            deps.append(line)
    return deps


def generate_pyproject_toml(
    project_name,
    dependencies,
    requires_python=">=3.10",
):
    """Generate a minimal pyproject.toml string from a list of dependencies."""
    doc = tomlkit.document()

    def make_and_do_update(maker, tpl):
        updated = toolz.do(
            operator.methodcaller("update", dict(tpl)),
            maker(),
        )
        return updated

    build_system_tuple = (
        ("requires", ["hatchling"]),
        ("build-backend", "hatchling.build"),
    )
    project_tuple = (
        ("name", project_name),
        ("version", "0.0.0"),
        ("requires-python", requires_python),
        ("dependencies", dependencies),
    )
    doc = make_and_do_update(
        tomlkit.document,
        (
            ("build-system", make_and_do_update(tomlkit.table, build_system_tuple)),
            ("project", make_and_do_update(tomlkit.table, project_tuple)),
        ),
    )
    return tomlkit.dumps(doc)


def generate_project_from_requirements(
    script_path,
    requirements_path,
    output_dir,
    requires_python=">=3.10",
):
    """Create a project directory with pyproject.toml, uv.lock, script, and requirements.txt.

    Parses requirements_path to generate pyproject.toml, copies the script and
    requirements.txt, then runs `uv lock` to produce uv.lock.
    """
    output_dir = Path(output_dir)
    pyproject_text = generate_pyproject_toml(
        project_name=Path(script_path).stem.replace("_", "-"),
        dependencies=parse_requirements(Path(requirements_path).read_text()),
        requires_python=requires_python,
    )
    output_dir.joinpath(PYPROJECT_NAME).write_text(pyproject_text)
    shutil.copy2(script_path, output_dir / script_path.name)
    shutil.copy2(requirements_path, output_dir / REQUIREMENTS_NAME)
    Popened.check_output(("uv", "lock", "--directory", str(output_dir)), shell=False)
    return output_dir
