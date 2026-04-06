"""
Wheel-based build and run pipeline for xorq expressions.

Pipeline: WheelPackager → PackagedBuilder → PackagedRunner

WheelPackager    project directory → wheel + requirements.txt sidecar
                 (via `uv build --wheel` and `uv export`).

PackagedBuilder  wheel + script → build directory (via `uv tool run xorq build`),
                 containing the serialized expression, the wheel, and requirements.txt.

PackagedRunner   build directory → execution output (via `uv tool run xorq run`),
                 in the wheel's isolated environment.
"""

import functools
import operator
import os
import shutil
import subprocess
from pathlib import Path
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

from xorq.common.utils.otel_utils import tracer
from xorq.common.utils.process_utils import in_nix_shell
from xorq.ibis_yaml.enums import DumpFiles


REQUIREMENTS_NAME = "requirements.txt"
PYPROJECT_NAME = "pyproject.toml"
UVLOCK_NAME = "uv.lock"


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


def _find_single_glob(directory, pattern):
    """Find exactly one file matching pattern in directory."""
    matches = list(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern} in {directory}, found {len(matches)}"
        )
    return matches[0]


@frozen
class WheelPackager:
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
    def _wheel_path(self):
        with tracer.start_as_current_span("packager.uv_build_wheel") as span:
            span.set_attribute("python_version", self.python_version)
            span.set_attribute("project_path", str(self.pyproject_path.parent))
            args = (
                "uv",
                "build",
                "--wheel",
                "--python",
                self.python_version,
                "--out-dir",
                str(self.tmpdir),
                str(self.pyproject_path.parent),
            )
            subprocess.run(args, check=True, capture_output=True)
            return _find_single_glob(self.tmpdir, "*.whl")

    def _ensure_uvlock(self):
        """Ensure uv.lock exists in the project directory (needed for uv export)."""
        uvlock_path = self.project_path.joinpath(UVLOCK_NAME)
        if uvlock_path.exists():
            return uvlock_path
        staging = self.tmpdir / "_uvlock_staging"
        staging.mkdir(exist_ok=True)
        shutil.copy2(self.pyproject_path, staging / PYPROJECT_NAME)
        subprocess.run(
            ("uv", "lock", "--directory", str(staging)),
            check=True,
            capture_output=True,
        )
        return staging / UVLOCK_NAME

    @functools.cached_property
    def requirements_path(self):
        with tracer.start_as_current_span("packager.uv_export_requirements"):
            uvlock_path = self._ensure_uvlock()
            # Stage pyproject.toml + uv.lock together for uv export
            export_dir = self.tmpdir / "_export_staging"
            export_dir.mkdir(exist_ok=True)
            shutil.copy2(self.pyproject_path, export_dir / PYPROJECT_NAME)
            shutil.copy2(uvlock_path, export_dir / UVLOCK_NAME)
            requirements_text = uv_export_requirements(export_dir)
            requirements_path = self.tmpdir / REQUIREMENTS_NAME
            requirements_path.write_text(requirements_text)
            return requirements_path

    @functools.cached_property
    def wheel_path(self):
        with tracer.start_as_current_span("packager.wheel_finalize"):
            # Trigger both builds
            _ = self._wheel_path
            _ = self.requirements_path
            return self._wheel_path

    @classmethod
    def from_script_path(cls, script_path, **kwargs):
        pyproject_path = find_file_upwards(script_path, PYPROJECT_NAME)
        return cls(pyproject_path.parent, **kwargs)

    @classmethod
    def from_script_and_requirements(
        cls, script_path, requirements_path, requires_python=">=3.10", **kwargs
    ):
        """Create a WheelPackager from a bare script and requirements.txt.

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
class PackagedBuilder:
    script_path = field(validator=instance_of(Path), converter=Path)
    wheel_path = field(validator=instance_of(Path), converter=Path)
    requirements_path = field(validator=instance_of(Path), converter=Path)
    expr_name = field(validator=instance_of(str), default="expr")
    builds_dir = field(validator=instance_of(str), default="builds")
    cache_dir = field(validator=optional(instance_of(str)), default=None)
    python_version = field(validator=_validate_python_version, default=None)
    maybe_packager = field(
        validator=optional(instance_of(WheelPackager)),
        default=None,
    )

    def __attrs_post_init__(self):
        if not self.wheel_path.exists():
            raise FileNotFoundError(f"wheel not found: {self.wheel_path}")
        if not self.requirements_path.exists():
            raise FileNotFoundError(f"requirements not found: {self.requirements_path}")
        if self.python_version is None:
            object.__setattr__(
                self,
                "python_version",
                resolve_python_version(self.wheel_path.parent),
            )

    @functools.cached_property
    def _tmpdir(self):
        return TemporaryDirectory()

    @property
    def tmpdir(self):
        return Path(self._tmpdir.name)

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
            with_=self.wheel_path,
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
    def _copy_artifacts(self):
        with tracer.start_as_current_span("packager.copy_artifacts"):
            build_path = self.get_build_path()
            wheel_target = build_path / DumpFiles.wheel
            reqs_target = build_path / DumpFiles.requirements
            shutil.copy2(self.wheel_path, wheel_target)
            shutil.copy2(self.requirements_path, reqs_target)
            return build_path

    @property
    def build_path(self):
        return self._copy_artifacts

    @classmethod
    def from_script_path(cls, script_path, project_path=None, **kwargs):
        packager = (
            WheelPackager(project_path)
            if project_path
            else WheelPackager.from_script_path(script_path)
        )
        return cls(
            script_path=script_path,
            wheel_path=packager.wheel_path,
            requirements_path=packager.requirements_path,
            python_version=packager.python_version,
            maybe_packager=packager,
            **kwargs,
        )


@frozen
class PackagedRunner:
    build_path = field(validator=instance_of(Path), converter=Path)
    cache_dir = field(validator=optional(instance_of(str)), default=None)
    output_path = field(validator=optional(instance_of(str)), default=None)
    output_format = field(validator=instance_of(str), default="parquet")
    python_version = field(validator=_validate_python_version, default=None)

    def __attrs_post_init__(self):
        if not self.build_path.exists():
            raise FileNotFoundError(f"build path does not exist: {self.build_path}")
        if not self.wheel_path.exists():
            raise FileNotFoundError(f"wheel not found at: {self.wheel_path}")
        if not self.requirements_path.exists():
            raise FileNotFoundError(
                f"requirements not found at: {self.requirements_path}"
            )
        if self.python_version is None:
            object.__setattr__(
                self,
                "python_version",
                resolve_python_version(self.wheel_path.parent),
            )

    @property
    def wheel_path(self):
        return self.build_path / DumpFiles.wheel

    @property
    def requirements_path(self):
        return self.build_path / DumpFiles.requirements

    @functools.cached_property
    def _uv_tool_run_xorq_run(self):
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
            with_=self.wheel_path,
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
    with tracer.start_as_current_span("packager.uv_tool_run") as span:
        command_v_xorq = subprocess.check_output(
            "command -v xorq", shell=True, text=True
        ).strip()
        args = tuple(el if el != command_v_xorq else "xorq" for el in args)
        run_args = (
            "uv",
            "tool",
            "run",
            *(("--python", python_version) if python_version else ()),
            *(("--isolated",) if isolated else ()),
            *(("--with", str(with_)) if with_ else ()),
            *(
                ("--with-requirements", str(with_requirements))
                if with_requirements
                else ()
            ),
            *args,
        )
        span.set_attribute("args", " ".join(run_args))
        span.set_attribute("python_version", python_version or "")
        span.set_attribute("isolated", isolated)
        capturing_kwargs = {"capture_output": True, "text": True} if capturing else {}
        nix_shell_kwargs = (
            {
                "env": os.environ
                | {
                    "LD_LIBRARY_PATH": os.environ.get("UV_TOOL_RUN_LD_LIBRARY_PATH", "")
                },
            }
            if in_nix_shell()
            else {}
        )
        kwargs = capturing_kwargs | nix_shell_kwargs
        result = subprocess.run(run_args, **kwargs)
        if check and result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode, run_args, result.stdout, result.stderr
            )
        return result


def get_acceptable_python_versions(
    path: str | Path,
    known_minors: Iterable[int] = range(8, 14),
) -> tuple[Version, ...]:
    if (path := Path(path)).name == PYPROJECT_NAME:
        pass
    elif path.is_dir() and path.joinpath(PYPROJECT_NAME).exists():
        path = path.joinpath(PYPROJECT_NAME)
    elif path.suffix == ".zip":
        from xorq.common.utils.zip_utils import ZipProxy  # noqa: PLC0415

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
    return subprocess.check_output(args, text=True)


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
    subprocess.run(
        ("uv", "lock", "--directory", str(output_dir)), check=True, capture_output=True
    )
    return output_dir
