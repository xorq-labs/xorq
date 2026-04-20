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
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

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

from xorq.common.utils.process_utils import in_nix_shell
from xorq.ibis_yaml.enums import DumpFiles


PYPROJECT_NAME = "pyproject.toml"
UVLOCK_NAME = "uv.lock"
DEFAULT_REQUIRES_PYTHON = ">=3.10"
PYTHON_VERSION_CAP = SpecifierSet("<3.14")


def _validate_python_version(instance, attribute, value):
    if value is None:
        return
    try:
        SpecifierSet(value)
    except Exception:
        # Bare versions like "3.11" aren't valid specifiers; treat as ==3.11
        try:
            SpecifierSet(f"=={value}")
        except Exception as e:
            raise ValueError(f"invalid python version specifier: {value!r}") from e


def _requires_python_from_pyproject(pyproject_path):
    """Read requires-python from a pyproject.toml, intersected with PYTHON_VERSION_CAP."""
    data = tomlkit.loads(Path(pyproject_path).read_text())
    raw = toolz.get_in(("project", "requires-python"), data)
    return str(SpecifierSet(raw or DEFAULT_REQUIRES_PYTHON) & PYTHON_VERSION_CAP)


def _read_requires_python(path):
    """Read requires-python from a project source, intersected with PYTHON_VERSION_CAP.

    Accepts a pyproject.toml path, a directory containing one, or a .whl file.
    Returns a PEP 440 specifier string.
    """
    path = Path(path)
    if path.name == PYPROJECT_NAME:
        return _requires_python_from_pyproject(path)
    elif path.is_dir() and path.joinpath(PYPROJECT_NAME).exists():
        return _requires_python_from_pyproject(path / PYPROJECT_NAME)
    elif path.suffix == ".whl":
        with zipfile.ZipFile(path) as zf:
            metadata_names = [
                n for n in zf.namelist() if n.endswith(".dist-info/METADATA")
            ]
            if not metadata_names:
                raise ValueError(f"no .dist-info/METADATA found in {path}")
            metadata_text = zf.read(metadata_names[0]).decode()
            for line in metadata_text.splitlines():
                if line.startswith("Requires-Python:"):
                    raw = line.split(":", 1)[1].strip()
                    return str(SpecifierSet(raw) & PYTHON_VERSION_CAP)
            raise ValueError(f"no Requires-Python in wheel metadata: {path}")
    else:
        raise ValueError(
            f"can only handle {PYPROJECT_NAME}, a directory containing one, or .whl"
        )


def _find_single_glob(directory, pattern):
    """Find exactly one file matching pattern in directory."""
    matches = list(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern} in {directory}, found {len(matches)}"
        )
    return matches[0]


@frozen
class WheelBundle:
    """Immutable triplet of wheel + requirements + python_version.

    Centralizes validation, python_version derivation, and optional
    tmpdir lifetime management for the three artifacts that always
    travel together through the packager pipeline.
    """

    wheel_path = field(validator=instance_of(Path), converter=Path)
    requirements_path = field(validator=instance_of(Path), converter=Path)
    python_version = field(validator=_validate_python_version, default=None)
    _tmpdir = field(
        validator=optional(instance_of(TemporaryDirectory)),
        repr=False,
        eq=False,
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
                _read_requires_python(self.wheel_path),
            )

    @classmethod
    def from_build_path(cls, build_path):
        """Discover artifacts from an existing build directory."""
        build_path = Path(build_path)
        return cls(
            wheel_path=_find_single_glob(build_path, "*.whl"),
            requirements_path=build_path / DumpFiles.requirements,
        )


@frozen
class WheelPackager:
    project_path = field(validator=instance_of(Path), converter=Path)
    python_version = field(validator=_validate_python_version, default=None)
    extras = field(factory=tuple, converter=tuple)
    all_extras = field(validator=instance_of(bool), default=True)
    _tmpdir = field(
        validator=instance_of(TemporaryDirectory),
        repr=False,
        eq=False,
        factory=TemporaryDirectory,
    )

    def __attrs_post_init__(self):
        if not self.project_path.exists():
            raise FileNotFoundError(f"project path does not exist: {self.project_path}")
        if not self.pyproject_path.exists():
            raise FileNotFoundError(
                f"pyproject.toml not found at: {self.pyproject_path}"
            )
        if self.python_version is None:
            object.__setattr__(
                self, "python_version", _read_requires_python(self.pyproject_path)
            )
        if not (
            self.project_path.joinpath(DumpFiles.requirements).exists()
            or self.project_path.joinpath(UVLOCK_NAME).exists()
        ):
            raise FileNotFoundError(
                f"neither {DumpFiles.requirements} nor {UVLOCK_NAME} found in "
                f"{self.project_path}; run 'uv lock' first"
            )

    @property
    def built(self):
        return bool(
            any(self.tmpdir.glob("*.whl"))
            and (self.tmpdir / DumpFiles.requirements).exists()
        )

    @property
    def pyproject_path(self):
        return self.project_path.joinpath(PYPROJECT_NAME)

    @property
    def tmpdir(self):
        return Path(self._tmpdir.name)

    def _build_wheel(self):
        from xorq.common.utils.otel_utils import tracer  # noqa: PLC0415

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
            subprocess.run(args, check=True, env=_nix_env())

    def _write_requirements_path(self):
        from xorq.common.utils.otel_utils import tracer  # noqa: PLC0415

        has_lockfile = self.project_path.joinpath(UVLOCK_NAME).exists()
        existing = self.project_path / DumpFiles.requirements

        if has_lockfile:
            with tracer.start_as_current_span("packager.uv_export_requirements"):
                exported = uv_export_requirements(
                    self.project_path,
                    self.python_version,
                    extras=self.extras,
                    all_extras=self.all_extras,
                )
            if existing.exists() and existing.read_text() != exported:
                raise RuntimeError(
                    f"{DumpFiles.requirements} in {self.project_path} does not match "
                    f"`uv export` output from {UVLOCK_NAME} (byte-exact comparison). "
                    f"This happens when {UVLOCK_NAME} changes, or when the in-tree "
                    f"{DumpFiles.requirements} was produced by a different uv version "
                    f"than the one running now. To resolve: delete "
                    f"{existing} and let the packager regenerate it, "
                    f"or re-export manually with `uv export --locked --no-dev "
                    f"--no-emit-project --no-header --no-annotate > {existing.name}`."
                )
            (self.tmpdir / DumpFiles.requirements).write_text(exported)
        else:
            shutil.copy2(existing, self.tmpdir / DumpFiles.requirements)

    def build(self):
        """Build the wheel and export requirements. Returns a WheelBundle."""
        if not self.built:
            self._build_wheel()
            self._write_requirements_path()
        return WheelBundle(
            wheel_path=_find_single_glob(self.tmpdir, "*.whl"),
            requirements_path=self.tmpdir / DumpFiles.requirements,
            python_version=self.python_version,
            tmpdir=self._tmpdir,
        )

    @classmethod
    def from_script_path(cls, script_path, **kwargs):
        pyproject_path = find_file_upwards(script_path, PYPROJECT_NAME)
        return cls(pyproject_path.parent, **kwargs)


@frozen
class PackagedBuilder:
    script_path = field(validator=instance_of(Path), converter=Path)
    bundle = field(validator=instance_of(WheelBundle))
    expr_name = field(validator=instance_of(str), default="expr")
    builds_dir = field(validator=instance_of(str), default="builds")
    cache_dir = field(validator=optional(instance_of(str)), default=None)

    @property
    def wheel_path(self):
        return self.bundle.wheel_path

    @property
    def requirements_path(self):
        return self.bundle.requirements_path

    @property
    def python_version(self):
        return self.bundle.python_version

    @functools.cached_property
    def _build(self):
        """Run xorq build and copy wheel + requirements into the build dir."""
        from xorq.common.utils.otel_utils import tracer  # noqa: PLC0415

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
        with tracer.start_as_current_span("packager.build"):
            result = uv_tool_run(
                *args,
                python_version=self.python_version,
                with_=self.wheel_path,
                with_requirements=self.requirements_path,
            )

        with tracer.start_as_current_span("packager.copy_artifacts"):
            # xorq build writes the build path as its final stdout line;
            # take the last non-empty line to tolerate stray preceding output.
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError(
                    f"xorq build produced no stdout path; stderr: {result.stderr}"
                )
            build_path = Path(lines[-1])
            shutil.copy2(self.wheel_path, build_path / self.wheel_path.name)
            shutil.copy2(self.requirements_path, build_path / DumpFiles.requirements)

        return result, build_path

    def build(self):
        self._build
        return self

    @property
    def build_result(self):
        return self._build[0]

    @property
    def build_path(self):
        return self._build[1]

    @classmethod
    def from_script_path(
        cls, script_path, project_path=None, extras=(), all_extras=True, **kwargs
    ):
        packager_kwargs = {"extras": extras, "all_extras": all_extras}
        packager = (
            WheelPackager(project_path, **packager_kwargs)
            if project_path
            else WheelPackager.from_script_path(script_path, **packager_kwargs)
        )
        return cls(
            script_path=script_path,
            bundle=packager.build(),
            **kwargs,
        )


@frozen
class PackagedRunner:
    build_path = field(validator=instance_of(Path), converter=Path)
    cache_dir = field(validator=optional(instance_of(str)), default=None)
    output_path = field(validator=optional(instance_of(str)), default=None)
    output_format = field(validator=instance_of(str), default="parquet")
    python_version = field(validator=_validate_python_version, default=None)
    _bundle = field(init=False, repr=False, eq=False, default=None)

    def __attrs_post_init__(self):
        if not self.build_path.exists():
            raise FileNotFoundError(f"build path does not exist: {self.build_path}")
        try:
            bundle = WheelBundle.from_build_path(self.build_path)
        except (RuntimeError, FileNotFoundError) as e:
            raise FileNotFoundError(
                f"invalid build path {self.build_path}: {e}"
            ) from None
        object.__setattr__(self, "_bundle", bundle)
        if self.python_version is None:
            object.__setattr__(self, "python_version", bundle.python_version)

    @property
    def wheel_path(self):
        return self._bundle.wheel_path

    @property
    def requirements_path(self):
        return self._bundle.requirements_path

    @functools.cached_property
    def _run(self):
        args = (
            "xorq",
            "run",
            str(self.build_path),
            *(("--cache-dir", self.cache_dir) if self.cache_dir else ()),
            *(("--output-path", self.output_path) if self.output_path else ()),
            "--format",
            self.output_format,
        )
        result = uv_tool_run(
            *args,
            python_version=self.python_version,
            with_=self.wheel_path,
            with_requirements=self.requirements_path,
            capture_output=False,
        )
        return result

    def run(self):
        self._run
        return self

    @property
    def run_result(self):
        return self._run


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


@functools.cache
def _xorq_exe_resolved():
    """Resolve the current xorq executable to a canonical path for comparison."""
    found = shutil.which("xorq")
    if found is None:
        return None
    return Path(found).resolve()


def _normalize_xorq_cmd(args):
    """If args[0] resolves to the current xorq exe, replace it with bare 'xorq'."""
    resolved = _xorq_exe_resolved()
    if resolved is None:
        return args
    if Path(args[0]).resolve() == resolved:
        return ("xorq", *args[1:])
    return args


def _nix_env():
    """Return an env dict with LD_LIBRARY_PATH fixed for nix, or None outside nix."""
    if not in_nix_shell():
        return None
    env = os.environ.copy()
    ld_override = os.environ.get("UV_TOOL_RUN_LD_LIBRARY_PATH")
    if ld_override is not None:
        env["LD_LIBRARY_PATH"] = ld_override
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def uv_tool_run(
    *args,
    isolated=True,
    python_version=None,
    with_=None,
    with_requirements=None,
    check=True,
    capture_output=True,
):
    from xorq.common.utils.otel_utils import tracer  # noqa: PLC0415

    with tracer.start_as_current_span("packager.uv_tool_run") as span:
        args = _normalize_xorq_cmd(args)
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
        kwargs = {"capture_output": True, "text": True} if capture_output else {}
        env = _nix_env()
        if env is not None:
            kwargs["env"] = env
        return subprocess.run(run_args, check=check, **kwargs)


def uv_export_requirements(project_dir, python_version, extras=(), all_extras=True):
    """Run uv export in a directory with pyproject.toml + uv.lock."""
    args = (
        "uv",
        "export",
        "--locked",
        "--no-dev",
        "--no-emit-project",
        "--no-header",
        "--no-annotate",
        "--python",
        python_version,
        "--directory",
        str(project_dir),
        *(("--all-extras",) if all_extras else ()),
        *(arg for extra in extras for arg in ("--extra", extra)),
    )
    result = subprocess.run(args, capture_output=True, text=True, env=_nix_env())
    if result.returncode != 0:
        raise RuntimeError(
            f"uv export failed (exit {result.returncode}) in {project_dir}:\n"
            f"{result.stderr}"
        )
    return result.stdout
