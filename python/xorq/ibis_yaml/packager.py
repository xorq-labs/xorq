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
    in_nix_shell,
)
from xorq.common.utils.tar_utils import (
    TGZAppender,
    TGZProxy,
    calc_tgz_content_hexdigest,
    copy_path,
)


PYPROJECT_NAME = "pyproject.toml"
UV_LOCK_NAME = "uv.lock"
BUILD_SDIST_NAME = "sdist.tar.gz"


def uv_pip_command(*args, **kwargs):
    """
    Execute a uv pip command with nix shell safety check.

    Direct uv pip commands (without --isolated) can have issues with nix shells
    and enclosing virtual environments. Use uv_tool_run() instead for isolated operations.

    This function exists to:
    1. Document that direct uv pip commands should be avoided
    2. Provide a safety check if they must be used
    """
    assert_not_in_nix_shell()
    return Popened(("uv", "pip", *args), **kwargs)


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

    def ensure_uv_lock_member(self):
        sdist_path = self._sdist_path
        # Copy uv.lock from project if it exists and is not already in the sdist
        if not TGZProxy(sdist_path).toplevel_name_exists(UV_LOCK_NAME):
            project_uv_lock = self.project_path.joinpath(UV_LOCK_NAME)
            if project_uv_lock.exists():
                TGZAppender.append_toplevel(sdist_path, project_uv_lock)

    @property
    @functools.cache
    def sdist_path(self):
        self.ensure_uv_lock_member()
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

    def __attrs_post_init__(self):
        # Require uv.lock for reproducible builds
        assert TGZProxy(self.sdist_path).toplevel_name_exists(UV_LOCK_NAME), (
            "sdist must contain uv.lock for reproducible builds"
        )

    @property
    @functools.cache
    def _uv_tool_run_xorq_build(self):
        args = self.args if self.args else ("xorq", "build", str(self.script_path))
        # Use uv tool run --with for building (not --frozen, since build doesn't need reproducibility)
        # The sdist will be installed which makes the package importable
        popened = uv_tool_run(
            *args,
            with_=self.sdist_path,
        )
        return popened

    popened = _uv_tool_run_xorq_build

    def get_build_path(self):
        # FIXME: don't capture stdout so user can still use --pdb
        return Path(self._uv_tool_run_xorq_build.stdout.strip())

    @functools.cache
    def copy_sdist(self):
        target = self.get_build_path().joinpath(BUILD_SDIST_NAME)
        # Ensure parent directory exists
        target.parent.mkdir(parents=True, exist_ok=True)
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
        # Require uv.lock for reproducible builds
        assert TGZProxy(self.sdist_path).toplevel_name_exists(UV_LOCK_NAME), (
            "sdist must contain uv.lock for reproducible builds"
        )

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
    def uv_lock_path(self):
        return self.tmpdir.joinpath(UV_LOCK_NAME)

    @property
    def pyproject_path(self):
        return self.tmpdir.joinpath("pyproject.toml")

    @property
    def requirements_path(self):
        return self.tmpdir.joinpath("requirements.txt")

    def ensure_requirements_path(self):
        """Extract uv.lock and pyproject.toml from sdist and export to requirements.txt"""
        if not self.requirements_path.exists():
            # Extract uv.lock and pyproject.toml from sdist (both needed for uv export)
            if not self.uv_lock_path.exists():
                TGZProxy(self.sdist_path).extract_toplevel_name(
                    UV_LOCK_NAME, self.uv_lock_path
                )
            if not self.pyproject_path.exists():
                TGZProxy(self.sdist_path).extract_toplevel_name(
                    "pyproject.toml", self.pyproject_path
                )

            # Export uv.lock to requirements.txt using uv export
            export_args = (
                "uv",
                "export",
                "--format",
                "requirements.txt",
                "--frozen",
                "--no-hashes",  # Don't include hashes for cleaner output
            )
            popened = Popened(
                export_args,
                kwargs_tuple=(
                    ("stdout", PIPE),
                    ("stderr", PIPE),
                    (
                        "cwd",
                        str(self.tmpdir),
                    ),  # Run in tmpdir where uv.lock and pyproject.toml are
                ),
            )
            popened.popen.wait()
            assert not popened.returncode, popened.stderr

            # Filter out local file:// dependencies since they'll be provided via --with sdist
            exported_reqs = popened.stdout
            filtered_lines = []
            for line in exported_reqs.split("\n"):
                # Skip local file:// paths (the template package itself, e.g., xorq-template-sklearn)
                # These are provided by --with <sdist>
                if "file:///" in line:
                    continue
                # Skip xorq itself ONLY when in nix shell (xorq development environment)
                # In nix shell: avoids conflict with editable xorq install
                # Outside nix shell: keep xorq for regular users who need it from uv.lock
                if in_nix_shell() and (
                    line.startswith("xorq @") or line.startswith("xorq==")
                ):
                    continue
                filtered_lines.append(line)

            self.requirements_path.write_text("\n".join(filtered_lines))

    @property
    @functools.cache
    def _uv_tool_run_xorq_run(self):
        self.ensure_requirements_path()
        args = self.args if self.args else ("xorq", "run", str(self.build_path))
        # Use uv tool run with sdist and requirements from uv.lock
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
    *args,
    isolated=True,
    with_=None,
    with_requirements=None,
    check=True,
    capturing=True,
):
    """
    Execute a command via uv tool run in an isolated environment.

    This is safe in nix shells because --isolated creates a completely separate
    Python environment that doesn't inherit from or conflict with nix packages.

    For direct uv pip commands (not isolated), use uv_pip_command() which includes
    the nix shell check.
    """
    # Try to get xorq path, but don't fail if it's not in PATH (e.g., with nix run)
    try:
        command_v_xorq = Popened.check_output("command -v xorq", shell=True).strip()
        args = tuple(el if el != command_v_xorq else "xorq" for el in args)
    except (AssertionError, Exception):
        pass  # xorq not in PATH, args remain unchanged
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
        popened.popen.wait()
        assert not popened.returncode, popened.stderr
    return popened
