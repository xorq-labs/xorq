import contextlib
import functools
import gzip
import hashlib
import shutil
import tarfile
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


REQUIREMENTS_NAME = "requirements.txt"
PYPROJECT_NAME = "pyproject.toml"
BUILD_SDIST_NAME = "sdist.tar.gz"


# paths that uv build injects that don't impact package functionality
uv_sdist_omit_suffixes = ("PKG-INFO", ".gitignore")


def uv_sdist_member_filter(member):
    return not any(member.name.endswith(suffix) for suffix in uv_sdist_omit_suffixes)


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
    def from_script_path(cls, script_path, project_path=None, args=()):
        packager = (
            Sdister(project_path)
            if project_path
            else Sdister.from_script_path(script_path)
        )
        return cls(
            script_path,
            packager.sdist_path,
            args=args,
            maybe_packager=packager,
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


def find_file_upwards(start, name):
    path = Path(start).absolute()
    if path.is_file():
        path = path.parent
    paths = (p.joinpath(name) for p in (path, *path.parents))
    found = next((p for p in paths if p.exists()), None)
    if not found:
        raise ValueError
    return found


def calc_tgz_content_hexdigest(path, member_filter=uv_sdist_member_filter):
    # ignore metadata like permissions and modification time
    with tarfile.TarFile.gzopen(path) as tf:
        dct = {
            name: hashlib.file_digest(fh, "md5").hexdigest()
            for name, fh in (
                (member.name, tf.extractfile(member))
                for member in tf.getmembers()
                if not member.isdir() and member_filter(member)
            )
            if fh
        }
        print(tuple(sorted(dct.items())))
        md5 = hashlib.md5()
        for key, value in sorted(dct.items()):
            # can't use key: top level dir changes shouldn't impact hash
            # md5.update(key.encode("ascii"))
            md5.update(value.encode("ascii"))
        return md5.hexdigest()


@frozen
class TGZProxy:
    tgz_path = field(validator=instance_of(Path), converter=Path)

    valid_full_suffixes = (".tar.gz", ".tgz")

    def __attrs_post_init__(self):
        full_suffix = "".join(self.tgz_path.suffixes)
        assert any(full_suffix.endswith(suffix) for suffix in self.valid_full_suffixes)

    @property
    @functools.cache
    def root_dir(self):
        return get_root_dir(self.tgz_path)

    def toplevel_name_exists(self, name):
        with self.open() as tf:
            return any(
                str(toolz.excepts(ValueError, Path(other).relative_to)(self.root_dir))
                == name
                for other in tf.getnames()
            )

    @contextlib.contextmanager
    def open(self):
        with tarfile.TarFile.gzopen(self.tgz_path) as tf:
            yield tf

    @contextlib.contextmanager
    def open_member(self, member_path):
        # this is a bytes stream
        with self.open() as tf:
            yield tf.extractfile(str(member_path))

    @contextlib.contextmanager
    def open_toplevel_member(self, member_path):
        # this is a bytes stream
        with self.open_member(self.root_dir.joinpath(member_path)) as fh:
            yield fh

    def extract_toplevel_name(self, name, dest):
        dest = Path(dest)
        with dest.open("wb") as ofh:
            with self.open_toplevel_member(name) as ifh:
                ofh.write(ifh.read())
        return dest

    def remove_rel_paths(self, *paths):
        return self.remove_paths(*(self.root_dir.joinpath(path) for path in paths))

    def add_rel_paths(self, *paths):
        return self.add_paths(*(self.root_dir.joinpath(path) for path in paths))

    def add_paths(self, *paths):
        assert paths
        raise NotImplementedError

    def remove_paths(self, *paths):
        assert paths
        raise NotImplementedError


def gunzip_path(from_, to_):
    with gzip.GzipFile(from_, "rb") as gfh:
        with to_.open("wb") as fh:
            shutil.copyfileobj(gfh, fh)
    return to_


def gzip_path(from_, to_):
    with from_.open("rb") as fh:
        with gzip.GzipFile(to_, "wb") as gfh:
            shutil.copyfileobj(fh, gfh)
    return to_


def copy_path(from_, to_):
    with from_.open("rb") as from_fh:
        with to_.open("wb") as to_fh:
            shutil.copyfileobj(from_fh, to_fh)


def tar_append(tar_path, append_path, **kwargs):
    with tarfile.TarFile(tar_path, mode="a") as tf:
        tf.add(append_path, **kwargs)


@frozen
class TGZAppender:
    tgz_path = field(validator=instance_of(Path), converter=Path)
    append_path = field(validator=instance_of(Path), converter=Path)
    kwargs_tuple = field(validator=instance_of(tuple))

    @property
    def kwargs(self):
        return dict(self.kwargs_tuple)

    @property
    @functools.cache
    def root_dir(self):
        return get_root_dir(self.tgz_path)

    @property
    @functools.cache
    def _tmpdir(self):
        return TemporaryDirectory()

    @property
    def tmpdir(self):
        return Path(self._tmpdir.name)

    @property
    @functools.cache
    def gunzipped_path(self):
        gunzipped_path = self.tmpdir.joinpath("gunzipped.tar")
        gunzip_path(self.tgz_path, gunzipped_path)
        return gunzipped_path

    @property
    @functools.cache
    def appended_path(self):
        appended_path = self.tmpdir.joinpath("appended.tar")
        copy_path(self.gunzipped_path, appended_path)
        tar_append(appended_path, self.append_path, **self.kwargs)
        return appended_path

    @property
    @functools.cache
    def appended_tgz_path(self):
        appended_tgz_path = self.tmpdir.joinpath(self.tgz_path.name)
        gzip_path(self.appended_path, appended_tgz_path)
        return appended_tgz_path

    @classmethod
    def append_toplevel(cls, tgz_path, append_path, suffix=".bak", **kwargs):
        # write to same location as the original tgz_path
        append_path = Path(append_path)
        root_dir = get_root_dir(tgz_path)
        arcname = str(root_dir.joinpath(append_path))
        kwargs_tuple = tuple(({"arcname": arcname} | kwargs).items())
        self = cls(tgz_path, append_path, kwargs_tuple=kwargs_tuple)
        # make sure the append is successful
        self.appended_tgz_path
        renamed = self.tgz_path.with_name(self.tgz_path.name + suffix)
        self.tgz_path.rename(renamed)
        self.appended_tgz_path.rename(self.tgz_path)
        return tgz_path, renamed


def get_root_dir(tgz_path):
    with tarfile.TarFile.gzopen(tgz_path) as tf:
        (name, *rest) = tf.getnames()
        (root_dir, *_) = Path(name).parts
        assert root_dir
        assert all(other.startswith(root_dir) for other in rest)
        return Path(root_dir)


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
        assert not popened.returncode
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
