import contextlib
import functools
import gzip
import hashlib
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)


# paths that uv build injects that don't impact package functionality
uv_sdist_omit_suffixes = ("PKG-INFO", ".gitignore")


@toolz.curried.excepts(ValueError)
def try_path_relative_to(from_, to_):
    return Path(from_).relative_to(to_)


def uv_sdist_member_filter(member):
    return not any(member.name.endswith(suffix) for suffix in uv_sdist_omit_suffixes)


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


def get_root_dir(tgz_path):
    with tarfile.TarFile.gzopen(tgz_path) as tf:
        (name, *rest) = tf.getnames()
        (root_dir, *_) = Path(name).parts
        assert root_dir
        assert all(other.startswith(root_dir) for other in rest)
        return Path(root_dir)


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
                other and str(other) == name
                for other in (
                    try_path_relative_to(other, self.root_dir)
                    for other in tf.getnames()
                )
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
