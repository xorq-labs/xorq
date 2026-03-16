import contextlib
import functools
import hashlib
import shutil
import zipfile
from pathlib import Path

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


def copy_path(from_, to_):
    with from_.open("rb") as from_fh:
        with to_.open("wb") as to_fh:
            shutil.copyfileobj(from_fh, to_fh)


@toolz.curried.excepts(ValueError)
def try_path_relative_to(from_, to_):
    return Path(from_).relative_to(to_)


def uv_sdist_member_filter(name):
    return not any(name.endswith(suffix) for suffix in uv_sdist_omit_suffixes)


def get_root_dir(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        (name, *rest) = zf.namelist()
        (root_dir, *_) = Path(name).parts
        assert root_dir
        assert all(n.startswith(root_dir) for n in rest)
        return Path(root_dir)


def calc_zip_content_hexdigest(path, member_filter=uv_sdist_member_filter):
    from xorq.common.utils.dask_normalize.dask_normalize_utils import (  # noqa: PLC0415
        file_digest,
    )

    with zipfile.ZipFile(path, "r") as zf:
        dct = {
            name: file_digest(zf.open(name), hashlib.md5)
            for name in zf.namelist()
            if not name.endswith("/") and member_filter(name)
        }
        md5 = hashlib.md5()
        for _, value in sorted(dct.items()):
            md5.update(value.encode("ascii"))
        return md5.hexdigest()


@frozen
class ZipProxy:
    zip_path = field(validator=instance_of(Path), converter=Path)

    def __attrs_post_init__(self):
        assert self.zip_path.suffix == ".zip"

    @functools.cached_property
    def root_dir(self):
        return get_root_dir(self.zip_path)

    def toplevel_name_exists(self, name):
        with self.open() as zf:
            return any(
                other and str(other) == name
                for other in (
                    try_path_relative_to(other, self.root_dir)
                    for other in zf.namelist()
                )
            )

    @contextlib.contextmanager
    def open(self):
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            yield zf

    @contextlib.contextmanager
    def open_member(self, member_path):
        with self.open() as zf:
            yield zf.open(str(member_path))

    @contextlib.contextmanager
    def open_toplevel_member(self, member_path):
        with self.open_member(self.root_dir.joinpath(member_path)) as fh:
            yield fh

    def extract_toplevel_name(self, name, dest):
        dest = Path(dest)
        with dest.open("wb") as ofh:
            with self.open_toplevel_member(name) as ifh:
                ofh.write(ifh.read())
        return dest

    @property
    def members(self):
        with self.open() as zf:
            return zf.namelist()

    def extract_toplevel(self, dest):
        dest = Path(dest).absolute()
        with self.open() as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                relpath = Path(name).relative_to(self.root_dir)
                destpath = dest.joinpath(relpath)
                destpath.parent.mkdir(exist_ok=True, parents=True)
                destpath.write_bytes(zf.read(name))
        return dest


@frozen
class ZipAppender:
    zip_path = field(validator=instance_of(Path), converter=Path)
    append_path = field(validator=instance_of(Path), converter=Path)
    arcname = field(validator=instance_of(str))

    @classmethod
    def append_toplevel(cls, zip_path, append_path, **kwargs):
        append_path = Path(append_path)
        arcname = str(ZipProxy(zip_path).root_dir.joinpath(append_path.name))
        self = cls(zip_path, append_path, arcname=arcname)
        with zipfile.ZipFile(self.zip_path, "a") as zf:
            zf.write(self.append_path, arcname=self.arcname)
        return zip_path


def tgz_to_zip(tgz_path, zip_path=None):
    """Convert a .tar.gz archive to a .zip archive."""
    import tarfile  # noqa: PLC0415

    tgz_path = Path(tgz_path)
    if zip_path is None:
        zip_path = tgz_path.with_suffix("").with_suffix(".zip")
    zip_path = Path(zip_path)

    with tarfile.TarFile.gzopen(tgz_path) as tf:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for member in tf.getmembers():
                if member.isdir():
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                zf.writestr(member.name, fh.read())

    return zip_path


__all__ = [
    "ZipAppender",
    "ZipProxy",
    "calc_zip_content_hexdigest",
    "copy_path",
    "get_root_dir",
    "tgz_to_zip",
]
