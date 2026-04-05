import contextlib
import functools
import hashlib
import shutil
import tarfile
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


__all__ = [
    "ZipProxy",
    "append_toplevel",
    "calc_zip_content_hexdigest",
    "replace_toplevel",
    "tgz_to_zip",
]


# paths that uv build injects that don't impact package functionality
uv_sdist_omit_names = ("PKG-INFO", ".gitignore")


def uv_sdist_member_filter(name):
    return Path(name).name not in uv_sdist_omit_names


def get_root_dir(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        (name, *rest) = zf.namelist()
        (root_dir, *_) = Path(name).parts
        if not root_dir:
            raise ValueError(f"zip archive has no root directory: {zip_path}")
        if not all(n.startswith(root_dir) for n in rest):
            raise ValueError(
                f"zip archive has members outside root directory {root_dir!r}: {zip_path}"
            )
        return Path(root_dir)


def calc_zip_content_hexdigest(path, member_filter=uv_sdist_member_filter):
    from xorq.common.utils.dask_normalize.dask_normalize_utils import (  # noqa: PLC0415
        file_digest,
    )

    with zipfile.ZipFile(path, "r") as zf:
        names = sorted(
            name
            for name in zf.namelist()
            if not name.endswith("/") and member_filter(name)
        )
        md5 = hashlib.md5(usedforsecurity=False)
        for name in names:
            md5.update(file_digest(zf.open(name), hashlib.md5).encode("ascii"))
        return md5.hexdigest()


@frozen
class ZipProxy:
    zip_path = field(validator=instance_of(Path), converter=Path)

    def __attrs_post_init__(self):
        if self.zip_path.suffix != ".zip":
            raise ValueError(
                f"expected .zip file, got {self.zip_path.suffix!r}: {self.zip_path}"
            )

    @functools.cached_property
    def root_dir(self):
        return get_root_dir(self.zip_path)

    def toplevel_name_exists(self, name):
        with self.open() as zf:
            return any(
                relpath and str(relpath) == name
                for relpath in (
                    try_path_relative_to(member, self.root_dir)
                    for member in zf.namelist()
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
                shutil.copyfileobj(ifh, ofh)
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
                with zf.open(name) as src, destpath.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        return dest


def append_toplevel(zip_path, append_path):
    append_path = Path(append_path)
    arcname = str(ZipProxy(zip_path).root_dir.joinpath(append_path.name))
    with zipfile.ZipFile(zip_path, "a") as zf:
        zf.write(append_path, arcname=arcname)
    return zip_path


def replace_toplevel(zip_path, replace_path):
    """Replace an existing toplevel member in a zip archive.

    Rewrites the zip, substituting the member whose name matches
    replace_path.name with the contents of replace_path.
    """
    zip_path = Path(zip_path)
    replace_path = Path(replace_path)
    arcname = str(ZipProxy(zip_path).root_dir.joinpath(replace_path.name))
    tmp_path = zip_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(zip_path, "r") as zf_in:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf_out:
            for item in zf_in.namelist():
                if item == arcname:
                    continue
                zf_out.writestr(item, zf_in.read(item))
            zf_out.write(replace_path, arcname=arcname)
    tmp_path.replace(zip_path)
    return zip_path


def tgz_to_zip(tgz_path, zip_path=None):
    """Convert a .tar.gz archive to a .zip archive."""
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


@toolz.curried.excepts(ValueError)
def try_path_relative_to(from_, to_):
    return Path(from_).relative_to(to_)
