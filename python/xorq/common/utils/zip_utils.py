import contextlib
import functools
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


__all__ = [
    "ZipProxy",
    "append_toplevel",
]


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


@toolz.curried.excepts(ValueError)
def try_path_relative_to(from_, to_):
    return Path(from_).relative_to(to_)
