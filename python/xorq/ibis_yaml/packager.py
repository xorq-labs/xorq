import functools
import hashlib
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import toml
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)

from xorq.common.utils.process_utils import (
    Popened,
)


@frozen
class Packager:
    script_path = field(validator=instance_of(Path), converter=Path)
    pyproject_path = field(validator=optional(instance_of(Path)), default=None)
    varname = field(validator=instance_of(str), default="expr")

    pyproject_name = "pyproject.toml"

    def __attrs_post_init__(self):
        assert self.script_path.exists() and self.script_path.suffix == ".py"
        if self.pyproject_path is None:
            found = find_file_upwards(self.script_path, self.pyproject_name)
            object.__setattr__(self, "pyproject_path", found)
        object.__setattr__(self, "pyproject_path", Path(self.pyproject_path))
        assert self.pyproject_path.name == self.pyproject_name

    @property
    def pyproject(self):
        return toml.loads(self.pyproject_path.read_text())

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
    def sdist_path(self):
        prefix = "Successfully built "
        (_, line) = self._uv_build_popened.stderr.strip().rsplit("\n", 1)
        (first, rest) = (line[: len(prefix)], line[len(prefix) :])
        assert first == prefix
        return Path(rest)

    @property
    def sdist_path_hexdigest(self):
        return calc_tgz_hexdigest(self.sdist_path)


def find_file_upwards(start, name):
    path = Path(start).absolute()
    if path.is_file():
        path = path.parent
    paths = (p.joinpath(name) for p in (path, *path.parents))
    found = next((p for p in paths if p.exists()), None)
    if not found:
        raise ValueError
    return found


def calc_tgz_hexdigest(path):
    # tafile itself contains metadata like permissions and modification time
    # so hash just the content
    with tarfile.TarFile.gzopen(path) as tf:
        dct = {
            name: hashlib.file_digest(fh, "md5").hexdigest()
            for name, fh in (
                (member.name, tf.extractfile(member)) for member in tf.getmembers()
            )
            if fh
        }
        md5 = hashlib.md5()
        for key, value in dct.items():
            md5.update(key.encode("ascii"))
            md5.update(value.encode("ascii"))
        return md5.hexdigest()


def main():
    import sys

    import toolz

    script_path, pyproject_path = (toolz.get(i, sys.argv, None) for i in (1, 2))
    if rest := sys.argv[2:]:
        raise ValueError(rest)
    packager = Packager(script_path, pyproject_path)
    print(packager.sdist_path)
    print(packager.sdist_path_hexdigest)
    return packager


if __name__ == "__main__":
    packager = main()
