import hashlib
import tempfile
import zipfile
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

from attr import field, frozen
from attr.validators import instance_of

from xorq.catalog.constants import (
    PREFERRED_SUFFIX,
    VALID_SUFFIXES,
)
from xorq.ibis_yaml.enums import REQUIRED_ARCHIVE_NAMES


def with_pure_suffix(path, suffix=""):
    return path.with_name(path.name.removesuffix("".join(path.suffixes))).with_suffix(
        suffix
    )


def test_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = {
            Path(info.filename).name for info in zf.infolist() if not info.is_dir()
        }
        missing = set(REQUIRED_ARCHIVE_NAMES).difference(names)
        assert not missing, (
            f"Archive is not a valid expression entry "
            f"(missing {missing}); found: {sorted(names)}"
        )
        wheels = [name for name in names if name.endswith(".whl")]
        assert len(wheels) == 1, (
            f"Archive must contain exactly one .whl file, "
            f"found {len(wheels)}: {sorted(names)}"
        )


@contextmanager
def make_zip_context(build_dir):
    # Determinism: fixed mtime, sorted iteration, fixed permissions. Two
    # builds of the same expression with identical member bytes must produce
    # byte-identical zip archives, so downstream content-addressed storage
    # (git-annex, etc.) dedups them.
    path_prefix = (build_dir := Path(build_dir)).parent
    fixed_date_time = (1980, 1, 1, 0, 0, 0)
    fixed_external_attr = 0o644 << 16
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td).joinpath(build_dir.name).with_suffix(PREFERRED_SUFFIX)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(build_dir.rglob("*")):
                if p.is_file():
                    info = zipfile.ZipInfo(
                        filename=str(p.relative_to(path_prefix)),
                        date_time=fixed_date_time,
                    )
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = fixed_external_attr
                    zf.writestr(info, p.read_bytes())
        yield zip_path


def extract_build_zip_to(zip_path, td):
    """Extract a build zip into `td` and return the single top-level build dir."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(td)
    (build_dir, *rest) = Path(td).iterdir()
    assert not rest
    return build_dir


@contextmanager
def extract_build_zip_context(zip_path):
    with tempfile.TemporaryDirectory() as td:
        yield extract_build_zip_to(zip_path, td)


def write_zip(path, relpath_to_bytes):
    path = Path(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for relpath, byts in dict(relpath_to_bytes).items():
            zf.writestr(str(relpath), byts)
    return path


@frozen
class BuildZip:
    """A validated build archive.  Checks suffix and required archive members on init."""

    path = field(validator=instance_of(Path), converter=Path)

    def __attrs_post_init__(self):
        assert "".join(self.path.suffixes) in VALID_SUFFIXES, (
            f"Invalid archive suffix '{self.path.suffixes}', expected one of {VALID_SUFFIXES}"
        )
        assert self.path.exists(), f"Build archive not found at {self.path}"
        test_zip(self.path)

    @property
    def name(self):
        return with_pure_suffix(self.path, "").name

    @cached_property
    def internal_prefix(self):
        """The top-level directory inside the zip archive."""
        with zipfile.ZipFile(self.path, "r") as zf:
            first = zf.namelist()[0]
            return first.split("/", 1)[0]

    @property
    def md5sum(self):
        from xorq.common.utils.dask_normalize.dask_normalize_utils import (  # noqa: PLC0415
            file_digest,
        )

        return file_digest(self.path, hashlib.md5)

    def read_member(self, member_path, read_f):
        """Read and parse a single member from the zip archive."""
        with zipfile.ZipFile(self.path, "r") as zf:
            return read_f(zf.read(member_path).decode())
