import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

from xorq.catalog.constants import (
    PREFERRED_SUFFIX,
)
from xorq.ibis_yaml.enums import REQUIRED_ARCHIVE_NAMES


def test_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = {Path(name).name for name in zf.namelist() if not name.endswith("/")}
        missing = set(REQUIRED_ARCHIVE_NAMES).difference(names)
        assert not missing, missing


@contextmanager
def make_zip_context(build_dir):
    path_prefix = (build_dir := Path(build_dir)).parent
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td).joinpath(build_dir.name).with_suffix(PREFERRED_SUFFIX)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in build_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(path_prefix))
        yield zip_path


@contextmanager
def extract_build_zip_context(zip_path):
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(td)
        (path, *rest) = Path(td).iterdir()
        assert not rest
        yield path


def write_zip(path, relpath_to_bytes):
    path = Path(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for relpath, byts in dict(relpath_to_bytes).items():
            zf.writestr(str(relpath), byts)
    return path
