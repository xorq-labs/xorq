import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path

from xorq.catalog.constants import (
    PREFERRED_SUFFIX,
    REQUIRED_TGZ_NAMES,
)


def test_tgz(tgz_path):
    with tarfile.TarFile.gzopen(tgz_path) as tfh:
        # https://docs.python.org/3/library/tarfile.html#cmdoption-tarfile-t
        relpaths = tuple(
            Path(member.path) for member in tfh.getmembers() if member.isfile()
        )
        missing = set(REQUIRED_TGZ_NAMES).difference(path.name for path in relpaths)
        assert not missing, missing


@contextmanager
def make_tgz_context(build_dir):
    path_prefix = (build_dir := Path(build_dir)).parent
    with tempfile.TemporaryDirectory() as td:
        tgz_path = Path(td).joinpath(build_dir.name).with_suffix(PREFERRED_SUFFIX)
        with tarfile.TarFile.gzopen(tgz_path, "w") as tf:
            for p in build_dir.iterdir():
                tf.add(p, arcname=p.relative_to(path_prefix))
        yield tgz_path


@contextmanager
def extract_build_tgz_context(tgz_path):
    with tempfile.TemporaryDirectory() as td:
        with tarfile.TarFile.gzopen(tgz_path) as tfh:
            tfh.extractall(td, filter="data")
        (path, *rest) = Path(td).iterdir()
        assert not rest
        yield path


def write_tgz(path, relpath_to_bytes):
    with tempfile.TemporaryDirectory() as td:
        with tarfile.TarFile.gzopen((path := Path(path)), "w") as tfh:
            for relpath, byts in dict(relpath_to_bytes).items():
                p = Path(td).joinpath(relpath)
                p.parent.mkdir(exist_ok=True, parents=True)
                p.write_bytes(byts)
                tfh.add(p, arcname=relpath)
    return path
