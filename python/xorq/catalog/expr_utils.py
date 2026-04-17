import atexit
import shutil
import tempfile
import weakref
from contextlib import contextmanager

from xorq.catalog.zip_utils import (
    extract_build_zip_to,
    make_zip_context,
)


# Tracks temp dirs that haven't been cleaned up yet.
# weakref.finalize handles per-expression cleanup; atexit sweeps stragglers.
_live_extract_dirs: set[str] = set()


def _cleanup_one(path: str):
    shutil.rmtree(path, ignore_errors=True)
    _live_extract_dirs.discard(path)


def _cleanup_all():
    for p in tuple(_live_extract_dirs):
        _cleanup_one(p)


atexit.register(_cleanup_all)


@contextmanager
def build_expr_context(expr):
    from xorq.ibis_yaml.compiler import build_expr  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        build_dir = build_expr(expr, builds_dir=td)
        yield build_dir


@contextmanager
def build_expr_context_zip(expr):
    with build_expr_context(expr) as build_dir:
        with make_zip_context(build_dir) as zip_path:
            yield zip_path


def load_expr_from_zip(zip_path, lazy=False, read_only_parquet_metadata=False):
    from xorq.ibis_yaml.compiler import load_expr  # noqa: PLC0415

    td = tempfile.mkdtemp(prefix="xorq-catalog-")
    _live_extract_dirs.add(td)
    try:
        build_dir = extract_build_zip_to(zip_path, td)
        # Invariant: `load_expr` must eagerly materialize all IO from
        # `build_dir`. The extract dir's lifetime is pinned to `expr` via
        # `weakref.finalize` below, so any lazy reference to files under
        # `build_dir` will break once `expr` is garbage-collected.
        expr = load_expr(
            build_dir,
            lazy=lazy,
            read_only_parquet_metadata=read_only_parquet_metadata,
        )
    except BaseException:
        _cleanup_one(td)
        raise

    weakref.finalize(expr, _cleanup_one, td)
    return expr
