from __future__ import annotations

import atexit
import shutil
import tempfile
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from xorq.catalog.zip_utils import (
    extract_build_zip_to,
    make_zip_context,
)
from xorq.common.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from xorq.api import Expr


logger = get_logger(__name__)


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
def build_expr_context(expr: Expr, relocate_reads: bool = False) -> Iterator[Path]:
    from xorq.ibis_yaml.compiler import build_expr  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        build_dir = build_expr(expr, builds_dir=td, relocate_reads=relocate_reads)
        yield build_dir


@contextmanager
def build_expr_context_zip(expr, project_path=None):
    from xorq.catalog.catalog import _ensure_wheel_artifacts  # noqa: PLC0415

    with build_expr_context(expr) as build_dir:
        _ensure_wheel_artifacts(build_dir, project_path=project_path)
        with make_zip_context(build_dir) as zip_path:
            yield zip_path


# Attribute used to pin a loaded expr onto the backends it reads from, so the
# expr (and its extract dir) outlives every expression derived from it. See
# `_pin_extract_dir_lifetime`.
_EXTRACT_DIR_ANCHOR_ATTR = "_xorq_extract_dir_anchors"


def _pin_extract_dir_lifetime(expr: "Expr", td: str) -> None:
    """Tie extract dir ``td`` to the lifetime of ``expr``'s backends.

    ``weakref.finalize(expr, ...)`` alone breaks once ``bind()`` /
    ``fuse_catalog_source`` rewrap the graph: they keep the op nodes (whose reads
    point into ``td``) but drop the exact ``expr`` wrapper, firing the finalizer
    while those paths are still live (#2133). The reads' ``.source`` backend
    survives rewrapping, so pin ``expr`` onto each backend it reads from; ``td``
    is then swept only once every expression derived from ``expr`` is gone.
    Backends are fresh per load (``Profile.get_con`` connects anew).

    Note: anchoring makes ``expr``/``backend`` mutually referential, so ``td`` is
    reclaimed by cyclic GC (or ``atexit``) rather than promptly at refcount-zero;
    this is invisible for one-shot CLI runs but lets dirs linger in long-lived
    processes (notebooks, servers) until a collection.
    """
    backends, _ = expr._find_backends()  # xorq-style: disable=protected-access
    if not backends:
        # No backend to anchor onto, so only weakref.finalize remains -- the
        # pre-#2133 behavior fuse/bind rewrapping defeats, silently reopening the
        # empty-result bug. No load hits this today; warn if one does.
        logger.warning(
            "expr load produced no backends to anchor its extract dir onto; a "
            "fused/rebound expr may resolve relocated reads to an empty result "
            "if its extract dir is swept early (#2133)",
        )
    for backend in backends:
        try:
            anchors = getattr(backend, _EXTRACT_DIR_ANCHOR_ATTR, None)
            if anchors is None:
                anchors = []
                setattr(backend, _EXTRACT_DIR_ANCHOR_ATTR, anchors)
            anchors.append(expr)
        except AttributeError:
            # Can't anchor `expr`, so only weakref.finalize remains -- the
            # pre-#2133 behavior fuse/bind rewrapping defeats, silently reopening
            # the empty-result bug. No backend hits this today; warn if one does.
            logger.warning(
                "backend %r rejected extract-dir anchoring; a fused/rebound "
                "expr from this load may resolve relocated reads to an empty "
                "result if its extract dir is swept early (#2133)",
                type(backend).__name__,
            )
            continue
    weakref.finalize(expr, _cleanup_one, td)


def load_expr_from_zip(
    zip_path: Path | str,
    lazy: bool = False,
    read_only_parquet_metadata: bool = False,
    cache_dir: str | None = None,
) -> "Expr":
    from xorq.ibis_yaml.compiler import load_expr  # noqa: PLC0415

    td = tempfile.mkdtemp(prefix="xorq-catalog-")
    _live_extract_dirs.add(td)
    try:
        build_dir = extract_build_zip_to(zip_path, td)
        # Invariant: `load_expr` must eagerly materialize all IO from
        # `build_dir`. The extract dir's lifetime is anchored to the loaded
        # expr's backends (see `_pin_extract_dir_lifetime`), so any lazy
        # reference to files under `build_dir` breaks only once every expression
        # derived from `expr` -- including a fused one -- is gone.
        expr = load_expr(
            build_dir,
            lazy=lazy,
            read_only_parquet_metadata=read_only_parquet_metadata,
            cache_dir=cache_dir,
        )
    except BaseException:
        _cleanup_one(td)
        raise

    _pin_extract_dir_lifetime(expr, td)
    return expr
