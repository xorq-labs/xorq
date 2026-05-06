from __future__ import annotations

import cProfile
import inspect
import pstats
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pstats import SortKey


def profile(stmt: str, field: SortKey | None = SortKey.CUMULATIVE) -> pstats.Stats:
    f_back = inspect.currentframe().f_back
    with tempfile.NamedTemporaryFile() as ntf:
        cProfile.runctx(
            stmt,
            globals=f_back.f_globals,
            locals=f_back.f_locals,
            filename=ntf.name,
        )
        p = pstats.Stats(ntf.name)
    if field is not None:
        p = p.sort_stats(field)
    return p


@contextmanager
def timed() -> Iterator[Callable[[], float]]:
    t = time.monotonic()
    yield lambda: time.monotonic() - t
