from __future__ import annotations

import sys
from typing import IO


try:
    from enum import StrEnum  # noqa: F401
except ImportError:
    from strenum import StrEnum  # noqa: F401


if sys.platform == "win32":
    import msvcrt

    def flock_exclusive(fd: IO) -> None:
        """Acquire an exclusive lock on an open file (Windows)."""
        msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
else:
    import fcntl

    def flock_exclusive(fd: IO) -> None:
        """Acquire an exclusive lock on an open file (POSIX)."""
        fcntl.flock(fd, fcntl.LOCK_EX)
