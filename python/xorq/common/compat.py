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


def raise_collected_errors(message: str, errors: list[BaseException]) -> None:
    """Raise multiple errors as a group, preserving natural order.

    Uses ``BaseExceptionGroup`` on 3.11+; on 3.10, chains via ``__cause__``
    so the first error prints first and the last is the raised exception.
    """
    if not errors:
        return
    if len(errors) == 1:
        raise errors[0]
    if sys.version_info >= (3, 11):
        raise BaseExceptionGroup(message, errors)  # noqa: F821
    for i in range(len(errors) - 1):
        errors[i + 1].__cause__ = errors[i]
    raise errors[-1]
