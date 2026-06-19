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
        """Acquire an exclusive lock on an open file (Windows).

        Platform difference: ``msvcrt.LK_LOCK`` retries ~10 times at 1-second
        intervals and then raises ``OSError`` if the lock is still held. The
        POSIX branch (``fcntl.flock(LOCK_EX)``) blocks indefinitely instead.
        Under heavy concurrent contention (many writers appending to the same
        file), Windows callers can therefore see an ``OSError`` after ~10s
        where POSIX callers would simply keep queueing.
        """
        msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
else:
    import fcntl

    def flock_exclusive(fd: IO) -> None:
        """Acquire an exclusive lock on an open file (POSIX).

        Blocks indefinitely until the lock is acquired. See the Windows branch
        for a platform difference: Windows raises after a ~10s timeout under
        contention rather than blocking.
        """
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
