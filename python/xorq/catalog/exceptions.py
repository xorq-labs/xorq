from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from xorq.catalog.enums import RebaseFailure
from xorq.common.exceptions import XorqError


if TYPE_CHECKING:
    from xorq.catalog.drift import LeafReport
    from xorq.catalog.inspection import BuildRecord


class ContentIntegrityError(XorqError):
    """Raised when content does not match the expected checksum."""


class ContentStoreError(XorqError):
    """Raised when an external content store operation fails."""


class ContentStoreCapabilityError(ContentStoreError):
    """Raised when a content-store operation is unsupported."""


class CatalogServiceHTTPError(ContentStoreError):
    """Raised when the hosted catalog service returns an HTTP error status.

    Every field lives in ``args``, as with ``RebaseError``: ``BaseException``
    pickles as ``cls(*args)``, so a field kept only as an attribute would make
    the error unpicklable across a process pool.
    """

    def __init__(
        self,
        message: str,
        status: int,
        error_code: str | None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message, status, error_code, retry_after)

    def __str__(self) -> str:
        return self.args[0]

    @property
    def status(self) -> int:
        return self.args[1]

    @property
    def error_code(self) -> str | None:
        return self.args[2]

    @property
    def retry_after(self) -> int | None:
        """Seconds the service asked the client to wait, if it said."""
        return self.args[3]


class CatalogPushError(RuntimeError):
    """Raised when ``catalog.push()`` cannot publish to a remote."""


class CatalogConfigurationError(RuntimeError):
    """Raised when the catalog's underlying repo violates a supported configuration.

    Fires on a sync-side operation (``push`` / ``pull`` / ``fetch`` /
    ``sync``) when the catalog finds more than one git remote (the catalog
    supports at most one, per ADR-0011), or when a hosted catalog's remote
    fails its binding check.
    """


class WheelCollisionError(ValueError):
    """Raised when two entries carry same-named wheels that differ."""


class RebaseError(XorqError):
    """A rebase refused before its own first write; ``failure`` says how, and
    its exit code is the CLI's. One raised after the sweep carries the sweep's
    ``record`` and ``reports``, so it reports the drift a conflict would. One
    raised after a sync's pull leaves what the pull merged in place, unpushed."""

    def __init__(
        self,
        message: str,
        failure: RebaseFailure,
        record: BuildRecord | None = None,
        reports: Iterable[LeafReport] = (),
    ) -> None:
        # `FAILED` is any error that is not a refusal; a refusal names which.
        if failure not in (RebaseFailure.REFUSED, RebaseFailure.UNREACHABLE):
            raise ValueError(f"a rebase refuses or is unreachable, not {failure}")
        super().__init__(message, RebaseFailure(failure), record, tuple(reports))

    def __str__(self) -> str:
        return self.args[0]

    @property
    def failure(self) -> RebaseFailure:
        return self.args[1]

    @property
    def record(self) -> BuildRecord | None:
        return self.args[2]

    @property
    def reports(self) -> tuple[LeafReport, ...]:
        return self.args[3]

    @property
    def exit_code(self) -> int:
        return self.failure.exit_code

    def after_sweep(
        self, record: BuildRecord, reports: Iterable[LeafReport]
    ) -> RebaseError:
        """This refusal, carrying the sweep it came after."""
        return type(self)(str(self), self.failure, record, reports)
