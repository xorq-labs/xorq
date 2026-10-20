from __future__ import annotations

from xorq.common.exceptions import XorqError


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
    """A rebase refused before writing anything; ``exit_code`` is the CLI's,
    a ``RebaseExit``."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message, exit_code)

    def __str__(self) -> str:
        return self.args[0]

    @property
    def exit_code(self) -> int:
        return self.args[1]


class LoginError(XorqError):
    """Signing in to a hosted catalog service failed."""


class LoginTimeoutError(LoginError):
    """An identity provider or catalog service did not answer in time."""
