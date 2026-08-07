from __future__ import annotations

from xorq.common.exceptions import XorqError


class ContentIntegrityError(XorqError):
    """Raised when content does not match the expected checksum."""


class ContentStoreError(XorqError):
    """Raised when an external content store operation fails."""


class ContentStoreCapabilityError(ContentStoreError):
    """Raised when a content-store operation is unsupported."""


class CatalogServiceHTTPError(ContentStoreError):
    """Raised when the hosted catalog service returns an HTTP error status."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        error_code: str | None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_code = error_code


class CatalogPushError(RuntimeError):
    """Raised when ``catalog.push()`` cannot publish to a remote."""


class CatalogConfigurationError(RuntimeError):
    """Raised when the catalog's underlying repo violates a supported configuration.

    Currently fires only when the catalog finds more than one git remote on
    a sync-side operation (``push`` / ``pull`` / ``fetch`` / ``sync``); the
    catalog supports at most one git remote per ADR-0011.
    """
