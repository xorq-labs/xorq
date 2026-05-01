class CatalogPushError(RuntimeError):
    """Raised when ``catalog.push()`` cannot publish to a remote."""


class CatalogConfigurationError(RuntimeError):
    """Raised when the catalog's underlying repo violates a supported configuration.

    Currently fires only when the catalog finds more than one git remote on
    a sync-side operation (``push`` / ``pull`` / ``fetch`` / ``sync``); the
    catalog supports exactly one git remote per ADR-0009.
    """
