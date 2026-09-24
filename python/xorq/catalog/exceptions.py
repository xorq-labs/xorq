from xorq.common.exceptions import XorqError


class ContentIntegrityError(XorqError):
    """Raised when content does not match the expected checksum."""


class CatalogPushError(RuntimeError):
    """Raised when ``catalog.push()`` cannot publish to a remote."""


class CatalogConfigurationError(RuntimeError):
    """Raised when the catalog's underlying repo violates a supported configuration.

    Currently fires only when the catalog finds more than one git remote on
    a sync-side operation (``push`` / ``pull`` / ``fetch`` / ``sync``); the
    catalog supports at most one git remote per ADR-0011.
    """


class RebaseError(XorqError):
    """A rebase refused before writing anything; ``exit_code`` is the CLI's.

    1: it never started (pinned entry, Python-minor mismatch, unprobeable
    source). 2: a source or the record could not be read.
    """

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message, exit_code)

    def __str__(self) -> str:
        return self.args[0]

    @property
    def exit_code(self) -> int:
        return self.args[1]
