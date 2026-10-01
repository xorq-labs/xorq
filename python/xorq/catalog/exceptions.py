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


class WheelCollisionError(ValueError):
    """Raised when two entries carry same-named wheels that differ."""


class RebaseError(XorqError):
    """A rebase that refused or failed; ``exit_code`` is the CLI's, a
    ``RebaseExit``. Nothing of its own was written, except where the subclass
    or the message says so: a failed rollback names what it left. One raised
    after a sync's pull leaves what the pull merged in place, unpushed."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message, exit_code)

    def __str__(self) -> str:
        return self.args[0]

    @property
    def exit_code(self) -> int:
        return self.args[1]


class RebasePushError(RebaseError):
    """The rebase was committed locally but not pushed; ``result`` is its
    ``RebaseResult``."""

    def __init__(self, message: str, exit_code: int, result: object) -> None:
        super().__init__(message, exit_code)
        self.result = result


class AliasRefusedError(XorqError):
    """A derived entry's alias request can't be met; nothing was written."""


class PullError(XorqError):
    """The sync's pull before a derived entry failed; nothing was written."""


class PushError(XorqError):
    """The push failed after the local commit; ``derived`` is what was committed,
    a ``derive.Derived``."""

    def __init__(self, message: str, derived: object) -> None:
        super().__init__(message, derived)
        self.derived = derived

    def __str__(self) -> str:
        return self.args[0]


class RollbackError(XorqError):
    """A failed alias move could not be rolled back; ``state`` says what is left."""

    def __init__(self, message: str, state: str) -> None:
        super().__init__(message, state)
        self.state = state

    def __str__(self) -> str:
        return self.args[0]
