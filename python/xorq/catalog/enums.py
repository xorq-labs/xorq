from enum import IntEnum

from xorq.common.compat import StrEnum


class CatalogInfix(StrEnum):
    ALIAS = "aliases"
    ENTRY = "entries"
    METADATA = "metadata"


class CatalogTag(StrEnum):
    SOURCE = "catalog-source"
    TRANSFORM = "catalog-transform"
    CODE = "catalog-code"


class ContentStoreType(StrEnum):
    DIRECTORY = "directory"
    S3 = "s3"


class OnUnrebuiltBuilder(StrEnum):
    """Policy when a builder tag has no rebuild protocol registered."""

    RAISE = "raise"
    WARN = "warn"


class LeafKind(StrEnum):
    """The two ``op`` names that denote an external source leaf.

    Values are the class names exactly as serialized into ``expr.yaml``'s ``op``
    field. Do NOT swap a call site for ``isinstance(node, ops.DatabaseTable)``:
    ``Read``, ``DatabaseTableView``, ``CachedNode``, ``RemoteTable``,
    ``FlightExpr`` and ``FlightUDXF`` all subclass it, so an isinstance test
    silently swallows caches, cross-backend transfers and Flight nodes. None of
    those is an external source.
    """

    DATABASE_TABLE = "DatabaseTable"
    READ = "Read"


class PinOp(StrEnum):
    """The ``op`` name that denotes a pin, as ``_cache_tag_to_yaml`` writes it.

    Its own enum rather than a member of ``PinKey`` because node-def key names
    and ``op`` names are different vocabularies. One enum holding both leaves
    every member set over it (``LEAF_OPS`` is the shape: ``frozenset`` of a whole
    op enum) with no way to say which vocabulary it means. Compared as a string
    for the same reason as ``LeafKind``: the walk reads dicts and never
    constructs a node.
    """

    CACHE_TAG = "CacheTag"


class PinKey(StrEnum):
    """The ``CacheTag`` node def's edges, as a source walk must treat them.

    ``PARENT`` is the frozen read of the cache artifact, ``UNCACHED`` the
    upstream the pin discarded. Neither is a source of the record, so both are
    cut where ``graph_utils`` cuts them for hashing. Keys only: every member is a
    prune-key and nothing else, and the op name lives in ``PinOp``.
    """

    PARENT = "parent"
    UNCACHED = "uncached"


class Verdict(StrEnum):
    """What a probe found. The exit code is a property of the verdict, not a
    table maintained beside it, so the two cannot drift apart.

    Declared worst-last: ``severity`` reads the order off this list, so a member
    added in the wrong place changes how a sweep rolls up and is caught by
    ``test_severity_refines_the_exit_code`` rather than by a consumer.
    """

    EQUAL = "equal"
    UNREACHABLE = "unreachable"
    UNREADABLE = "unreadable"
    CHANGED = "changed"
    TABLE_MISSING = "table-missing"

    @property
    def severity(self) -> int:
        """Where this verdict sorts when several roll up into one.

        A total order, which the exit code is not: `unreachable` shares 2 with
        `unreadable` and `changed` shares 3 with `table-missing`, so rolling up
        on the code alone leaves each tie to be broken by whatever order the
        verdicts happened to arrive in -- which for a sweep is the order the
        names were typed on the command line.

        It refines the code rather than reordering it, so the worst verdict and
        the worst exit code are always the same leaf. Within a code the more
        durable finding wins: `unreadable` over `unreachable`, because a record
        that will not parse is not fixed by retrying, and `table-missing` over
        `changed`, because the source is gone rather than different.
        """
        return tuple(type(self)).index(self)

    @property
    def exit_code(self) -> int:
        match self:
            case Verdict.EQUAL:
                return 0
            case Verdict.UNREACHABLE | Verdict.UNREADABLE:
                return 2
            case Verdict.CHANGED | Verdict.TABLE_MISSING:
                return 3
            # Every member is named above, so a new one fails here rather than
            # inheriting a catch-all's "drift" -- which is the drift this
            # property exists to prevent.
            case _:
                raise ValueError(f"no exit code for verdict {self}")


class RebaseStatus(StrEnum):
    """How a successful ``rebase_entry`` ended.

    ``ATTEMPTED``: no source could be probed, so the entry was re-derived and
    came back with its own hash; unlike ``NOOP``, that doesn't prove no drift.
    """

    NOOP = "noop"
    ATTEMPTED = "attempted"
    REBASED = "rebased"


class RebaseExit(IntEnum):
    """The exit codes ``xorq catalog rebase`` refuses with."""

    # Never started: a property of the entry or the request; retrying won't help.
    REFUSED = 1
    # A source or the record could not be read; retryable.
    UNREACHABLE = 2
    # The re-derivation can't follow the drift; retrying won't help.
    CONFLICT = 4
