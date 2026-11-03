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


class RebaseTag(StrEnum):
    """``HashingTag`` names ``xorq catalog rebase`` writes. Apart from
    ``CatalogTag``, which bind, replay and the composer strip."""

    # Over a `--rename`: holds the mapping, so a later refresh can rebuild it.
    RENAME = "rebase-rename"


class ContentStoreType(StrEnum):
    DIRECTORY = "directory"
    PRESIGNED = "presigned"
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
    """How a ``rebase_entry`` that raised no ``RebaseError`` ended.

    ``UNPROBED``: no source can be probed, so nothing could be refreshed and
    nothing was loaded or written; unlike ``NOOP``, that doesn't prove no drift.
    ``CONFLICT``: the re-derivation can't follow the drift, and nothing was
    written. ``PUSH_FAILED``: rebased as ``REBASED`` is, but the sync's push
    failed, so the new entry and its alias moves are committed only locally.
    ``PREVIEW``: a rebase would happen, and ``dry_run`` wrote nothing.
    ``BUILT``: the rebased build was written, and ``add=False`` left the
    catalog alone.
    """

    NOOP = "noop"
    UNPROBED = "unprobed"
    REBASED = "rebased"
    CONFLICT = "conflict"
    PUSH_FAILED = "push-failed"
    PREVIEW = "preview"
    BUILT = "built"

    @property
    def exit_code(self) -> int:
        match self:
            case RebaseStatus.NOOP | RebaseStatus.UNPROBED | RebaseStatus.REBASED:
                return 0
            case RebaseStatus.PREVIEW | RebaseStatus.BUILT:
                return RebaseExit.UNCATALOGED
            case RebaseStatus.CONFLICT:
                return RebaseExit.CONFLICT
            case RebaseStatus.PUSH_FAILED:
                return RebaseExit.PUSH_FAILED
            # As with `Verdict`: a new member fails here rather than exiting 0,
            # which a script would read as success.
            case _:
                raise ValueError(f"no exit code for rebase status {self}")


class RebaseExit(IntEnum):
    """The exit codes ``xorq catalog rebase`` refuses, previews or conflicts with."""

    # Nothing of the rebase's own written (a sync's pull may have merged). A
    # refusal is a property of the entry or the request, and retrying won't
    # help; a write that failed and was rolled back (`failed`) exits 1 too,
    # and may be transient: its `error` says which. So does one whose rollback
    # failed too (`rollback-failed`), the one exit 1 that may leave the local
    # catalog dirty: its `error` names what is left.
    REFUSED = 1
    # A source or the record could not be read; retryable.
    UNREACHABLE = 2
    # `--dry-run` or `--no-add`: a rebase was derived, and nothing cataloged.
    # Unlike 0, which on a no-op also prints a name, it says the name on
    # stdout is no entry (a hash, or a build directory).
    UNCATALOGED = 3
    # The re-derivation can't follow the drift; retrying won't help.
    CONFLICT = 4
    # Rebased locally, but the push failed; `xorq catalog sync` finishes it
    # once the push can succeed: a push the remote rejected needs the pull
    # first, one that failed a check needs that fixed.
    PUSH_FAILED = 5


class RebaseFailure(StrEnum):
    """How a rebase that returned no ``RebaseResult`` ended.

    ``REFUSED`` and ``UNREACHABLE`` are what a ``RebaseError`` carries;
    ``ROLLBACK_FAILED`` is a write that failed and could not be undone, so the
    local catalog may hold part of it; ``FAILED`` is any other error, such as
    a write that failed and was rolled back.
    """

    REFUSED = "refused"
    UNREACHABLE = "unreachable"
    ROLLBACK_FAILED = "rollback-failed"
    FAILED = "failed"

    @property
    def exit_code(self) -> int:
        match self:
            case (
                RebaseFailure.REFUSED
                | RebaseFailure.ROLLBACK_FAILED
                | RebaseFailure.FAILED
            ):
                return RebaseExit.REFUSED
            case RebaseFailure.UNREACHABLE:
                return RebaseExit.UNREACHABLE
            case _:
                raise ValueError(f"no exit code for failure {self}")
