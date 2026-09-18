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
    table maintained beside it, so the two cannot drift apart."""

    EQUAL = "equal"
    UNREACHABLE = "unreachable"
    UNREADABLE = "unreadable"
    CHANGED = "changed"
    TABLE_MISSING = "table-missing"

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
