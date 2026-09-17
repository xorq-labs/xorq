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

    Values are the exact class names as serialized into ``expr.yaml``'s ``op``
    field.  Do NOT replace call sites with ``isinstance(node,
    ops.DatabaseTable)``: ``Read``, ``DatabaseTableView``, ``CachedNode``,
    ``RemoteTable``, ``FlightExpr`` and ``FlightUDXF`` all subclass it, so an
    isinstance test silently swallows caches, cross-backend transfers and
    Flight nodes -- none of which are external sources.
    """

    DATABASE_TABLE = "DatabaseTable"
    READ = "Read"


class PinKey(StrEnum):
    """The serialized ``CacheTag`` members a source walk must treat specially.

    ``OP`` is the ``op`` value ``_cache_tag_to_yaml`` writes; the other two are
    that node def's edges.  ``PARENT`` is the frozen read of the cache artifact
    and ``UNCACHED`` the upstream the pin discarded -- neither is a source of
    the record, so both are cut where ``graph_utils`` cuts them for hashing.
    Compared as strings for the same reason as ``LeafKind``: the walk reads
    dicts and never constructs a node.
    """

    OP = "CacheTag"
    PARENT = "parent"
    UNCACHED = "uncached"
