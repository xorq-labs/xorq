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


class DriftState(StrEnum):
    """Verdict for one source leaf: the recorded schema vs the live schema."""

    equal = "equal"
    changed = "changed"
    table_missing = "table-missing"
    unreachable = "unreachable"


class LeafKind(StrEnum):
    """The two ``op`` names that denote an external source leaf.

    Values are the exact class names as serialized into ``expr.yaml``'s ``op``
    field.  Do NOT replace call sites with ``isinstance(node,
    ops.DatabaseTable)``: ``Read``, ``DatabaseTableView``, ``CachedNode``,
    ``RemoteTable``, ``FlightExpr`` and ``FlightUDXF`` all subclass it, so an
    isinstance test silently swallows caches, cross-backend transfers and
    Flight nodes -- none of which are external sources.
    """

    database_table = "DatabaseTable"
    read = "Read"
