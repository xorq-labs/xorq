from __future__ import annotations

import enum

from xorq.common.compat import StrEnum


class WritePhase(enum.IntEnum):
    """Ordering for deferred writes; lower phases run first.

    DATA content files must exist before the expr YAML is written, because
    the translator tokenizes memtable parquets by content on the way out.
    """

    DATA = 0  # parquet + copied read files — tokenized by the expr YAML
    ARTIFACT = 1  # metadata / profiles / debug SQL — order-independent
    EXPR = 2  # expr YAML — must follow DATA so its inputs exist


class DumpFiles(StrEnum):
    deferred_reads = "deferred_reads.yaml"
    expr = "expr.yaml"
    expr_metadata = "expr_metadata.json"
    build_metadata = "build_metadata.json"
    profiles = "profiles.yaml"
    sql = "sql.yaml"
    requirements = "requirements.txt"


REQUIRED_ARCHIVE_NAMES = (
    DumpFiles.expr,
    DumpFiles.expr_metadata,
    DumpFiles.build_metadata,
    DumpFiles.profiles,
    DumpFiles.requirements,
)


class ExprKind(StrEnum):
    Source = "source"
    Expr = "expr"
    UnboundExpr = "unbound_expr"
    Composed = "composed"
    ExprBuilder = "expr_builder"


class BundledSourceTypes(StrEnum):
    inmemory = "memtables"
    database_table = "database_tables"
    read = "reads"


class DocKey(StrEnum):
    """Top-level keys of a serialized expression document (``expr.yaml``),
    written and read back by ``YamlExpressionTranslator``.
    """

    definitions = "definitions"
    expression = "expression"


class NodeKey(StrEnum):
    """Keys of the ``DatabaseTable`` and ``Read`` node defs, plus the hash the
    ``Registry`` stamps on them.

    Other ops still spell their keys out; this covers that subset only.

    Keys only: the ``op`` *values* stay literals (``"DatabaseTable"``,
    ``"Read"``). The enum naming them is ``xorq.catalog.enums.LeafKind``, and
    ``ibis_yaml`` must not import ``catalog`` -- the dependency runs the other
    way -- so that boundary is deliberate, not an oversight.
    """

    op = "op"
    name = "name"
    table = "table"
    profile = "profile"
    namespace = "namespace"
    method_name = "method_name"
    read_kwargs = "read_kwargs"
    normalize_method = "normalize_method"
    snapshot_hash = "snapshot_hash"


class ReadKwarg(StrEnum):
    """Serialized ``read_kwargs`` keys that name a ``Read``'s source."""

    hash_path = "hash_path"
    read_path = "read_path"
    table_name = "table_name"


class NamespaceKey(StrEnum):
    """Keys of a ``DatabaseTable``'s serialized namespace."""

    catalog = "catalog"
    database = "database"


class RefEnum(StrEnum):
    dtype_ref = "dtype_ref"
    node_ref = "node_ref"
    schema_ref = "schema_ref"


class RegistryEnum(StrEnum):
    dtypes = "dtypes"
    nodes = "nodes"
    schemas = "schemas"
