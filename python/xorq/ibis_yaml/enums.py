from xorq.common.compat import StrEnum


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


class RefEnum(StrEnum):
    dtype_ref = "dtype_ref"
    node_ref = "node_ref"
    schema_ref = "schema_ref"


class RegistryEnum(StrEnum):
    dtypes = "dtypes"
    nodes = "nodes"
    schemas = "schemas"
