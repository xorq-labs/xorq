try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class DumpFiles(StrEnum):
    deferred_reads = "deferred_reads.yaml"
    expr = "expr.yaml"
    expr_metadata = "expr_metadata.json"
    build_metadata = "build_metadata.json"
    profiles = "profiles.yaml"
    sql = "sql.yaml"


REQUIRED_ARCHIVE_NAMES = (
    DumpFiles.expr,
    DumpFiles.expr_metadata,
    DumpFiles.build_metadata,
    DumpFiles.profiles,
)


class ExprKind(StrEnum):
    Source = "source"
    Expr = "expr"
    UnboundExpr = "unbound_expr"
    Composed = "composed"


class MemtableTypes(StrEnum):
    inmemory = "memtables"
    database_table = "database_tables"
