try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class DumpFiles(StrEnum):
    deferred_reads = "deferred_reads.yaml"
    expr = "expr.yaml"
    expr_signature = "expr_signature.json"
    metadata = "metadata.json"
    profiles = "profiles.yaml"
    sql = "sql.yaml"


REQUIRED_TGZ_NAMES = (
    DumpFiles.expr,
    DumpFiles.expr_signature,
    DumpFiles.metadata,
    DumpFiles.profiles,
)


class ExprKind(StrEnum):
    Expr = "expr"
    UnboundExpr = "unbound_expr"


class MemtableTypes(StrEnum):
    inmemory = "memtables"
    database_table = "database_tables"
