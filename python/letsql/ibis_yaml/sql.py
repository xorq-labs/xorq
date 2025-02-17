from typing import Any, Dict, TypedDict

import ibis
import ibis.expr.operations as ops
import ibis.expr.types as ir

from letsql.expr.relations import RemoteTable


class QueryInfo(TypedDict):
    engine: str
    profile_name: str
    sql: str


class SQLPlans(TypedDict):
    queries: Dict[str, QueryInfo]


def find_remote_tables(op) -> Dict[str, Dict[str, Any]]:
    remote_tables = {}
    seen = set()

    def traverse(node):
        if node is None or id(node) in seen:
            return

        seen.add(id(node))

        if isinstance(node, ops.Node) and isinstance(node, RemoteTable):
            remote_expr = node.remote_expr
            original_backend = remote_expr._find_backend()
            if (
                not hasattr(original_backend, "profile_name")
                or original_backend.profile_name is None
            ):
                raise AttributeError(
                    "Backend does not have a valid 'profile_name' attribute."
                )

            engine_name = original_backend.name
            profile_name = original_backend.profile_name
            remote_tables[node.name] = {
                "engine": engine_name,
                "profile_name": profile_name,
                "sql": ibis.to_sql(remote_expr),
            }

        if isinstance(node, ops.Node):
            for arg in node.args:
                if isinstance(arg, ops.Node):
                    traverse(arg)
                elif isinstance(arg, (list, tuple)):
                    for item in arg:
                        if isinstance(item, ops.Node):
                            traverse(item)
                elif isinstance(arg, dict):
                    for v in arg.values():
                        if isinstance(v, ops.Node):
                            traverse(v)

    traverse(op)
    return remote_tables


# TODO: rename to sqls
def generate_sql_plans(expr: ir.Expr) -> SQLPlans:
    remote_tables = find_remote_tables(expr.op())

    main_sql = ibis.to_sql(expr)
    backend = expr._find_backend()

    if not hasattr(backend, "profile_name") or backend.profile_name is None:
        raise AttributeError("Backend does not have a valid 'profile_name' attribute.")

    engine_name = backend.name
    profile_name = backend.profile_name

    plans: SQLPlans = {
        "queries": {
            "main": {
                "engine": engine_name,
                "profile_name": profile_name,
                "sql": main_sql.strip(),
            }
        }
    }

    for table_name, info in remote_tables.items():
        plans["queries"][table_name] = {
            "engine": info["engine"],
            "profile_name": info["profile_name"],
            "sql": info["sql"].strip(),
        }

    return plans
