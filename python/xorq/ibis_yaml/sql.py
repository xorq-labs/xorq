from typing import Any, Dict, List, Tuple, TypedDict

import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.types as ir
from xorq.common.utils.graph_utils import find_all_sources, walk_nodes
from xorq.expr.relations import Read, RemoteTable


class QueryInfo(TypedDict):
    engine: str
    profile_name: str
    sql: str


class SQLPlans(TypedDict):
    queries: Dict[str, QueryInfo]


class DeferredReadsPlan(TypedDict):
    reads: Dict[str, QueryInfo]


def find_relations(expr: ir.Expr) -> List[str]:
    node_types = (RemoteTable, Read, ops.DatabaseTable)
    nodes = walk_nodes(node_types, expr)
    relations = []
    seen = set()
    for node in nodes:
        name = None
        if isinstance(node, RemoteTable):
            name = node.name
        elif isinstance(node, Read):
            name = node.make_unbound_dt().name
        elif isinstance(node, ops.DatabaseTable):
            name = node.name
        if name and name not in seen:
            seen.add(name)
            relations.append(name)
    return relations


def find_tables(expr: ir.Expr) -> Tuple[Dict[str, QueryInfo], Dict[str, QueryInfo]]:
    remote_tables: Dict[str, QueryInfo] = {}
    deferred_reads: Dict[str, QueryInfo] = {}

    node_types = (RemoteTable, Read)
    nodes = walk_nodes(node_types, expr)

    for node in nodes:
        if isinstance(node, RemoteTable):
            remote_expr = node.remote_expr
            backends = find_all_sources(node)
            if len(backends) > 1:
                backends = tuple(
                    x for x in backends if x != node.to_expr()._find_backend()
                )
            for backend in backends:
                engine_name = backend.name
                profile_name = backend._profile.hash_name
                key = f"{node.name}"
                remote_tables[key] = {
                    "engine": engine_name,
                    "profile_name": profile_name,
                    "relations": find_relations(remote_expr),
                    "sql": ibis.to_sql(remote_expr).strip(),
                    "options": {},
                }
        elif isinstance(node, Read):
            backend = node.source
            if backend is not None:
                dt = node.make_unbound_dt()
                key = dt.name
                deferred_reads[key] = {
                    "engine": backend.name,
                    "profile_name": backend._profile.hash_name,
                    "relations": [dt.name],
                    "sql": ibis.to_sql(dt.to_expr()).strip(),
                    "options": get_read_options(node),
                }
    return remote_tables, deferred_reads


def get_read_options(read_instance) -> Dict[str, Any]:
    read_kwargs_list = [{k: v} for k, v in read_instance.read_kwargs]
    return {
        "method_name": read_instance.method_name,
        "name": read_instance.name,
        "read_kwargs": read_kwargs_list,
    }


def generate_sql_plans(expr: ir.Expr) -> Tuple[SQLPlans, DeferredReadsPlan]:
    remote_tables, deferred_reads = find_tables(expr)
    main_sql = ibis.to_sql(expr)
    backend = expr._find_backend()

    queries: Dict[str, QueryInfo] = {
        "main": {
            "engine": backend.name,
            "profile_name": backend._profile.hash_name,
            "relations": find_relations(expr),
            "sql": main_sql.strip(),
            "options": {},
        }
    }

    for table_name, info in remote_tables.items():
        queries[table_name] = info

    sql_plans: SQLPlans = {"queries": queries}
    deferred_reads_plans: DeferredReadsPlan = {"reads": deferred_reads}
    return sql_plans, deferred_reads_plans
