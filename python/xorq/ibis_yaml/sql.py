import warnings
from typing import Any, Dict, List, Tuple, TypedDict

import toolz

import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.types as ir
from xorq.common.exceptions import XorqError
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import Read, RemoteTable


class QueryInfo(TypedDict):
    engine: str
    profile_name: str
    sql: str


class SQLPlans(TypedDict):
    queries: Dict[str, QueryInfo]


class DeferredReadsPlan(TypedDict):
    reads: Dict[str, QueryInfo]


def to_sql(expr: ir.Expr) -> str:
    try:
        compiler_provider = expr._find_backend(use_default=True)
        if getattr(compiler_provider, "compiler", None) is None:
            warnings.warn(
                f"{compiler_provider} is not a SQL backend, so no SQL string will be generated"
            )
            return ""
    except XorqError:
        pass

    return ibis.to_sql(expr.ls.uncached)


def find_relations(expr: ir.Expr) -> List[str]:
    def get_name(node):
        name = None
        if isinstance(node, RemoteTable):
            name = node.name
        elif isinstance(node, Read):
            name = node.make_unbound_dt().name
        elif isinstance(node, ops.DatabaseTable):
            name = node.name
        return name

    node_types = (RemoteTable, Read, ops.DatabaseTable)
    nodes = walk_nodes(node_types, expr)
    relations = sorted(set(filter(None, map(get_name, nodes))))
    return relations


def find_tables(expr: ir.Expr) -> Tuple[Dict[str, QueryInfo], Dict[str, QueryInfo]]:
    def get_remote_table_backend(node):
        return node.remote_expr._find_backend()

    grouped = toolz.groupby(type, walk_nodes((RemoteTable, Read), expr))
    remote_tables: Dict[str, QueryInfo] = {
        node.name: {
            "engine": backend.name,
            "profile_name": backend._profile.hash_name,
            "relations": find_relations(node.remote_expr),
            "sql": to_sql(node.remote_expr).strip(),
            "options": {},
        }
        for node in grouped.get(RemoteTable, ())
        if (backend := get_remote_table_backend(node))
    }
    deferred_reads: Dict[str, QueryInfo] = {
        dt.name: {
            "engine": backend.name,
            "profile_name": backend._profile.hash_name,
            "relations": [dt.name],
            "sql": to_sql(dt.to_expr()).strip(),
            "options": get_read_options(node),
        }
        for node in grouped.get(Read, ())
        if (backend := node.source) and (dt := node.make_unbound_dt())
    }
    remote_tables = dict(sorted(remote_tables.items()))
    deferred_reads = dict(sorted(deferred_reads.items()))
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
    backend = expr._find_backend()

    queries: Dict[str, QueryInfo] = {
        "main": {
            "engine": backend.name,
            "profile_name": backend._profile.hash_name,
            "relations": find_relations(expr),
            "sql": to_sql(expr).strip(),
            "options": {},
        }
    } | remote_tables

    sql_plans: SQLPlans = {"queries": queries}
    deferred_reads_plans: DeferredReadsPlan = {"reads": deferred_reads}
    return sql_plans, deferred_reads_plans
