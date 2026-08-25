import warnings
from typing import Any, Dict, List, Tuple, TypedDict

import toolz

import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.types as ir
from xorq.common.exceptions import XorqError
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import Read, RemoteTable
from xorq.vendor.ibis.expr.types.core import SqlQueries


class QueryInfo(TypedDict):
    engine: str
    profile_name: str
    sql: str
    # every query's referenced relation names; _extract_sql_queries records
    # them into expr_metadata and the TUI SQL DAG orders queries from them
    relations: List[str]
    options: Dict[str, Any]


class SQLPlans(TypedDict):
    queries: Dict[str, QueryInfo]


class DeferredReadsPlan(TypedDict):
    reads: Dict[str, QueryInfo]


def to_sql(expr: ir.Expr) -> str:
    from xorq.expr.api import _remove_tee_nodes  # noqa: PLC0415

    # A TeeNode is a transparent, schema-preserving pass-through with no SQL
    # compiler visitor; strip it to its parent before compiling (the same
    # non-executing treatment the `.sql()` view path applies).
    uncached = _remove_tee_nodes(expr.ls.uncached)
    try:
        compiler_provider = (
            uncached._find_backend(  # xorq-style: disable=protected-access
                use_default=True
            )
        )
        if getattr(compiler_provider, "compiler", None) is None:
            warnings.warn(
                f"{compiler_provider} is not a SQL backend, so no SQL string will be generated",
                stacklevel=2,
            )
            return ""
    except XorqError:
        pass

    return ibis.to_sql(uncached)


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


def sql_query_deps(sql_queries: SqlQueries) -> Dict[str, frozenset]:
    """name → the query names it depends on, from recorded relations.

    Relations list every relation a query references: plain source tables
    (not queries here), the query's own name (deferred reads), and possibly
    "main" (a source table named like duckdb's default schema). Only
    references to other recorded queries are edges; "main" is the root plan
    key and never a dependency, so a source table named "main" cannot
    introduce a cycle. These rules live here, next to the producer of
    relations, so consumers do not each rediscover them.
    """
    names = {q[0] for q in sql_queries}
    return {
        name: frozenset(
            ref for ref in relations if ref not in (name, "main") and ref in names
        )
        for name, _, _, relations in sql_queries
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
