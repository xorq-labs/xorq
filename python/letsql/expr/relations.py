import functools
from collections import defaultdict
from typing import Any

import ibis
import pyarrow as pa
from ibis import Schema, Expr
from ibis.common.collections import FrozenDict
from ibis.common.graph import Graph
from ibis.expr import operations as ops
from ibis.expr.operations import Relation, Node

from letsql.common.utils.graph_utils import replace_fix


def replace_cache_table(node, _, **kwargs):
    if isinstance(node, CachedNode):
        return kwargs["parent"].op().replace(replace_fix(replace_cache_table))
    elif isinstance(node, RemoteTable):
        return kwargs["remote_expr"].op().replace(replace_fix(replace_cache_table))
    else:
        return node.__recreate__(kwargs)


# https://stackoverflow.com/questions/6703594/is-the-result-of-itertools-tee-thread-safe-python
class SafeTee(object):
    """tee object wrapped to make it thread-safe"""

    def __init__(self, teeobj, lock):
        self.teeobj = teeobj
        self.lock = lock

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.teeobj)

    def __copy__(self):
        return SafeTee(self.teeobj.__copy__(), self.lock)

    @classmethod
    def tee(cls, iterable, n=2):
        """tuple of n independent thread-safe iterators"""
        from itertools import tee
        from threading import Lock

        lock = Lock()
        return tuple(cls(teeobj, lock) for teeobj in tee(iterable, n))


def recursive_update(obj, replacements):
    if isinstance(obj, Node):
        if obj in replacements:
            return replacements[obj]
        else:
            return obj.__recreate__(
                {
                    name: recursive_update(arg, replacements)
                    for name, arg in zip(obj.argnames, obj.args)
                }
            )
    elif isinstance(obj, (tuple, list)):
        return tuple(recursive_update(o, replacements) for o in obj)
    elif isinstance(obj, dict):
        return {
            recursive_update(k, replacements): recursive_update(v, replacements)
            for k, v in obj.items()
        }
    else:
        return obj


def replace_source_factory(source: Any):
    def replace_source(node, _, **kwargs):
        if "source" in kwargs:
            kwargs["source"] = source
        return node.__recreate__(kwargs)

    return replace_source


def make_native_op(node):
    # FIXME: how to reference let.Backend.name?
    if node.source.name != "let":
        raise ValueError
    sources = node.source._sources
    native_source = sources.get_backend(node)
    if native_source.name == "let":
        raise ValueError

    def replace_table(_node, _, **_kwargs):
        return sources.get_table_or_op(_node, _node.__recreate__(_kwargs))

    return node.replace(replace_fix(replace_table)).to_expr()


class CachedNode(ops.Relation):
    schema: Schema
    parent: Any
    source: Any
    storage: Any
    values = FrozenDict()


gen_name = functools.partial(ibis.util.gen_name, "remote-expr-placeholder")


class RemoteTable(ops.DatabaseTable):
    remote_expr: Expr = None

    @classmethod
    def from_expr(cls, con, expr, name=None):
        name = name or gen_name()
        return cls(
            name=name,
            schema=expr.schema(),
            source=con,
            remote_expr=expr,
        )


def into_backend(expr, con, name=None):
    return RemoteTable.from_expr(con=con, expr=expr, name=name).to_expr()


class Read(ops.Relation):
    method_name: str
    name: str
    schema: Schema
    source: Any
    read_kwargs: Any
    values = FrozenDict()

    def make_dt(self):
        method = getattr(self.source, self.method_name)
        dt = method(**dict(self.read_kwargs)).op()
        return dt

    def make_unbound_dt(self):
        import dask

        name = f"{self.name}-{dask.base.tokenize(self)}"
        return ops.UnboundTable(
            name=name,
            schema=self.schema,
        )


def register_and_transform_remote_tables(expr):
    import letsql as ls

    created = {}

    op = expr.op()
    graph, _ = Graph.from_bfs(op).toposort()
    counts = defaultdict(int)
    for node in graph:
        if isinstance(node, RemoteTable):
            counts[node] += 1

        if isinstance(node, Relation):
            for arg in node.__args__:
                if isinstance(arg, RemoteTable):
                    counts[arg] += 1

    batches_table = {}
    for arg, count in counts.items():
        ex = arg.remote_expr
        if not ex.op().find((RemoteTable, CachedNode, Read)):
            batches = ex.to_pyarrow_batches()  # execute in native backend
        else:
            batches = ls.to_pyarrow_batches(ex)
        schema = ex.as_table().schema().to_pyarrow()
        replicas = SafeTee.tee(batches, count)
        batches_table[arg] = (schema, list(replicas))

    def mark_remote_table(node):
        schema, batchess = batches_table[node]
        name = f"{node.name}_{len(batchess)}"
        reader = pa.RecordBatchReader.from_batches(schema, batchess.pop())
        result = node.source.read_record_batches(reader, table_name=name)
        created[name] = node.source
        return result.op()

    def replacer(node, _, **kwargs):
        if isinstance(node, Relation):
            updated = {}
            for k, v in list(kwargs.items()):
                try:
                    if v in batches_table:
                        updated[v] = mark_remote_table(v)

                except TypeError:  # v may not be hashable
                    continue

            if len(updated) > 0:
                kwargs = {k: recursive_update(v, updated) for k, v in kwargs.items()}

        node = node.__recreate__(kwargs)
        if isinstance(node, RemoteTable):
            result = mark_remote_table(node)
            batches_table[result] = batches_table.pop(node)
            node = result

        return node

    expr = op.replace(replace_fix(replacer)).to_expr()
    return expr, created
