from functools import partial, singledispatch

import pyiceberg.expressions as ice

import xorq.common.exceptions as com
import xorq.vendor.ibis.expr.operations as ops
from xorq.backends.pyiceberg.relations import PyIcebergTable
from xorq.vendor.ibis.backends.sql.dialects import Postgres


class PyIceberg(Postgres):
    """Subclass of Postgres dialect for PyIceberg.

    This is here to allow referring to the Postgres dialect as "pyiceberg"
    """


@singledispatch
def translate(expr, **_):
    raise NotImplementedError(expr)


@translate.register(ops.Node)
def operation(op, **_):
    raise NotImplementedError(f"No translation rule for {type(op)}")


@translate.register(ops.Literal)
def literal(op, **_):
    return op.value


@translate.register(ops.Field)
def column(op, **_):
    return op.name


@translate.register(PyIcebergTable)
def iceberg_table(op, catalog=None, namespace=None, **kwargs):
    if catalog is None:
        raise ValueError("catalog cannot be None")
    if namespace is None:
        raise ValueError("namespace cannot be None")

    ice_table = catalog.load_table(f"{namespace}.{op.name}")

    if op.snapshot_id is not None:
        return ice_table.scan(snapshot_id=op.snapshot_id)

    return ice_table.scan()


@translate.register(ops.DatabaseTable)
def table(op, catalog=None, namespace=None, **kwargs):
    if catalog is None:
        raise ValueError("catalog cannot be None")
    if namespace is None:
        raise ValueError("namespace cannot be None")

    ice_table = catalog.load_table(f"{namespace}.{op.name}")

    return ice_table.scan()


@translate.register(ops.Filter)
def filter_(op, **kw):
    scan = translate(op.parent, **kw)
    left, right, *rest = list(map(partial(translate, **kw), op.predicates)) + [
        ice.AlwaysTrue(),
        ice.AlwaysTrue(),
    ]
    predicate = ice.And(left, right, *rest)
    return scan.filter(predicate)


@translate.register(ops.Project)
def project(op, **kw):
    scan = translate(op.parent, **kw)

    selections = []
    for name, arg in op.values.items():
        if isinstance(arg, ops.Value):
            translated = translate(arg, **kw)
            selections.append(translated)
        else:
            raise NotImplementedError(
                f"PyIceberg backend is unable to compile selection with operation type of {type(arg)}"
            )

    return scan.select(*selections)


@translate.register(ops.Limit)
def limit(op, **kw):
    if (n := op.n) is not None and not isinstance(n, int):
        raise NotImplementedError("Dynamic limit not supported")

    if op.offset != 0:
        raise NotImplementedError("Offset not supported")

    scan = translate(op.parent, **kw)

    return scan.update(limit=n)


_comparisons = {
    ops.Equals: ice.EqualTo,
    ops.Greater: ice.GreaterThan,
    ops.GreaterEqual: ice.GreaterThanOrEqual,
    ops.Less: ice.LessThan,
    ops.LessEqual: ice.LessThanOrEqual,
    ops.NotEquals: ice.NotEqualTo,
}


@translate.register(ops.Comparison)
def comparison(op, **kw):
    left = translate(op.left, **kw)
    right = translate(op.right, **kw)
    func = _comparisons.get(type(op))
    if func is None:
        raise com.OperationNotDefinedError(f"{type(op).__name__} not supported")
    return func(left, right)
