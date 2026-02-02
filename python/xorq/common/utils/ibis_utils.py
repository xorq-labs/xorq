import datetime
import functools
import importlib
from contextlib import contextmanager

import toolz
from ibis import Schema as IbisSchema
from ibis.backends import BaseBackend as IbisBaseBackend
from ibis.common.collections import FrozenOrderedDict as IbisFrozenOrderedDict
from ibis.common.temporal import IntervalUnit as IbisIntervalUnit
from ibis.common.temporal import TimestampUnit as IbisTimestampUnit
from ibis.common.temporal import TimeUnit as IbisTimeUnit
from ibis.expr.datatypes import DataType as IbisDataType
from ibis.expr.datatypes.core import Interval as IbisInterval
from ibis.expr.operations.generic import Cast as IbisCast
from ibis.expr.operations.relations import Namespace as IbisNamespace
from ibis.formats.pandas import PandasDataFrameProxy as IbisPandasDataFrameProxy
from ibis.formats.pyarrow import PyArrowTableProxy as IbisPyArrowTableProxy

import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis import Schema
from xorq.vendor.ibis.backends.profiles import Profile
from xorq.vendor.ibis.common.collections import FrozenOrderedDict
from xorq.vendor.ibis.common.temporal import IntervalUnit, TimestampUnit, TimeUnit
from xorq.vendor.ibis.expr.datatypes import DataType
from xorq.vendor.ibis.expr.datatypes.core import Interval
from xorq.vendor.ibis.expr.operations import Node
from xorq.vendor.ibis.expr.operations.relations import Namespace
from xorq.vendor.ibis.formats.pandas import PandasDataFrameProxy
from xorq.vendor.ibis.formats.pyarrow import PyArrowTableProxy


@functools.singledispatch
def map_ibis(val, kwargs=None):
    try:
        attr = val.__class__.__name__
        module = val.__class__.__module__

        cls = getattr(importlib.import_module(f"xorq.vendor.{module}"), attr)

        kwargs = kwargs if kwargs else dict(zip(val.argnames, val.args))
        kwargs = toolz.valmap(
            map_ibis,
            kwargs if kwargs else dict(zip(val.argnames, val.args)),
        )

        return cls(**kwargs)

    except AttributeError:
        raise NotImplementedError(f"{type(val)} is not implemented")


@map_ibis.register(int)
@map_ibis.register(str)
@map_ibis.register(float)
@map_ibis.register(Node)
@map_ibis.register(type(None))
@map_ibis.register(datetime.datetime)
def map_pass_through(op, kwargs=None):
    return op


@map_ibis.register(tuple)
@map_ibis.register(list)
@map_ibis.register(set)
@map_ibis.register(frozenset)
def map_builtin_container(op, kwargs=None):
    return type(op)(map_ibis(val, None) for val in op)


@map_ibis.register(dict)
def map_builtin_dict(op, kwargs=None):
    return {map_ibis(k, None): map_ibis(v, None) for k, v in op.items()}


@map_ibis.register(IbisCast)
def map_cast(cast, kwargs=None):
    return ops.Cast(arg=map_ibis(cast.arg, None), to=map_ibis(cast.to, None))


@map_ibis.register(IbisSchema)
def map_schema(schema, kwargs=None):
    return Schema(
        dict(zip(schema.names, tuple(map_ibis(typ, kwargs) for typ in schema.types)))
    )


@map_ibis.register(IbisIntervalUnit)
def map_interval_unit(unit, kwargs=None):
    return IntervalUnit(unit.value)


@map_ibis.register(IbisTimestampUnit)
def map_timestamp_unit(unit: IbisTimestampUnit, kwargs=None):
    return TimestampUnit(unit.value)


@map_ibis.register(IbisTimeUnit)
def map_time_unit(unit, kwargs=None):
    return TimeUnit(unit.value)


@map_ibis.register(IbisInterval)
def map_interval(interval, kwargs=None):
    return Interval(
        unit=map_ibis(interval.unit, None), nullable=map_ibis(interval.nullable, None)
    )


@map_ibis.register(IbisDataType)
def map_datatype(datatype, kwargs=None):
    return DataType.from_pyarrow(datatype.to_pyarrow())


@map_ibis.register(IbisPandasDataFrameProxy)
def map_pandas_dataframe_proxy(proxy, kwargs=None):
    return PandasDataFrameProxy(proxy.obj)


@map_ibis.register(IbisPyArrowTableProxy)
def map_pyarrow_table_proxy(proxy, kwargs=None):
    return PyArrowTableProxy(proxy.obj)


@map_ibis.register(IbisFrozenOrderedDict)
def map_frozendict(frozendict, kwargs=None):
    return FrozenOrderedDict(
        tuple(
            (map_ibis(key, kwargs), map_ibis(value, kwargs))
            for key, value in frozendict.items()
        )
    )


backend_registry = {}


@contextmanager
def backend_registry_context():
    assert not backend_registry
    yield
    backend_registry.clear()


@map_ibis.register(IbisBaseBackend)
def map_backend(backend, kwargs=None):
    backend_id = id(backend)
    if new_backend := backend_registry.get(backend_id):
        pass
    else:
        new_backend = Profile.from_con(backend).get_con()
        new_backend.con = backend.con
        backend_registry[backend_id] = new_backend
    return new_backend


@map_ibis.register(IbisNamespace)
def map_namespace(namespace, kwargs=None):
    return Namespace(catalog=namespace.catalog, database=namespace.database)


def from_ibis(ibis_expr):
    with backend_registry_context():
        return ibis_expr.op().replace(map_ibis).to_expr()
