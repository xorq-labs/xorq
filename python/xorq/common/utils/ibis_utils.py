import datetime
import functools
import importlib

from ibis import Schema as IbisSchema
from ibis.backends import BaseBackend as IbisBaseBackend
from ibis.common.collections import FrozenOrderedDict as IbisFrozenOrderedDict
from ibis.common.temporal import IntervalUnit as IbisIntervalUnit
from ibis.expr.datatypes import DataType as IbisDataType
from ibis.expr.datatypes.core import Interval as IbisInterval
from ibis.expr.operations.generic import Cast as IbisCast
from ibis.expr.operations.relations import Namespace as IbisNamespace
from ibis.formats.pandas import PandasDataFrameProxy as IbisPandasDataFrameProxy

import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis import Schema
from xorq.vendor.ibis.backends import Profile
from xorq.vendor.ibis.common.collections import FrozenOrderedDict
from xorq.vendor.ibis.common.temporal import IntervalUnit
from xorq.vendor.ibis.expr.datatypes import DataType
from xorq.vendor.ibis.expr.datatypes.core import Interval
from xorq.vendor.ibis.expr.operations import Node
from xorq.vendor.ibis.expr.operations.relations import Namespace
from xorq.vendor.ibis.formats.pandas import PandasDataFrameProxy


@functools.singledispatch
def map_ibis(val, kwargs):
    try:
        attr = val.__class__.__name__
        module = val.__class__.__module__

        cls = getattr(importlib.import_module(f"xorq.vendor.{module}"), attr)

        _kwargs = (
            kwargs
            if kwargs
            else dict(zip(val.argnames, tuple(map_ibis(arg, None) for arg in val.args)))
        )

        return cls(**_kwargs)

    except AttributeError:
        raise NotImplementedError(f"{type(val)} is not implemented")


@map_ibis.register(int)
@map_ibis.register(str)
@map_ibis.register(float)
@map_ibis.register(Node)
@map_ibis.register(type(None))
@map_ibis.register(datetime.datetime)
def map_pass_through(op, kwargs):
    return op


@map_ibis.register(IbisCast)
def map_cast(cast, kwargs):
    return ops.Cast(arg=map_ibis(cast.arg, None), to=map_ibis(cast.to, None))


@map_ibis.register(IbisSchema)
def map_schema(schema, kwargs):
    return Schema(
        dict(zip(schema.names, tuple(map_ibis(typ, kwargs) for typ in schema.types)))
    )


@map_ibis.register(IbisIntervalUnit)
def map_interval_unit(unit, kwargs):
    return IntervalUnit(unit.value)


@map_ibis.register(IbisInterval)
def map_interval(interval, kwargs):
    return Interval(
        unit=map_ibis(interval.unit, None), nullable=map_ibis(interval.nullable, None)
    )


@map_ibis.register(IbisDataType)
def map_datatype(datatype, kwargs):
    return DataType.from_pyarrow(datatype.to_pyarrow())


@map_ibis.register(IbisPandasDataFrameProxy)
def map_pandas_dataframe_proxy(proxy, kwargs):
    return PandasDataFrameProxy(proxy.obj)


@map_ibis.register(IbisFrozenOrderedDict)
def map_frozendict(frozendict, kwargs):
    return FrozenOrderedDict(
        tuple(
            (map_ibis(key, kwargs), map_ibis(value, kwargs))
            for key, value in frozendict.items()
        )
    )


@map_ibis.register(IbisBaseBackend)
def map_backend(backend, kwargs):
    new_backend = Profile.from_con(backend).get_con()
    new_backend.con = backend.con
    return new_backend


@map_ibis.register(IbisNamespace)
def map_namespace(namespace, kwargs):
    return Namespace(catalog=namespace.catalog, database=namespace.database)


def from_ibis(ibis_expr):
    return ibis_expr.op().replace(map_ibis).to_expr()
