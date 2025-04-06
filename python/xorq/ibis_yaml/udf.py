from typing import Any, Callable, Dict

import toolz

import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.rules as rlz
from xorq.expr.relations import FlightExpr, FlightUDXF
from xorq.ibis_yaml.common import (
    TranslationContext,
    _translate_type,
    deserialize_callable,
    register_from_yaml_handler,
    serialize_callable,
    translate_from_yaml,
    translate_to_yaml,
)
from xorq.ibis_yaml.utils import freeze
from xorq.vendor.ibis.common.annotations import Argument


@translate_to_yaml.register(ops.ScalarUDF)
def _scalar_udf_to_yaml(op: ops.ScalarUDF, compiler: Any) -> dict:
    input_type = getattr(op.__class__, "__input_type__", None)

    if input_type not in [ops.udf.InputType.BUILTIN, ops.udf.InputType.PYARROW]:
        raise NotImplementedError(
            f"Translation of UDFs with input type {getattr(op.__class__, '__input_type__', None)} is not supported"
        )
    arg_names = [
        name
        for name in dir(op)
        if not name.startswith("__") and name not in op.__class__.__slots__
    ]

    return freeze(
        {
            "op": "ScalarUDF",
            "func_name": op.__func_name__,
            "input_type": input_type,
            "args": [translate_to_yaml(arg, compiler) for arg in op.args],
            "type": _translate_type(op.dtype),
            "pickle": serialize_callable(op.__func__),
            "module": op.__module__,
            "class_name": op.__class__.__name__,
            "arg_names": arg_names,
        }
    )


@register_from_yaml_handler("ScalarUDF")
def _scalar_udf_from_yaml(yaml_dict: dict, compiler: any) -> any:
    # TODO: use make_pandas_udf (expr/udf.py)
    encoded_fn = yaml_dict.get("pickle")
    if not encoded_fn:
        raise ValueError("Missing pickle data for ScalarUDF")
    fn = deserialize_callable(encoded_fn)

    args = tuple(
        translate_from_yaml(arg, compiler) for arg in yaml_dict.get("args", [])
    )
    if not args:
        raise ValueError("ScalarUDF requires at least one argument")

    arg_names = yaml_dict.get("arg_names", [f"arg{i}" for i in range(len(args))])
    input_type = yaml_dict.get("input_type")

    if input_type == "BUILTIN":
        input_type = ops.udf.InputType.BUILTIN
    elif input_type == "PYARROW":
        input_type = ops.udf.InputType.PYARROW

    fields = {
        name: Argument(pattern=rlz.ValueOf(arg.type()), typehint=arg.type())
        for name, arg in zip(arg_names, args)
    }

    bases = (ops.ScalarUDF,)
    meta = {
        "dtype": dt.dtype(yaml_dict["type"]["name"]),
        "__input_type__": input_type,
        "__func__": property(fget=lambda _, f=fn: f),
        "__config__": {"volatility": "immutable"},
        "__udf_namespace__": None,
        "__module__": yaml_dict.get("module", "__main__"),
        "__func_name__": yaml_dict["func_name"],
    }

    kwds = {**fields, **meta}
    class_name = yaml_dict.get("class_name", yaml_dict["func_name"])

    node = type(
        class_name,
        bases,
        kwds,
    )

    return node(*args).to_expr()


@translate_to_yaml.register(bool)
def _bool_to_yaml(value: bool, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "bool",
            "value": value,
        }
    )


@register_from_yaml_handler("bool")
def _bool_from_yaml(yaml_dict: dict, context: TranslationContext) -> bool:
    return yaml_dict["value"]


@translate_to_yaml.register(dt.DataType)
def _datatype_to_yaml(dtype: dt.DataType, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "DataType",
            "type": type(dtype).__name__,
        }
        | {
            argname: translate_to_yaml(arg, context)
            for argname, arg in zip(dtype.argnames, dtype.args)
        }
    )


@register_from_yaml_handler("DataType")
def _datatype_from_yaml(yaml_dict: dict, context: TranslationContext) -> any:
    typ = getattr(dt, yaml_dict["type"])
    dct = toolz.dissoc(yaml_dict, "op", "type")
    return typ(
        **{key: translate_from_yaml(value, context) for key, value in dct.items()}
    )


@translate_to_yaml.register(dict)
def _dict_to_yaml(dct: dict, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "dict",
        }
        | {key: translate_to_yaml(value, context) for key, value in dct.items()}
    )


@register_from_yaml_handler("dict")
def _dict_from_yaml(yaml_dict: dict, context: TranslationContext) -> any:
    dct = {
        key: translate_from_yaml(value, context)
        for key, value in toolz.dissoc(yaml_dict, "op").items()
    }
    return dct


@translate_to_yaml.register(ops.Namespace)
def _namespace_to_yaml(ns: ops.Namespace, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "Namespace",
            "dict": {argname: arg for argname, arg in zip(ns.argnames, ns.args)},
        }
    )


@register_from_yaml_handler("Namespace")
def _namespace_from_yaml(yaml_dict: dict, compiler: any) -> any:
    return ops.Namespace(**yaml_dict["dict"])


@translate_to_yaml.register(ops.udf.InputType)
def _inputtype_to_yaml(it: ops.udf.InputType, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "InputType",
            "name": it.name,
        }
    )


@register_from_yaml_handler("InputType")
def _inputtype_from_yaml(yaml_dict: dict, context: TranslationContext) -> any:
    return getattr(ops.udf.InputType, yaml_dict["name"])


def require_input_types(input_types, op):
    if (input_type := getattr(op.__class__, "__input_type__", None)) not in input_types:
        raise NotImplementedError(
            f"Translation of UDFs with input type {input_type} is not supported"
            f"\n\tsupported input types: {input_types}"
        )


def make_op_kwargs(op):
    (*argnames, where) = op.argnames
    if where != "where":
        raise ValueError
    kwargs = {argname: arg for (argname, arg) in zip(argnames, op.args)}
    return kwargs


def make_dunder_func(fn, kwargs):
    import pandas as pd

    import xorq as xo

    schema = xo.schema(
        {
            argname: typ
            for (argname, typ) in (
                (argname, arg.type()) for argname, arg in kwargs.items()
            )
        }
    )

    def fn_from_arrays(*arrays):
        df = pd.DataFrame(
            {name: array.to_pandas() for (name, array) in zip(schema, arrays)}
        )
        return fn(df)

    __func__ = property(fget=lambda _, fn_from_arrays=fn_from_arrays: fn_from_arrays)
    return __func__


@translate_to_yaml.register(object)
def _object_to_yaml(obj: object, compiler: Any) -> dict:
    if isinstance(obj, Callable):
        return freeze(
            {
                "op": "Callabe",
                "pickled_fn": serialize_callable(obj),
            }
        )
    else:
        raise ValueError


@register_from_yaml_handler("Callabe")
def _callable_from_yaml(yaml_dict: dict, compiler: any) -> Callable:
    return deserialize_callable(yaml_dict["pickled_fn"])


@translate_to_yaml.register(ops.udf.AggUDF)
def _aggudf_to_yaml(op: ops.udf.AggUDF, compiler: Any) -> dict:
    require_input_types((ops.udf.InputType.PYARROW,), op)
    kwargs = make_op_kwargs(op)
    meta = {
        name: getattr(op, name)
        for name in (
            "dtype",
            "__config__",
            "__input_type__",
            "__udf_namespace__",
            "__module__",
            "__func_name__",
        )
    }
    return freeze(
        {
            "op": "AggUDF",
            "class_name": op.__class__.__name__,
            "kwargs": translate_to_yaml(freeze(kwargs), compiler),
            "meta": translate_to_yaml(freeze(meta), compiler),
        }
    )


@register_from_yaml_handler("AggUDF")
def _aggudf_from_yaml(yaml_dict: dict, compiler: any) -> any:
    (kwargs, meta) = (
        translate_from_yaml(yaml_dict[name], compiler) for name in ("kwargs", "meta")
    )
    __func__ = make_dunder_func(meta["__config__"]["fn"], kwargs)
    fields = {
        argname: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
        for (argname, typ) in ((argname, arg.type()) for argname, arg in kwargs.items())
    }
    #
    class_name = yaml_dict["class_name"]
    bases = (ops.udf.AggUDF,)
    kwds = fields | meta | {"__func__": __func__}
    node = type(
        class_name,
        bases,
        kwds,
    )
    return node(**kwargs).to_expr()


@translate_to_yaml.register(FlightExpr)
def flight_expr_to_yaml(op: FlightExpr, context: any) -> dict:
    input_expr_yaml = translate_to_yaml(op.input_expr, context)
    unbound_expr_yaml = translate_to_yaml(op.unbound_expr, context)

    schema_id = context.schema_registry.register_schema(op.schema)

    make_server_pickle = serialize_callable(op.make_server)
    make_connection_pickle = serialize_callable(op.make_connection)

    return freeze(
        {
            "op": "FlightExpr",
            "name": op.name,
            "schema_ref": schema_id,
            "input_expr": input_expr_yaml,
            "unbound_expr": unbound_expr_yaml,
            "make_server": make_server_pickle,
            "make_connection": make_connection_pickle,
            "do_instrument_reader": op.do_instrument_reader,
        }
    )


@register_from_yaml_handler("FlightExpr")
def flight_expr_from_yaml(yaml_dict: Dict, context: Any) -> Any:
    name = yaml_dict.get("name")
    input_expr_yaml = yaml_dict.get("input_expr")
    unbound_expr_yaml = yaml_dict.get("unbound_expr")
    make_server_pickle = yaml_dict.get("make_server")
    make_connection_pickle = yaml_dict.get("make_connection")
    do_instrument_reader = yaml_dict.get("do_instrument_reader", False)

    input_expr = translate_from_yaml(input_expr_yaml, context)
    unbound_expr = translate_from_yaml(unbound_expr_yaml, context)

    make_server = (
        deserialize_callable(make_server_pickle) if make_server_pickle else None
    )
    make_connection = (
        deserialize_callable(make_connection_pickle) if make_connection_pickle else None
    )

    return FlightExpr.from_exprs(
        input_expr=input_expr,
        unbound_expr=unbound_expr,
        make_server=make_server,
        make_connection=make_connection,
        name=name,
        do_instrument_reader=do_instrument_reader,
    ).to_expr()


@translate_to_yaml.register(FlightUDXF)
def flight_udxf_to_yaml(op: FlightUDXF, context: any) -> dict:
    input_expr_yaml = translate_to_yaml(op.input_expr, context)
    schema_id = context.schema_registry.register_schema(op.schema)
    udxf_pickle = serialize_callable(op.udxf)
    make_server_pickle = serialize_callable(op.make_server)
    make_connection_pickle = serialize_callable(op.make_connection)

    return freeze(
        {
            "op": "FlightUDXF",
            "name": op.name,
            "schema_ref": schema_id,
            "input_expr": input_expr_yaml,
            "udxf": udxf_pickle,
            "make_server": make_server_pickle,
            "make_connection": make_connection_pickle,
            "do_instrument_reader": op.do_instrument_reader,
        }
    )


@register_from_yaml_handler("FlightUDXF")
def flight_udxf_from_yaml(yaml_dict: Dict, context: Any) -> Any:
    name = yaml_dict.get("name")
    input_expr_yaml = yaml_dict.get("input_expr")
    udxf_pickle = yaml_dict.get("udxf")
    make_server_pickle = yaml_dict.get("make_server")
    make_connection_pickle = yaml_dict.get("make_connection")
    do_instrument_reader = yaml_dict.get("do_instrument_reader", False)

    input_expr = translate_from_yaml(input_expr_yaml, context)
    udxf = deserialize_callable(udxf_pickle)
    make_server = (
        deserialize_callable(make_server_pickle) if make_server_pickle else None
    )
    make_connection = (
        deserialize_callable(make_connection_pickle) if make_connection_pickle else None
    )

    return FlightUDXF.from_expr(
        input_expr=input_expr,
        udxf=udxf,
        make_server=make_server,
        make_connection=make_connection,
        name=name,
        do_instrument_reader=do_instrument_reader,
    ).to_expr()
