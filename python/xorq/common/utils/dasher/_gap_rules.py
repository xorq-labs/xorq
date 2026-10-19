"""Generic Python / stdlib / toolz / numpy / pandas / ibis-Schema normalizers.

These cover types that xorq_dasher 0.1.0's ``DEFAULT_HASHER`` doesn't reach,
but aren't xorq-specific — most could plausibly be upstreamed.  Kept in their
own module so the xorq-specific normalization logic stays focused.
"""

from __future__ import annotations

import contextvars
import dis
import functools
import operator
import sys
import types
from typing import TYPE_CHECKING

import toolz
from xorq_dasher.rules.functions import normalize_function

from xorq.common.utils.inspect_utils import get_partial_arguments


if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from sklearn.utils._param_validation import (
        Hidden as _SklearnHidden,  # noqa: PLC2701
    )
    from sklearn.utils._param_validation import (
        _Constraint as _SklearnConstraint,  # noqa: PLC2701
    )

    from xorq.vendor.ibis.expr.schema import Schema


def _extract_methodcaller_fields(
    mc: operator.methodcaller,
) -> tuple[str, tuple, dict]:
    """Uses the pickle protocol (__reduce__) instead of ctypes, so this works
    on CPython, PyPy, GraalPy, and any other Python implementation.
    """
    constructor, constructor_args = mc.__reduce__()[:2]
    if isinstance(constructor, functools.partial):
        return constructor.args[0], constructor_args, constructor.keywords
    return constructor_args[0], constructor_args[1:], {}


def normalize_attrs(obj: object) -> tuple:
    """Stable normalization for any ``attrs.frozen`` object.

    Used by classes that previously aliased ``__dask_tokenize__ = normalize_attrs``.
    """
    if not hasattr(type(obj), "__attrs_attrs__"):
        raise TypeError(
            f"normalize_attrs expected an attrs class, got {type(obj).__name__!r}"
        )
    return tuple(sorted(obj.__getstate__().items()))


# Functions whose globals are being tokenized right now, by id. A function
# reachable from its own globals (through a ``partial`` or ``curry`` global,
# say) meets this guard and folds by name instead of recursing until the
# stack runs out.
_functions_in_progress: contextvars.ContextVar[frozenset[int]] = contextvars.ContextVar(
    "_xorq_functions_in_progress", default=frozenset()
)


def _referenced_names(code: types.CodeType) -> frozenset[str]:
    """Every global a code object loads, nested code objects (lambdas,
    comprehensions, inner defs) included. Read off the ``LOAD_GLOBAL``
    instructions rather than ``co_names``, which also lists attribute names
    (``pc.add`` would drag in an unrelated global called ``add``), and not
    ``LOAD_NAME``, which a nested class body resolves locally first."""
    loads = frozenset(
        instruction.argval
        for instruction in dis.get_instructions(code)
        if instruction.opname == "LOAD_GLOBAL"
    )
    nested = (const for const in code.co_consts if isinstance(const, types.CodeType))
    return loads.union(*map(_referenced_names, nested))


def _is_importable(func: types.FunctionType) -> bool:
    """True when ``module.qualname`` resolves back to ``func``: a library
    function its name can stand for. A closure, a lambda or an exec'd helper
    does not resolve, and carries state its name says nothing about."""
    module = sys.modules.get(func.__module__ or "")
    try:
        return functools.reduce(getattr, func.__qualname__.split("."), module) is func
    except AttributeError:
        return False


def _global_token(hasher, value) -> tuple | str:
    """A referenced global's contribution to its function's token.

    An importable function folds by reference (module and qualname), the
    way dasher already folds a class or a module: by-value recursion into a
    library function's own globals reaches thread-locals and sentinels that
    have no normalizer (``sklearn.metrics.*`` dies on ``_threadlocal``).
    Everything else, a closure-made helper included, goes through the active
    hasher like a closure cell or a default would, so a value without a
    normalizer raises the same way.
    """
    match value:
        case types.FunctionType() if _is_importable(value):
            return ("function", value.__module__, value.__qualname__)
        case _:
            return hasher.tokenize(value)


def normalize_function_with_globals(func: types.FunctionType) -> tuple:
    """``normalize_function`` plus the globals the function references.

    ``co_names`` crossed with ``__globals__`` is the half of a function's
    captured environment that ``__closure__`` does not carry; without it two
    functions with identical bytecode and different global bindings collide
    (#2147). The globals are tokenized eagerly, inside the in-progress guard,
    rather than returned for the hasher's lazy recursion: that is what lets a
    function reachable from its own globals through a ``partial`` or a
    ``curry`` terminate.
    """
    from xorq.common.utils.dasher import (  # noqa: PLC0415  # xorq-style: disable=unlisted-import
        HASHER,
        _current_hasher,
    )

    in_progress = _functions_in_progress.get()
    if id(func) in in_progress:
        return ("function", func.__module__, func.__qualname__)
    hasher = _current_hasher.get() or HASHER
    names = sorted(_referenced_names(func.__code__) & func.__globals__.keys())
    token = _functions_in_progress.set(in_progress | {id(func)})
    try:
        globals_ = tuple(
            (name, _global_token(hasher, func.__globals__[name])) for name in names
        )
    finally:
        _functions_in_progress.reset(token)
    return (*normalize_function(func), globals_)


def normalize_lru_cache(func: functools._lru_cache_wrapper) -> tuple:
    inner = func
    while hasattr(inner, "__wrapped__"):
        inner = inner.__wrapped__
    return normalize_function(inner)


def normalize_property(prop: property) -> tuple:
    return ("property", prop.fget, prop.fset, prop.fdel)


def normalize_toolz_compose(composed: toolz.functoolz.Compose) -> tuple:
    return ("toolz.Compose", composed.first, composed.funcs)


def normalize_toolz_curry(curried: toolz.curry) -> tuple:
    partial_arguments = get_partial_arguments(
        curried.func, *curried.args, **curried.keywords
    )
    return ("toolz.curry", curried.func, tuple(sorted(partial_arguments.items())))


def normalize_toolz_excepts(f: toolz.functoolz.excepts) -> tuple:
    return ("toolz.excepts", f.exc, f.func)


def normalize_methodcaller(obj: operator.methodcaller) -> tuple:
    return ("operator.methodcaller", *_extract_methodcaller_fields(obj))


def normalize_functools_partial(p: functools.partial) -> tuple:
    """``functools.partial`` is callable; capture func + args + sorted kwargs."""
    return (
        "functools.partial",
        p.func,
        tuple(p.args),
        tuple(sorted(p.keywords.items())),
    )


def normalize_builtin_callable(
    func: types.BuiltinFunctionType | types.BuiltinMethodType,
) -> tuple:
    """Builtin C functions / methods (e.g. ``json.dumps``)."""
    return (
        "builtins.builtin",
        getattr(func, "__module__", None),
        getattr(func, "__qualname__", getattr(func, "__name__", repr(func))),
    )


def normalize_slice(s: slice) -> tuple:
    return ("slice", s.start, s.stop, s.step)


def normalize_ibis_schema(schema: Schema) -> tuple:
    """Schema normalizer that preserves ibis type identity.

    xorq_dasher 0.1.0's rule uses ``schema.to_pandas()`` which collapses
    decimal/array/struct/map to ``dtype('O')`` — two semantically distinct
    schemas with the same column names but different "complex" ibis dtypes
    would collide. Round-tripping through ``str(dtype)`` preserves full ibis
    type info (precision, parameterization, etc.).
    """
    return ("ibis.Schema", tuple((name, str(dtype)) for name, dtype in schema.items()))


def normalize_numpy_dtype(dtype: np.dtype) -> tuple:
    return ("numpy.dtype", str(dtype), dtype.kind, dtype.itemsize)


def normalize_pandas_series(series: pd.Series) -> tuple:
    """Promotes to a single-column DataFrame and delegates so both paths
    share the same ``pa.Table`` canonical hashing.
    """
    return ("pandas.Series", series.name, normalize_pandas_dataframe(series.to_frame()))


def normalize_pandas_dataframe(df: pd.DataFrame) -> tuple:
    """Returns the raw ``pa.Table`` so the registered ``pa.Table`` rule
    (``_canonical.normalize_pyarrow_table_canonical``) does the hashing —
    per-column, layout- and pyarrow-version-independent.
    """
    import pyarrow as pa  # noqa: PLC0415

    table = pa.Table.from_pandas(df)
    return (
        "pandas.DataFrame",
        tuple(df.columns),
        tuple(str(t) for t in df.dtypes),
        table,
    )


def normalize_sklearn_constraint(obj: _SklearnConstraint) -> tuple:
    """Normalize sklearn parameter-validation constraint objects.

    These live in function ``__dict__["_skl_parameter_constraints"]`` and
    are reached when ``normalize_function`` recurses into ``__dict__``.
    """
    cls = type(obj)
    fqn = f"{cls.__module__}.{cls.__qualname__}"
    items = tuple(
        (
            k,
            tuple(sorted(v)) if isinstance(v, (set, frozenset)) else v,
        )
        for k, v in sorted(obj.__dict__.items())
    )
    return ("sklearn._Constraint", fqn, items)


def normalize_sklearn_hidden(obj: _SklearnHidden) -> tuple:
    return ("sklearn.Hidden", obj.constraint)


__all__ = [
    "normalize_attrs",
    "normalize_builtin_callable",
    "normalize_function_with_globals",
    "normalize_functools_partial",
    "normalize_ibis_schema",
    "normalize_lru_cache",
    "normalize_methodcaller",
    "normalize_numpy_dtype",
    "normalize_pandas_dataframe",
    "normalize_pandas_series",
    "normalize_property",
    "normalize_sklearn_constraint",
    "normalize_sklearn_hidden",
    "normalize_slice",
    "normalize_toolz_compose",
    "normalize_toolz_curry",
    "normalize_toolz_excepts",
]
