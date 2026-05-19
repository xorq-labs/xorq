"""Generic Python / stdlib / toolz / numpy / pandas / ibis-Schema normalizers.

These cover types that xorq_dasher 0.1.0's ``DEFAULT_HASHER`` doesn't reach,
but aren't xorq-specific — most could plausibly be upstreamed.  Kept in their
own module so the xorq-specific normalization logic stays focused.
"""

from __future__ import annotations

import functools
import operator
from typing import Any

import numpy as np
import pandas as pd
from xorq_dasher.rules.functions import normalize_function


def _extract_methodcaller_fields(
    mc: operator.methodcaller,
) -> tuple[str, tuple, dict]:
    """Extract (name, args, kwargs) from an operator.methodcaller portably.

    Uses the pickle protocol (__reduce__) instead of ctypes, so this works
    on CPython, PyPy, GraalPy, and any other Python implementation.
    """
    constructor, constructor_args = mc.__reduce__()[:2]
    if isinstance(constructor, functools.partial):
        return constructor.args[0], constructor_args, constructor.keywords
    return constructor_args[0], constructor_args[1:], {}


def normalize_attrs(obj: Any) -> tuple:
    """Stable normalization for any ``attrs.frozen`` object.

    Used by classes that previously aliased ``__dask_tokenize__ = normalize_attrs``.
    """
    return tuple(sorted(obj.__getstate__().items()))


def normalize_lru_cache(func: functools._lru_cache_wrapper) -> tuple:
    inner = func
    while hasattr(inner, "__wrapped__"):
        inner = inner.__wrapped__
    return normalize_function(inner)


def normalize_property(prop: property) -> tuple:
    return ("property", prop.fget, prop.fset, prop.fdel)


def normalize_toolz_compose(composed: Any) -> tuple:
    return ("toolz.Compose", composed.first, composed.funcs)


def normalize_toolz_curry(curried: Any) -> tuple:
    from xorq.common.utils.inspect_utils import get_partial_arguments  # noqa: PLC0415

    partial_arguments = get_partial_arguments(
        curried.func, *curried.args, **curried.keywords
    )
    return ("toolz.curry", curried.func, tuple(sorted(partial_arguments.items())))


def normalize_toolz_excepts(f: Any) -> tuple:
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


def normalize_builtin_callable(func: Any) -> tuple:
    """Builtin C functions / methods (e.g. ``json.dumps``)."""
    return (
        "builtins.builtin",
        getattr(func, "__module__", None),
        getattr(func, "__qualname__", getattr(func, "__name__", repr(func))),
    )


def normalize_slice(s: slice) -> tuple:
    return ("slice", s.start, s.stop, s.step)


def normalize_ibis_schema(schema: Any) -> tuple:
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
    share the same ``pa.Table`` → ``normalize_pyarrow_table`` hashing.
    """
    return ("pandas.Series", series.name, normalize_pandas_dataframe(series.to_frame()))


def normalize_pandas_dataframe(df: pd.DataFrame) -> tuple:
    """Returns the raw ``pa.Table`` so dasher's registered ``pa.Table`` rule
    (``xorq_dasher.rules.other.normalize_pyarrow_table``) does the hashing —
    serializes each batch to bytes and xxhashes.
    """
    import pyarrow as pa  # noqa: PLC0415

    table = pa.Table.from_pandas(df)
    return (
        "pandas.DataFrame",
        tuple(df.columns),
        tuple(str(t) for t in df.dtypes),
        table,
    )


__all__ = [
    "normalize_attrs",
    "normalize_builtin_callable",
    "normalize_functools_partial",
    "normalize_ibis_schema",
    "normalize_lru_cache",
    "normalize_methodcaller",
    "normalize_numpy_dtype",
    "normalize_pandas_dataframe",
    "normalize_pandas_series",
    "normalize_property",
    "normalize_slice",
    "normalize_toolz_compose",
    "normalize_toolz_curry",
    "normalize_toolz_excepts",
]
