"""Generic Python / stdlib / toolz / numpy / pandas / ibis-Schema normalizers.

These cover types that xorq_dasher 0.1.0's ``DEFAULT_HASHER`` doesn't reach,
but aren't xorq-specific — most could plausibly be upstreamed.  Kept in their
own module so the xorq-specific normalization logic stays focused.
"""

from __future__ import annotations

from ctypes import POINTER, Structure, c_size_t, c_void_p, cast, py_object

from xorq_dasher.rules.functions import normalize_function


_PYOBJECT_HEAD = [("ob_refcnt", c_size_t), ("ob_type", c_void_p)]


def _ctypes_field(fields, field, obj):
    cls = type(
        "ctypes-hack",
        (Structure,),
        {"_fields_": _PYOBJECT_HEAD + [(f, c_void_p) for f in fields]},
    )
    inst = cast(c_void_p(id(obj)), POINTER(cls)).contents
    return cast(getattr(inst, field), py_object).value


def normalize_attrs(obj):
    """Stable normalization for any ``attrs.frozen`` object.

    Used by classes that previously aliased ``__dask_tokenize__ = normalize_attrs``.
    Raises ``TypeError`` if ``obj`` isn't an attrs class, so callers get a
    clear diagnostic instead of an ``AttributeError: __getstate__`` from the
    fallback.
    """
    if not hasattr(type(obj), "__attrs_attrs__"):
        raise TypeError(
            f"normalize_attrs expected an attrs class, got {type(obj).__name__!r}"
        )
    return tuple(sorted(obj.__getstate__().items()))


def normalize_lru_cache(func):
    inner = func
    while hasattr(inner, "__wrapped__"):
        inner = inner.__wrapped__
    return normalize_function(inner)


def normalize_property(prop):
    return ("property", prop.fget, prop.fset, prop.fdel)


def normalize_toolz_compose(composed):
    return ("toolz.Compose", composed.first, composed.funcs)


def normalize_toolz_curry(curried):
    from xorq.common.utils.inspect_utils import get_partial_arguments  # noqa: PLC0415

    partial_arguments = get_partial_arguments(
        curried.func, *curried.args, **curried.keywords
    )
    return ("toolz.curry", curried.func, tuple(sorted(partial_arguments.items())))


def normalize_toolz_excepts(f):
    return ("toolz.excepts", f.exc, f.func)


def normalize_methodcaller(obj):
    fields = ("name", "args", "kwargs")
    return ("operator.methodcaller", *(_ctypes_field(fields, f, obj) for f in fields))


def normalize_functools_partial(p):
    """``functools.partial`` is callable; capture func + args + sorted kwargs."""
    return (
        "functools.partial",
        p.func,
        tuple(p.args),
        tuple(sorted(p.keywords.items())),
    )


def normalize_builtin_callable(func):
    """Builtin C functions / methods (e.g. ``json.dumps``)."""
    return (
        "builtins.builtin",
        getattr(func, "__module__", None),
        getattr(func, "__qualname__", getattr(func, "__name__", repr(func))),
    )


def normalize_slice(s):
    return ("slice", s.start, s.stop, s.step)


def normalize_ibis_schema(schema):
    """Schema normalizer that preserves ibis type identity.

    xorq_dasher 0.1.0's rule uses ``schema.to_pandas()`` which collapses
    decimal/array/struct/map to ``dtype('O')`` — two semantically distinct
    schemas with the same column names but different "complex" ibis dtypes
    would collide. Round-tripping through ``str(dtype)`` preserves full ibis
    type info (precision, parameterization, etc.).
    """
    return ("ibis.Schema", tuple((name, str(dtype)) for name, dtype in schema.items()))


def normalize_numpy_dtype(dtype):
    return ("numpy.dtype", str(dtype), dtype.kind, dtype.itemsize)


def normalize_pandas_series(series):
    """Series elements go through ``to_pylist()`` because dasher has no
    ``pa.Array`` rule to delegate to.  ``normalize_pandas_dataframe`` below
    uses the faster ``pa.Table`` path because dasher *does* register a
    ``pa.Table`` rule (serialize each batch to bytes, xxhash) — the two
    helpers look inconsistent for that reason, not because either is wrong.
    """
    import pyarrow as pa  # noqa: PLC0415

    return (
        "pandas.Series",
        series.name,
        str(series.dtype),
        pa.Array.from_pandas(series).to_pylist(),
    )


def normalize_pandas_dataframe(df):
    """Returns the raw ``pa.Table`` so dasher's registered ``pa.Table`` rule
    (``xorq_dasher.rules.other.normalize_pyarrow_table``) does the hashing —
    serializes each batch to bytes and xxhashes, identical to the legacy
    dask-era path.
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
