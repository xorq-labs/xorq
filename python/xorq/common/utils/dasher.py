"""Project-wide deterministic hashing for xorq, built on xorq_dasher.

This module exposes the canonical ``HASHER`` instance and the ``tokenize`` /
``normalize`` helpers that the rest of xorq uses for cache keys, build
hashes, deterministic names, and lineage.

``DEFAULT_HASHER`` from xorq_dasher already covers ibis/xorq expression
types, Python callables (FunctionType/MethodType/CodeType/CellType/classmethod
/staticmethod), and the common builtins/numpy/pandas/pyarrow/sklearn rules.
This module adds the few gap rules that exist in xorq's legacy
dask-normalize code but are not yet in xorq_dasher 0.1.0:
``functools._lru_cache_wrapper``, ``property``, ``toolz.functoolz.Compose``,
``toolz.curry`` (both stock and the xorq ``toolz_utils`` variant),
``toolz.functoolz.excepts``, and ``operator.methodcaller``.
"""

from __future__ import annotations

import functools
import operator
from ctypes import POINTER, Structure, c_size_t, c_void_p, cast, py_object

import toolz
from xorq_dasher import DEFAULT_HASHER, Hasher, fqn
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
    """
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
    import pyarrow as pa  # noqa: PLC0415

    return (
        "pandas.Series",
        series.name,
        str(series.dtype),
        pa.Array.from_pandas(series).to_pylist(),
    )


def normalize_pandas_dataframe(df):
    import pyarrow as pa  # noqa: PLC0415

    table = pa.Table.from_pandas(df)
    return (
        "pandas.DataFrame",
        tuple(df.columns),
        tuple(str(t) for t in df.dtypes),
        table,
    )


def _normalize_read_xorq(read):
    """xorq-flavored Read normalizer.

    xorq stores the read path under the canonical ``hash_path`` key (defer_utils
    renames backend-specific kwargs), so the dasher 0.1.0 Read rule (which looks
    up ``path/paths/source/source_list``) does not match. This restores the
    legacy xorq behavior covering http(s), cloud, build-bundle relative, and
    local-filesystem paths.
    """
    import pathlib  # noqa: PLC0415

    read_kwargs = dict(read.read_kwargs)
    path = read_kwargs["hash_path"]
    if isinstance(path, (list, tuple)):
        path = path[0] if len(path) == 1 else path
    if isinstance(path, (str, pathlib.Path)):
        path = str(path)
        if path.startswith(("http://", "https://", "s3://", "gs://", "gcs://")):
            # Remote paths: defer to the legacy stat helper if available.
            from xorq.expr import api  # noqa: PLC0415

            if path.startswith(("http://", "https://")):
                import urllib.request  # noqa: PLC0415

                req = urllib.request.Request(
                    path, method="HEAD", headers={"User-Agent": "xorq-cache"}
                )
                resp = urllib.request.urlopen(req, timeout=10)
                headers = resp.info()
                tpls = (
                    ("url", path),
                    *(
                        (k, headers.get(k))
                        for k in ("Last-Modified", "Content-Length", "Content-Type")
                    ),
                )
            else:
                meta = api.get_object_metadata(
                    path, **{k: v for k, v in read_kwargs.items() if k != "hash_path"}
                )
                tpls = tuple(
                    (k, meta.get(k))
                    for k in ("location", "last_modified", "size", "e_tag", "version")
                )
        elif not pathlib.Path(path).is_absolute() and path == read_kwargs.get(
            "read_path"
        ):
            # Build-bundled Read: relative read_path is already a content hash.
            tpls = (("build-relative-path", path),)
        elif (p := pathlib.Path(path)).exists():
            tpls = read.normalize_method(p)
        else:
            raise NotImplementedError(f'Don\'t know how to deal with path "{path}"')
    else:
        raise NotImplementedError(f'Don\'t know how to deal with path "{path}"')
    tpls += tuple(
        (k, v) for k, v in read.read_kwargs if k in ("mode", "schema", "temporary")
    )
    return ("xorq.Read", read.schema, tpls)


def _stable_opaque_name(prefix, *parts):
    """Build a deterministic placeholder name from xxhash of structural parts.

    xorq_dasher 0.1.0's ``_opaque_to_placeholder`` uses ``id(node)`` for some
    leaf names, which breaks across catalog reloads (different Python object
    identities for semantically-identical Reads). This helper keys on a
    content-stable hash of the supplied parts instead.
    """
    import xxhash  # noqa: PLC0415

    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return f"{prefix}-{xxhash.xxh128(payload).hexdigest()[:16]}"


def _parent_token(thing):
    """Tokenize an opaque sub-expression's parent / inner expr structurally.

    Used to fold the inner expression's identity into the placeholder name so
    two opaque wrappers with the same schema/cache-type/etc. but different
    inner expressions do not collide. Accepts either Op or Expr; falls back
    to repr-hash if neither is recognized so the function never raises in
    pathological op trees.
    """
    try:
        if hasattr(thing, "to_expr") and not hasattr(thing, "op"):
            thing = thing.to_expr()
        return HASHER.tokenize(thing)
    except Exception:
        import xxhash  # noqa: PLC0415

        return xxhash.xxh128(repr(thing).encode("utf-8")).hexdigest()


def _xorq_opaque_to_placeholder(node, _, **kwargs):
    """Replace opaque leaf nodes with UnboundTable placeholders.

    Mirrors xorq_dasher.rules.expr._opaque_to_placeholder but
    (a) uses content-stable hashes instead of ``id()`` so tokenize is
    reproducible across catalog reloads, and
    (b) folds the *parent/inner* expression's structural token into each
    placeholder name so wrappers with identical schema but distinct inner
    expressions do not collide.
    """
    from xorq.expr import api  # noqa: PLC0415
    from xorq.expr.relations import (  # noqa: PLC0415
        CachedNode,
        FlightExpr,
        FlightUDXF,
        HashingTag,
        Read,
        RemoteTable,
    )

    match node:
        case CachedNode():
            name = _stable_opaque_name(
                "cached",
                node.schema,
                type(node.cache).__name__,
                _parent_token(node.parent),
            )
        case Read():
            read_kwargs = dict(node.read_kwargs)
            anchor = read_kwargs.get("read_path") or read_kwargs.get("hash_path")
            name = _stable_opaque_name("read", node.schema, anchor)
        case RemoteTable():
            name = _stable_opaque_name(
                "remote",
                node.schema,
                _parent_token(node.remote_expr),
                getattr(node.source, "name", ""),
            )
        case FlightExpr():
            name = _stable_opaque_name(
                "flight-expr",
                node.schema,
                _parent_token(node.input_expr),
                _parent_token(node.unbound_expr),
            )
        case FlightUDXF():
            name = _stable_opaque_name(
                "flight-udxf",
                node.schema,
                _parent_token(node.input_expr),
                _parent_token(getattr(node.udxf, "exchange_f", None)),
            )
        case HashingTag():
            name = _stable_opaque_name(
                "tag",
                node.schema,
                node.metadata,
                _parent_token(node.parent),
            )
        case _:
            if kwargs:
                return node.__recreate__(kwargs)
            return node
    return api.table(node.schema, name=name).op()


def _normalize_expr_xorq(expr):
    """Deterministic Expr normalizer; replaces dasher's id()-based version."""
    from xorq_dasher.rules.expr import normalize_inmemorytable  # noqa: PLC0415

    from xorq.expr.api import get_compiler, to_sql  # noqa: PLC0415
    from xorq.expr.relations import CachedNode, Read  # noqa: PLC0415
    from xorq.vendor.ibis.expr.operations.relations import (  # noqa: PLC0415
        DatabaseTable,
        InMemoryTable,
    )
    from xorq.vendor.ibis.expr.operations.udf import AggUDF, ScalarUDF  # noqa: PLC0415

    op = expr.op()
    compiler = get_compiler(expr)
    sql = str(
        to_sql(
            op.replace(_xorq_opaque_to_placeholder).to_expr().unbind(),
            compiler=compiler,
        )
    )
    reads = op.find(Read)
    dts = tuple(
        n for n in op.find(DatabaseTable) if not isinstance(n, (CachedNode, Read))
    )
    udfs = op.find((AggUDF, ScalarUDF))
    mems = op.find(InMemoryTable)
    return (
        "ibis.Expr",
        sql,
        reads,
        dts,
        udfs,
        tuple(normalize_inmemorytable(m) for m in mems),
    )


def _databasetable_dispatcher(dt):
    """Dispatch DatabaseTable subclasses to their specific normalizers.

    xorq_dasher 0.1.0's normalize_databasetable does not handle the
    ``xorq_datafusion`` backend name (only ``xorq``) and its DatabaseTable
    rule outranks the more-specific Read/CachedNode/RemoteTable rules in
    MRO-with-earliest-match-wins lookup. This wrapper restores the
    most-specific-wins behavior xorq depends on.
    """
    from xorq_dasher.rules.expr import (  # noqa: PLC0415
        normalize_cached_node,
        normalize_databasetable,
        normalize_datafusion_databasetable,
        normalize_remote_table,
        normalize_xorq_databasetable,
    )

    from xorq.expr.relations import (  # noqa: PLC0415
        CachedNode,
        FlightExpr,
        FlightUDXF,
        Read,
        RemoteTable,
    )

    if isinstance(dt, Read):
        return _normalize_read_xorq(dt)
    if isinstance(dt, CachedNode):
        return normalize_cached_node(dt)
    if isinstance(dt, RemoteTable):
        return normalize_remote_table(dt)
    # FlightExpr/FlightUDXF carry input_expr / make_connection that the plain
    # datafusion path would silently flatten away — route to the dedicated
    # handler. Plain xorq_datafusion DTs still go through the datafusion path
    # (normalize_xorq_databasetable assumes ``_sources`` which not every
    # Backend variant exposes).
    if isinstance(dt, (FlightExpr, FlightUDXF)):
        return normalize_xorq_databasetable(dt)
    if dt.source.name == "xorq_datafusion":
        return normalize_datafusion_databasetable(dt)
    return normalize_databasetable(dt)


def _build_extra_rules():
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from xorq.expr.relations import Read  # noqa: PLC0415
    from xorq.vendor.ibis.expr.operations.relations import (  # noqa: PLC0415
        DatabaseTable,
        Schema,
    )
    from xorq.vendor.ibis.expr.types import Expr  # noqa: PLC0415

    rules = [
        (fqn(functools._lru_cache_wrapper), normalize_lru_cache),
        (fqn(property), normalize_property),
        (fqn(toolz.functoolz.Compose), normalize_toolz_compose),
        (fqn(toolz.curry), normalize_toolz_curry),
        (fqn(toolz.functoolz.excepts), normalize_toolz_excepts),
        (fqn(operator.methodcaller), normalize_methodcaller),
        (fqn(DatabaseTable), _databasetable_dispatcher),
        (fqn(Read), _normalize_read_xorq),
        (fqn(Expr), _normalize_expr_xorq),
        (fqn(Schema), normalize_ibis_schema),
        (fqn(np.dtype), normalize_numpy_dtype),
        (fqn(pd.Series), normalize_pandas_series),
        (fqn(pd.DataFrame), normalize_pandas_dataframe),
    ]
    try:
        from xorq.common.utils.toolz_utils import curry as xo_curry  # noqa: PLC0415

        rules.append((fqn(xo_curry), normalize_toolz_curry))
    except ImportError:
        pass
    return tuple(rules)


HASHER: Hasher = DEFAULT_HASHER.override(*_build_extra_rules())


def tokenize(*objs) -> str:
    """Return a deterministic hex digest for one or more objects."""
    return HASHER.tokenize(*objs)


def normalize(obj):
    """Return the primitive-tuple normalization of an object."""
    return HASHER.normalize(obj)


def snapshot_hasher(*extra_rules) -> Hasher:
    """Return a Hasher with snapshot-specific overrides layered on top of HASHER.

    Used by ``SnapshotStrategy`` to swap in backend / DatabaseTable / Read
    normalizers for the duration of a single key calculation.
    """
    return HASHER.override(*extra_rules)


__all__ = [
    "HASHER",
    "Hasher",
    "fqn",
    "tokenize",
    "normalize",
    "normalize_attrs",
    "snapshot_hasher",
]
