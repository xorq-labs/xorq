"""Project-wide deterministic hashing for xorq, built on xorq_dasher.

This package exposes the canonical ``HASHER`` instance and the ``tokenize`` /
``normalize`` helpers that the rest of xorq uses for cache keys, build
hashes, deterministic names, and lineage.

``DEFAULT_HASHER`` from xorq_dasher already covers ibis/xorq expression
types, Python callables (FunctionType/MethodType/CodeType/CellType/classmethod
/staticmethod), and the common builtins/numpy/pandas/pyarrow/sklearn rules.
This package adds the gap rules and xorq-specific normalizers split across
submodules:

* ``_gap_rules`` — stdlib/toolz/numpy/pandas/ibis-Schema normalizers
* ``_paths``     — catalog path canonicalization, stat helpers, plan/DDL
                   extractors
* ``_relations`` — ``Read`` and per-backend ``DatabaseTable`` normalizers,
                   plus the DatabaseTable dispatcher
* ``_opaque``    — opaque-leaf placeholder rewriting,
                   ``ExprScalarUDF.computed_kwargs_expr``, and the
                   ``Expr`` / ``ScalarUDF`` normalizers
"""

from __future__ import annotations

import contextvars
import functools
import operator

import toolz
from xorq_dasher import DEFAULT_HASHER, Hasher, fqn


# Active hasher for transitive tokenize calls (e.g. ``_parent_token`` inside
# the opaque-placeholder replacer). Snapshot strategy sets this so its
# data-blind rules propagate into recursive parent normalization. Unset →
# use global ``HASHER`` (the data-sensitive default).
#
# Defined at the top of ``__init__`` so submodules that import it during
# their own module load (e.g. ``_opaque`` would otherwise hit a cycle) can
# resolve it before the rest of this module finishes loading.
_current_hasher: contextvars.ContextVar[Hasher | None] = contextvars.ContextVar(
    "_xorq_current_hasher", default=None
)


# Submodule imports.  Order matters slightly: ``_paths`` and ``_gap_rules``
# have no in-package deps; ``_opaque`` lazily imports HASHER from this
# module so it can be loaded before HASHER is built; ``_relations`` depends
# on ``_paths`` + ``_opaque._rename_unbound_xorq``.
from xorq.common.utils.dasher._gap_rules import (  # noqa: E402
    normalize_attrs,
    normalize_builtin_callable,
    normalize_functools_partial,
    normalize_ibis_schema,
    normalize_lru_cache,
    normalize_methodcaller,
    normalize_numpy_dtype,
    normalize_pandas_dataframe,
    normalize_pandas_series,
    normalize_property,
    normalize_slice,
    normalize_toolz_compose,
    normalize_toolz_curry,
    normalize_toolz_excepts,
)
from xorq.common.utils.dasher._opaque import (  # noqa: E402, F401
    _normalize_computed_kwargs_expr,
    _normalize_expr_xorq,
    _normalize_scalar_udf_xorq,
    _xorq_opaque_to_placeholder,
)
from xorq.common.utils.dasher._opaque import (  # noqa: E402
    expr_metadata as _expr_metadata_unwrapped,
)

# Re-exported so existing callers (and tests) keep ``from
# xorq.common.utils.dasher import _canonicalize_catalog_path`` working after
# the package split; the trailing four are only used internally but importing
# them here lets the package's import-time wiring fail loudly if any
# submodule fails to load.
from xorq.common.utils.dasher._paths import (  # noqa: E402, F401
    _canonicalize_catalog_path,
    _extract_datafusion_plan_paths,
    _extract_duckdb_file_paths,
    _normalize_path_stat,
)
from xorq.common.utils.dasher._recompute import (  # noqa: E402, F401
    compute_expr_token,
)
from xorq.common.utils.dasher._relations import (  # noqa: E402, F401
    _databasetable_dispatcher,
    _normalize_read_xorq,
)


def _build_extra_rules():
    import types as _types  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from xorq.expr.relations import Read  # noqa: PLC0415
    from xorq.vendor.ibis.expr.operations.relations import (  # noqa: PLC0415
        DatabaseTable,
        Schema,
    )
    from xorq.vendor.ibis.expr.operations.udf import ScalarUDF  # noqa: PLC0415
    from xorq.vendor.ibis.expr.types import Expr  # noqa: PLC0415

    rules = [
        (fqn(functools._lru_cache_wrapper), normalize_lru_cache),
        (fqn(functools.partial), normalize_functools_partial),
        (fqn(_types.BuiltinFunctionType), normalize_builtin_callable),
        (fqn(_types.BuiltinMethodType), normalize_builtin_callable),
        (fqn(slice), normalize_slice),
        (fqn(property), normalize_property),
        (fqn(toolz.functoolz.Compose), normalize_toolz_compose),
        (fqn(toolz.curry), normalize_toolz_curry),
        (fqn(toolz.functoolz.excepts), normalize_toolz_excepts),
        (fqn(operator.methodcaller), normalize_methodcaller),
        (fqn(DatabaseTable), _databasetable_dispatcher),
        (fqn(Read), _normalize_read_xorq),
        (fqn(Expr), _normalize_expr_xorq),
        (fqn(Schema), normalize_ibis_schema),
        (fqn(ScalarUDF), _normalize_scalar_udf_xorq),
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


def _install_per_call_memos():
    """Install per-call memos for the duration of one outer tokenize/normalize.

    Returns the list of contextvar reset tokens; the caller must reset each
    in ``finally``.  Only installs each memo if it isn't already set, so
    nested ``tokenize`` calls share the outermost memo (snapshot strategy
    cooperatively installs the same memos via :func:`with_caches`).
    """
    from xorq.common.utils.dasher._opaque import (  # noqa: PLC0415
        _expr_normalize_memo,
        _parent_token_memo,
    )
    from xorq.common.utils.dasher._relations import _dt_normalize_memo  # noqa: PLC0415

    tokens = []
    if _parent_token_memo.get() is None:
        tokens.append((_parent_token_memo, _parent_token_memo.set({})))
    if _dt_normalize_memo.get() is None:
        tokens.append((_dt_normalize_memo, _dt_normalize_memo.set({})))
    if _expr_normalize_memo.get() is None:
        tokens.append((_expr_normalize_memo, _expr_normalize_memo.set({})))
    return tokens


def _reset_per_call_memos(tokens):
    for var, token in tokens:
        var.reset(token)


def with_caches(fn):
    """Wrap a callable so a single invocation installs+tears down the dasher
    per-call memos.  Use at user-facing entry points (``tokenize``,
    ``SnapshotStrategy.calc_key``) so all rule invocations within the call
    share the same memos."""
    import functools  # noqa: PLC0415

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        tokens = _install_per_call_memos()
        try:
            return fn(*args, **kwargs)
        finally:
            _reset_per_call_memos(tokens)

    return wrapper


@with_caches
def tokenize(*objs) -> str:
    """Return a deterministic hex digest for one or more objects."""
    return HASHER.tokenize(*objs)


@with_caches
def normalize(obj):
    """Return the primitive-tuple normalization of an object."""
    return HASHER.normalize(obj)


@with_caches
def expr_metadata(expr) -> dict:
    """Decompose an expression token into structural + per-slot data hashes.

    See :func:`~xorq.common.utils.dasher._opaque.expr_metadata` for details.
    """
    return _expr_metadata_unwrapped(expr)


def snapshot_hasher(*extra_rules) -> Hasher:
    """Return a Hasher with snapshot-specific overrides layered on top of HASHER.

    Used by ``SnapshotStrategy`` to swap in backend / DatabaseTable / Read
    normalizers for the duration of a single key calculation.
    """
    return HASHER.override(*extra_rules)


__all__ = [
    "HASHER",
    "Hasher",
    "compute_expr_token",
    "expr_metadata",
    "fqn",
    "tokenize",
    "normalize",
    "normalize_attrs",
    "snapshot_hasher",
]
