"""By-name registry for ``Read.normalize_method``.

``normalize_method`` used to be cloudpickled into ``expr.yaml`` (see #2155). The
pickle embedded a module path, so loading a build produced by a different xorq
version failed uncatchably (``ModuleNotFoundError`` or SIGSEGV inside
``cloudpickle.loads``). We instead serialize it *by name* against this registry,
mirroring how ``Read.method_name`` has always been stored (a string resolved with
``getattr``).

The on-disk shape is a tagged envelope with a ``kind`` discriminator::

    {"kind": "none"}                              # normalize_method is None
    {"kind": "named", "name": "read_path_stat"}   # a registry key
    {"kind": "pickle", "pickle": "<base64>"}      # RESERVED -- never emitted

Only ``none`` and ``named`` are emitted. The ``pickle`` tag is reserved so that
re-enabling custom callables later stays purely additive (see the plan). Legacy
builds stored a bare base64 string; ``deserialize_normalize_method`` still reads
those through the guarded legacy branch.
"""

from __future__ import annotations

import pickle
from typing import Any, Callable, Optional

from xorq.common.exceptions import NormalizeMethodError
from xorq.common.utils.file_utils import (
    normalize_read_path_md5sum,
    normalize_read_path_stat,
)


# Append-only contract: these keys ARE the serialization contract. Never rename
# or repurpose a key; only append. (Same discipline as dasher._EXTRA_RULES.)
_NORMALIZE_RULES: tuple[tuple[str, object], ...] = (
    ("read_path_stat", normalize_read_path_stat),
    ("read_path_md5sum", normalize_read_path_md5sum),
)
_KEY_TO_FN = dict(_NORMALIZE_RULES)
_FN_TO_KEY = {fn: key for key, fn in _NORMALIZE_RULES}


def is_registered(fn: Optional[Callable]) -> bool:
    """True if ``fn`` (or ``None``) can be serialized by name."""
    return fn is None or fn in _FN_TO_KEY


def key_for(fn: Callable) -> str:
    """Registry key for a normalize_method, or raise.

    Used by the lockdown validators so an unserializable callable fails early --
    at Read construction / compiler init -- rather than at build time.
    """
    try:
        return _FN_TO_KEY[fn]
    except KeyError:
        raise NormalizeMethodError(
            f"{getattr(fn, '__qualname__', fn)!r} is not a registered "
            f"normalize_method; use one of {sorted(_KEY_TO_FN)}"
        ) from None


def validate(fn: Optional[Callable]) -> None:
    """Raise ``NormalizeMethodError`` unless ``fn`` (or ``None``) is serializable
    by name. Used to lock down the surfaces that can set ``normalize_method`` so
    an unserializable callable fails early rather than at build time."""
    if not is_registered(fn):
        key_for(fn)  # raises with the list of valid keys


def serialize_normalize_method(fn: Optional[Callable]) -> dict:
    if fn is None:
        return {"kind": "none"}
    return {"kind": "named", "name": key_for(fn)}


def _require(payload: dict, key: str) -> Any:
    """Fetch ``key`` from a tagged payload or raise a catchable error.

    Malformed/foreign builds must surface ``NormalizeMethodError`` -- the whole
    point of this module -- never a raw ``KeyError``. The message does *not*
    re-index ``payload`` (that would raise a second ``KeyError`` from inside the
    handler and mask the intended error)."""
    try:
        return payload[key]
    except KeyError:
        raise NormalizeMethodError(
            f"malformed normalize_method payload {payload!r}: missing {key!r}"
        ) from None


def deserialize_normalize_method(payload: Any) -> Optional[Callable]:
    # absent (pre-#1064 builds omitted the key entirely) -> no normalize_method.
    if payload is None:
        return None
    # backward compat: pre-fix builds stored a bare base64 cloudpickle string.
    if isinstance(payload, str):
        return _legacy_unpickle(payload)
    if not isinstance(payload, dict):
        raise NormalizeMethodError(
            f"malformed normalize_method payload {payload!r}: expected a tagged "
            "dict or a legacy base64 string"
        )
    kind = _require(payload, "kind")
    if kind == "none":
        return None
    if kind == "named":
        name = _require(payload, "name")
        try:
            return _KEY_TO_FN[name]
        except KeyError:
            raise NormalizeMethodError(
                f"unknown normalize_method {name!r}; this build was "
                "produced by a newer or incompatible xorq version"
            ) from None
    if kind == "pickle":
        # reserved tag; only legacy/foreign builds reach here.
        return _legacy_unpickle(_require(payload, "pickle"))
    raise NormalizeMethodError(f"unknown normalize_method encoding {kind!r}")


# cloudpickle.loads of a foreign/legacy pickle can fail many ways depending on
# how the target moved between versions: the module is gone (ModuleNotFoundError,
# a subclass of ImportError), the symbol moved within a surviving module
# (ImportError/AttributeError), or the payload itself is malformed
# (UnpicklingError/EOFError, or binascii.Error -- a ValueError subclass -- from
# b64decode). All must become a catchable NormalizeMethodError; only the SIGSEGV
# (module moved, 0.3.33) is uncatchable, and new builds never pickle at all.
_LEGACY_UNPICKLE_ERRORS = (
    ImportError,  # covers ModuleNotFoundError
    AttributeError,
    ValueError,  # covers binascii.Error
    EOFError,
    pickle.UnpicklingError,
)


def _legacy_unpickle(encoded: str) -> Callable:
    # CANNOT catch the 0.3.33 SIGSEGV -- cloudpickle can segfault inside loads
    # before Python raises. Only legacy/foreign builds reach here.
    from xorq.ibis_yaml.common import deserialize_callable  # noqa: PLC0415

    try:
        return deserialize_callable(encoded)
    except _LEGACY_UNPICKLE_ERRORS as e:
        # .name is only defined on ImportError subclasses; fall back otherwise.
        missing = getattr(e, "name", None)
        detail = (
            f"module {missing!r} is absent" if missing else f"{type(e).__name__}: {e}"
        )
        raise NormalizeMethodError(
            "normalize_method was pickled against an incompatible environment "
            f"({detail}); rebuild with a matching xorq version"
        ) from e
