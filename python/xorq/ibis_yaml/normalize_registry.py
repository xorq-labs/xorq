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


def deserialize_normalize_method(payload: Any) -> Optional[Callable]:
    # backward compat: pre-fix builds stored a bare base64 cloudpickle string.
    if isinstance(payload, str):
        return _legacy_unpickle(payload)
    kind = payload["kind"]
    if kind == "none":
        return None
    if kind == "named":
        name = payload.get("name")
        try:
            return _KEY_TO_FN[name]
        except KeyError:
            raise NormalizeMethodError(
                f"unknown normalize_method {name!r}; this build was "
                "produced by a newer or incompatible xorq version"
            ) from None
    if kind == "pickle":
        # reserved tag; only legacy/foreign builds reach here.
        return _legacy_unpickle(payload["pickle"])
    raise NormalizeMethodError(f"unknown normalize_method encoding {kind!r}")


def _legacy_unpickle(encoded: str) -> Callable:
    # Guards ModuleNotFoundError (the 0.3.28 case). CANNOT catch the 0.3.33
    # SIGSEGV -- cloudpickle can segfault inside loads before Python raises.
    # Only legacy/foreign builds reach here; new builds never pickle.
    from xorq.ibis_yaml.common import deserialize_callable  # noqa: PLC0415

    try:
        return deserialize_callable(encoded)
    except ModuleNotFoundError as e:
        raise NormalizeMethodError(
            "normalize_method was pickled against a module absent in this "
            f"environment ({e.name!r}); rebuild with a matching xorq version"
        ) from e
