"""Standalone expression-token recomputation — only needs xxhash + struct.

``compute_expr_token`` reproduces ``tokenize(expr)`` from the hex strings
stored in ``expr_metadata`` output.  It embeds a copy of xorq_dasher's
``_encode``/``_write`` binary encoding so the only runtime dependencies are
``xxhash`` and the stdlib ``struct`` module — no xorq, ibis, or dasher
import required.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Union

import xxhash


if TYPE_CHECKING:
    Primitive = Union[
        None,
        bool,
        int,
        float,
        str,
        bytes,
        "tuple[Primitive, ...]",
        "list[Primitive]",
    ]

_TAG_NONE = b"\x00"
_TAG_BOOL = b"\x01"
_TAG_INT = b"\x02"
_TAG_FLOAT = b"\x03"
_TAG_STR = b"\x04"
_TAG_BYTES = b"\x05"
_TAG_SEQ = b"\x06"


def _write_fallback(obj: Primitive, out: list[bytes]) -> None:
    match obj:
        case None:
            out.append(_TAG_NONE)
        case bool():
            out.append(_TAG_BOOL)
            out.append(b"\x01" if obj else b"\x00")
        case int():
            out.append(_TAG_INT)
            sign = b"\x01" if obj < 0 else b"\x00"
            magnitude = abs(obj).to_bytes(
                (abs(obj).bit_length() + 7) // 8 or 1, "little"
            )
            out.append(sign)
            out.append(struct.pack("<I", len(magnitude)))
            out.append(magnitude)
        case float():
            out.append(_TAG_FLOAT)
            out.append(struct.pack("<d", obj))
        case str():
            encoded = obj.encode("utf-8")
            out.append(_TAG_STR)
            out.append(struct.pack("<I", len(encoded)))
            out.append(encoded)
        case bytes():
            out.append(_TAG_BYTES)
            out.append(struct.pack("<I", len(obj)))
            out.append(obj)
        case tuple() | list():
            out.append(_TAG_SEQ)
            out.append(struct.pack("<I", len(obj)))
            for el in obj:
                _write_fallback(el, out)
        case _:
            raise TypeError(
                f"Cannot encode type {type(obj).__name__!r}; "
                "expected nested tuples of str, int, float, bool, bytes, or None"
            )


def _encode_fallback(obj: Primitive) -> bytes:
    out: list[bytes] = []
    _write_fallback(obj, out)
    return b"".join(out)


try:
    from xorq_dasher.core import _encode
except ImportError:
    _encode = _encode_fallback


def compute_expr_token(structural_hash: str, slot_hashes: tuple[str, ...]) -> str:
    """Recompute an expression token from ``expr_metadata`` component hashes.

    Reproduces ``tokenize(expr)`` using only xxhash and struct.  The formula
    mirrors the binary encoding that ``Hasher.tokenize`` applies to the
    normalized tuple returned by ``_normalize_expr_xorq_impl``::

        inner = ("ibis.Expr.v4", structural_hash, slot_0, slot_1, ...)
        token = xxhash.xxh128(_encode((inner,))).hexdigest()
    """
    inner = ("ibis.Expr.v4", structural_hash) + tuple(slot_hashes)
    return xxhash.xxh128(_encode((inner,))).hexdigest()
