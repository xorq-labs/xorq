"""Stable digests of an identity-bearing rule table (ADR-0020).

Two tables fold into the build hash: the dasher normalize/tokenize rules
(``dasher._EXTRA_RULES`` -> ``HASHER``) and the by-name ``normalize_method``
registry (``ibis_yaml.normalize_registry``). They share the encoding here but
never each other's data -- each digest is ``hashlib.sha256`` over its own pairs
only, so one table's fingerprint cannot depend on the other's rules.

That independence is a property of the *inputs*, not of having two copies of
the encoder: what ADR-0020 forbids is routing a fingerprint through
``HASHER.tokenize`` (which would make it rule-set-dependent), not sharing a
pure function. Sharing it is what keeps the two schemes from drifting apart
under later edits.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Iterable


# Bumped when the encoding below changes for reasons unrelated to the rules
# themselves -- a new field, different framing, a different digest. Without it
# an encoding change is indistinguishable from a rule-set change: both move
# every build hash, and nothing in the artifact says which happened. Same
# discipline as ``dasher._canonical.NORMALIZATION_VERSION``.
FINGERPRINT_SCHEME_VERSION = "rf1"

_RECORD_SEP = "\x00"
_FIELD_SEP = "\x1f"


def normalizer_name(fn: Callable) -> str:
    """``module.qualname`` for a rule's normalizer, or raise.

    The name *is* the contract (#2155): it is what the fingerprint digests, and
    therefore what a build hash commits to. A callable whose name is missing or
    not unique cannot carry that contract, so it is rejected rather than
    silently weakening identity.

    Raising here means a bad rule fails where it is declared. Without it the
    failure surfaces as an ``AttributeError`` deep inside ``get_expr_hash`` --
    every build breaks, with a traceback that never mentions rule
    registration.
    """
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname:
        raise TypeError(
            f"identity rule normalizer {fn!r} has no module-qualified name. "
            "Normalizers must be named functions or methods -- not partials, "
            "callable objects, or builtins -- because the name is what the "
            "rule-set fingerprint (ADR-0020) commits to."
        )
    if "<lambda>" in qualname or "<locals>" in qualname:
        raise TypeError(
            f"identity rule normalizer {module}.{qualname} is a lambda or a "
            "closure. Its name is not unique, so replacing it with a different "
            "implementation defined in the same scope would be invisible to "
            "the rule-set fingerprint (ADR-0020). Use a module-level function."
        )
    return f"{module}.{qualname}"


def validate_rule_pairs(pairs: Iterable[tuple[str, Callable]]) -> tuple:
    """Check that every ``(key, normalizer)`` pair can be fingerprinted.

    Called where rules are *registered* so the error names the offending rule,
    rather than at build time where it would only name the callable. Returns
    the pairs as a tuple so it can wrap a rule table inline.
    """
    pairs = tuple(pairs)
    for key, fn in pairs:
        if _RECORD_SEP in key or _FIELD_SEP in key:
            raise ValueError(
                f"identity rule key {key!r} contains a fingerprint field "
                "separator, which would make the digest ambiguous"
            )
        normalizer_name(fn)
    return pairs


def fingerprint_rule_pairs(pairs: Iterable[tuple[str, Callable]]) -> str:
    """sha256 over ``(rule key, normalizer name)`` pairs, in the order given.

    Order is preserved, not imposed: callers sort first when their lookup is
    by name, and pass declaration order when earliest-match-wins makes order
    identity-bearing.
    """
    return hashlib.sha256(
        _RECORD_SEP.join(
            (
                FINGERPRINT_SCHEME_VERSION,
                *(
                    f"{key}{_FIELD_SEP}{normalizer_name(fn)}"
                    for key, fn in validate_rule_pairs(pairs)
                ),
            )
        ).encode()
    ).hexdigest()
