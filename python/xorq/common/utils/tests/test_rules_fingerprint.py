"""Rule-set fingerprint folded into the build hash (ADR-0020).

The identity-bearing rule set (dasher normalize/tokenize rules + the by-name
normalize_method registry) is a build-identity input under ADR-0015: a build
produced under a different rule set is a different artifact. These tests pin
the fingerprint's contract -- what it is and is not sensitive to -- and that it
folds into the *build* hash without disturbing the cache hash.
"""

from __future__ import annotations

from typing import Any

import xorq.api as xo
import xorq.common.utils.dasher as dasher
from xorq.common.utils.dasher import HASHER, Hasher, rules_fingerprint
from xorq.common.utils.provenance_utils import get_expr_hash
from xorq.ibis_yaml import normalize_registry as NR


def _dummy(_obj: Any) -> tuple:
    return ()


def _with_rules(rules: tuple) -> Hasher:
    return Hasher(rules=rules)


def test_fingerprint_is_deterministic() -> None:
    assert rules_fingerprint() == rules_fingerprint()
    assert NR.rules_fingerprint() == NR.rules_fingerprint()


def test_fingerprint_tracks_add_remove_reorder() -> None:
    base = rules_fingerprint()
    added = HASHER.override(("some.new.Type", _dummy))
    assert rules_fingerprint(added) != base
    removed = _with_rules(HASHER.rules[1:])
    assert rules_fingerprint(removed) != base
    # order is identity-bearing: earliest match wins on MRO ties
    reordered = _with_rules(HASHER.rules[::-1])
    assert rules_fingerprint(reordered) != base


def test_fingerprint_ignores_implementation_body() -> None:
    # names are the contract, not pickled callables (#2155): swapping every
    # rule's function while keeping names + order leaves the fingerprint intact
    same_names = _with_rules(tuple((name, _dummy) for name, _ in HASHER.rules))
    assert rules_fingerprint(same_names) == rules_fingerprint()


def test_build_hash_folds_the_rule_set() -> None:
    t = xo.memtable({"a": [1, 2, 3]})
    expr = t.filter(t.a > 1)
    base = get_expr_hash(expr)
    assert base == get_expr_hash(expr)  # deterministic

    # a rule-set change yields a different build identity, with no edit to the
    # expression itself
    original = HASHER.rules
    try:
        dasher.HASHER = HASHER.override(("some.new.Type", _dummy))
        assert get_expr_hash(expr) != base
    finally:
        dasher.HASHER = _with_rules(original)
    assert get_expr_hash(expr) == base  # restored
