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
from xorq.caching.strategy import SnapshotStrategy
from xorq.common.enums import ProvenanceField
from xorq.common.utils.dasher import HASHER, Hasher, rules_fingerprint
from xorq.common.utils.provenance_utils import (
    build_provenance_metadata,
    get_expr_hash,
)
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


def test_fingerprint_tracks_replacement() -> None:
    # overriding an existing rule key with a differently-named function is a
    # rule-regime change and must move the fingerprint (the normalizer is
    # identified by its module-qualified name)
    key, _fn = HASHER.rules[0]
    replaced = HASHER.override((key, _dummy))
    assert len(replaced.rules) == len(HASHER.rules)  # replaced in place
    assert rules_fingerprint(replaced) != rules_fingerprint()
    # ...but the digest input is names-only by construction (#2155: names are
    # the contract, never pickled callables): a hasher rebuilt from the SAME
    # (key, fn) pairs fingerprints identically
    rebuilt = _with_rules(tuple(HASHER.rules))
    assert rules_fingerprint(rebuilt) == rules_fingerprint()


def test_declared_regime_fingerprint_is_expr_independent() -> None:
    # get_expr_hash folds the *declared* regime (base rules + strategy layer),
    # not the in-context hasher, which carries per-expression derived backend
    # rules. The declared fingerprint must be deterministic and free of any
    # per-expr contribution.
    strategy = SnapshotStrategy()
    fp = rules_fingerprint(strategy.declared_hasher())
    assert fp == rules_fingerprint(SnapshotStrategy().declared_hasher())
    # and it is a strategy-layer regime: distinct from the base HASHER's
    assert fp != rules_fingerprint()


def test_provenance_metadata_records_fingerprints() -> None:
    expr = xo.memtable({"a": [1]})
    metadata = build_provenance_metadata(expr, SnapshotStrategy(), object())
    dasher_fp = metadata[ProvenanceField.dasher_rules_fingerprint.encode()]
    registry_fp = metadata[ProvenanceField.normalize_registry_fingerprint.encode()]
    assert dasher_fp and registry_fp
    assert dasher_fp.decode() == rules_fingerprint(SnapshotStrategy().declared_hasher())
    assert registry_fp.decode() == NR.rules_fingerprint()


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
