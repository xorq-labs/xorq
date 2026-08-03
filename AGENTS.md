# xorq — working conventions

## Invariants need an enforcement tier

Every "must / cannot / never" sentence in a docstring or comment needs one of:

1. **Impossible by construction** — derive it, type it (e.g. `opaque_ops = tuple(OPAQUE_SPECS)`).
2. **Validated at import time** — attrs validators, `__attrs_post_init__`, module-level checks.
3. **Pinned by a test** co-located with the module that owns it.

A prose-only invariant is a review finding, not documentation. When touching code
whose docstring asserts a constraint, ask which tier enforces it; if none does, add one.

## Removing behavior requires a blast-radius grep

Before deleting any code path — including provably dead ones — grep the whole repo
(tests, docs, `.github/` included) for the removed **symbols** and their
**error-message strings**. Dead code can still have live tests: guards get
monkeypatched into firing, error messages get asserted with
`pytest.raises(match=...)` from other modules' test files. Use the
`/blast-radius` skill.

## Contract tests live with the contract owner

A test that pins module A's behavior belongs in A's test module, not in whichever
test file first observed the behavior. Monkeypatching another module's globals in
a test is a smell: it couples the test to internals invisible to that module's
editors, and it can silently stop testing anything when those internals are
restructured.

## Graph traversal / opaque ops (`common/utils/graph_utils.py`)

An op holding sub-expressions the `__children__` protocol does not surface
(`Expr`-typed fields, `__config__` payloads) **must** be registered in
`OPAQUE_SPECS` — otherwise hashing, lineage, and source discovery silently skip
its subtree. A runtime tripwire raises on unregistered `Expr`-bearing ops, and a
completeness test sweeps `Expr`-annotated fields; an `Expr` field that is
deliberately *not* a descent edge must be recorded in `NON_EDGE_EXPR_FIELDS`
with the reason.
