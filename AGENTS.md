# xorq — working conventions

Sources of truth: `CONTRIBUTING.md` § "Removing or changing behavior" (process
conventions) and `docs/adr/` (design decisions). Summaries below; when in
doubt, read those.

## Removing behavior requires a blast-radius grep

Before deleting any code path — including provably dead ones — grep the whole
repo (tests, docs, `.github/`) for the removed symbols AND their error-message
strings. Dead code can still have live tests. Use the `/blast-radius` skill.

## Invariants need an enforcement tier

Every "must / cannot / never" in a docstring needs one of: impossible by
construction, validated at import time, or pinned by a co-located test. A
prose-only invariant is a review finding, not documentation.

## Contract tests live with the contract owner

A test pinning module A's behavior belongs in A's test module. Monkeypatching
another module's globals in a test is a smell.

## Sequential reviews + retrospective (trial policy)

Shared-infrastructure changes (traversal, hashing, serialization) get two or
more independent cold reviews; when rounds produce findings, run
`/review-retro` before merge. See CONTRIBUTING § "Review policy:
shared-infrastructure changes".

## Graph traversal / opaque ops (`common/utils/graph_utils.py`)

An op holding sub-expressions that `__children__` does not surface
(`Expr`-typed fields, `__config__` payloads) MUST be registered in
`OPAQUE_SPECS`; a deliberately-not-descended `Expr` field goes in
`NON_EDGE_EXPR_FIELDS` with the reason. Runtime tripwires and a completeness
test enforce this, except for `__config__` payloads (invisible to both —
review-enforced). Full rationale: ADR-0016.
