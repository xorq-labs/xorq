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

A change landing as several parts also gets one pass reading the parts
together: a guard that is latent in the part adding it goes live in the part
adding its first caller, and no per-part review can rank that correctly.

## Documents describe the tree they land in

Never write a design doc or docstring in the perfect tense about work that has
not landed ("the workaround is deleted", "verified: all N tests pass").
Intended work goes under Consequences; a Proposed status also names what does
not exist yet and the gate for accepting it. This recurs even right after being
diagnosed, so check it deliberately: grep changed docs for "verified", "now",
"currently", "this branch".

## Isolate before you diagnose

A story that explains the evidence is not the cause. Apply the one suspected
commit alone and check the result reproduces byte-identically. Re-run
load-bearing numbers yourself instead of trusting a report — including your own
earlier one.

## Graph traversal / opaque ops (`common/utils/graph_utils.py`)

An op holding sub-expressions that `__children__` does not surface
(`Expr`-typed fields, `__config__` payloads) MUST be registered in
`OPAQUE_SPECS`; a deliberately-not-descended `Expr` field goes in
`NON_EDGE_EXPR_FIELDS` with the reason. Runtime tripwires and a completeness
test enforce this, except for `__config__` payloads (invisible to both —
review-enforced). Full rationale: ADR-0016.
