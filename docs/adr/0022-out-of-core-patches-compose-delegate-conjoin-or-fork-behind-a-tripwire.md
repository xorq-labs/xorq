# ADR-0022: Out-of-core patches compose — delegate, conjoin, or fork behind a source tripwire

- **Status:** Proposed
- **Date:** 2026-07-29
- **Deciders:** Dan Lovell

## Context

The extension-point audit behind ADR-0019/0020 established the durable line:
when a feature tempts a shim, a *primitive* is missing — add the capability
once, in core, and every backend composes it. But between "a user needs this
today" and "the enabler ships in a release" there is a legitimate gap, and
`prototype/rest-plugin-shim` lives in it: the rest/github/mixpanel backends
ship as entry-point plugins against **released** xorq 0.3.36 by patching
seven private core surfaces at import (`plugins/xorq-rest-plugin/src/
xorq_rest_plugin/shims.py`).

What the enabler principle does not say is *how* such a stop-gap may patch
core so that it stays safe while it lives. Review of the shim surfaced two
concrete hazards, both of which had passed a working test suite:

- A validator patched by **replacement**: the shimmed
  `check_for_exposed_secrets` ran the plugin's declared-key check *instead
  of* the original for plugin con_names, silently dropping the release's own
  default (unknown backends flag `password`). Nothing crashed; the guard was
  simply weaker.
- A function patched by **copy**: `_sanitize_generated_names` was forked
  wholesale (the needed branch sits mid-loop). A fork runs for *every*
  expression, plugin or not — if the pinned original ever drifts, the copy
  is unverified behavior on all builds, and a version-string warning is the
  only tell.

Reasonable people could disagree here: "it's temporary, pinned, and tested"
is a defensible position. This ADR takes the other side — a stop-gap's test
suite proves the new domain works; it cannot prove the old domain was left
alone. That property has to come from the patch's *shape*.

## Decision drivers

- Old-domain behavior (everything released xorq already does) must be
  preserved **by construction**, not by faithfully copying core and hoping.
- Guards may never get weaker by being extended.
- A patch that *can* silently diverge on non-plugin inputs must fail loudly
  rather than drift, because nothing else will catch it (build hashes do not
  fold rule provenance until ADR-0020 ships).
- Every patch should name its deletion path — the core enabler that retires
  it — keeping the shim-signals-missing-primitive line intact.

## Decision

Every out-of-core patch must keep the original callable **live inside the
composition**, in one of three sanctioned shapes, in strict order of
preference:

### 1. Guarded delegation (value-producing functions)

`patched(x) = plugin(x) if x ∈ new-domain else original(x)`, where the new
domain is one released core **crashes on or cannot construct** — path-less
Reads (vanilla raises `KeyError: 'hash_path'`), sources implementing a
protocol no release backend defines (`read_to_pyarrow_batches`). The
original is captured and called, never re-implemented; old-domain behavior
is identical because the old domain never reaches plugin code.

One widening is permitted: behavior on the old domain may change when it is
**transport-only under row/hash equivalence** (the bare-read streaming fast
path: same rows, identical Read hashes, nothing else observable). Identity
is never in scope for this clause.

### 2. Conjunction (guards and validators)

`patched(x) = original(x); plugin_check(x)` — both run, always. A patch may
**add rejections, never remove them**. Replacement is forbidden even when
the plugin's check looks strictly better informed: a replaced validator
weakens silently, because a missing rejection produces no crash, no warning,
and no failing test unless someone thought to write the negative case.

### 3. Fork + tripwire (last resort)

When the change sits mid-function with no seam — and the seam cannot be
created without reinterpreting shared core behavior — copy the function,
pin the sha256 of `inspect.getsource(original)`, and **refuse to patch on
mismatch** (`shims._assert_fork_source`). The asymmetry with the version
check is deliberate: delegating patches tolerate version drift with a
warning because the original stays live inside them; a fork whose original
drifted is undefined behavior on every input, so it hard-fails the import.

A fork is also the loudest possible TODO: each one names the core-enabler
commit that deletes it.

### Corollary: never widen a shared predicate

The tempting alternative to forking `_sanitize_generated_names` was teaching
`compiler._is_relocatable_read` to return True for path-less reads — one
line, no copy. But that predicate is consulted from multiple call sites,
including build-path asserts (`"relocatable Read must have hash_path"`), so
widening it reinterprets the old domain for *every* caller. When the only
available seam is a shared predicate, the honest options are a fork (shape
3) or waiting for the enabler.

### Corollary: names are the semantic contract

By-name registration (`read_source_identity` in `normalize_registry`) is
what lets artifacts built under the shim load against future core. The
obligation runs both ways: a semantic change to a registered function means
a **new name**, never a new body under the old name. ADR-0020's fingerprint
is deliberately body-blind, and `Hasher.override` on an existing rule key
replaces in place — the rule-name tuple is unchanged — so no identity
mechanism, present or planned, will catch a body swap.

## Alternatives considered

### No rule — per-patch judgment, backed by tests

Rejected because:
- Both hazards above passed review and a green suite in a careful prototype.
  Tests exercise the new domain; the old domain's preservation is exactly
  what nobody writes tests for.

### Forbid forks entirely

Rejected because:
- Against a *released* version, some changes genuinely have no seam
  (mid-loop branches). Forbidding the fork forces the predicate-widening
  move, which is worse: it corrupts shared behavior instead of copying it.
  The fork is acceptable precisely because it is pinned and loud.

### Version pin alone (`xorq==0.3.36`), no shape discipline

Rejected because:
- A pin polices *environments*, not patch structure. Editable installs,
  `--no-deps`, and future constraint loosening all slip it, and a
  replacement-shaped patch is silently wrong even on the pinned version
  (the secret-check hazard existed at 0.3.36).

## Consequences

### Positive

- Old-domain preservation becomes a structural property of shapes 1 and 2 —
  reviewable from the patch's form, not from re-deriving core behavior.
- Guard strength is monotone: extensions can only add rejections.
- The one fork fails loudly on drift instead of silently diverging; the
  tripwire converts "unverified on every build" into an import error.
- Each patch names its deletion path, so the shim stays a stop-gap rather
  than quietly becoming architecture.

### Negative

- The tripwire requires source availability (`inspect.getsource`), so a
  bytecode-only core distribution would break it — acceptable for xorq.
- A tripwire mismatch hard-fails the whole plugin import, even for users
  who never touch the forked path. Intended, but blunt.
- Conjunction can double-report an exposed key that both the original and
  the declared set flag; harmless, but the messages differ.
- The rule constrains stop-gaps only; it neither accelerates nor replaces
  the core enablers that retire them.

## References

- Implementation: `plugins/xorq-rest-plugin/src/xorq_rest_plugin/shims.py`
  (module docstring states the rule; `_assert_fork_source` is the tripwire),
  commit 8533613a; 13 tests in `plugins/xorq-rest-plugin/tests/test_plugin.py`
  including the conjunctive-secret and tripwire-drift cases.
- ADR-0019 (shim-vs-enabler line), ADR-0020 (body-blind fingerprint; names
  are the contract), ADR-0018 (by-name registries)
- `feat/rest-backend-config` f15361a5 — the streaming fast path shape 1
  mirrors; `feat/pathless-read-core-enablers` — the deletion path for the
  identity patches
- `notes/rest-api-source-registration-threads.md` — Thread C, the
  shim-signals-missing-primitive principle this ADR is the stop-gap
  counterpart to
