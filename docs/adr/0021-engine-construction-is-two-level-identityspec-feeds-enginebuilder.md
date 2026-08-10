# ADR-0021: Engine construction is two-level — a cross-engine IdentitySpec feeds an EngineBuilder; per-engine specs feed `build()`

- **Status:** Proposed
- **Date:** 2026-07-30
- **Deciders:** Dan Lovell

> **This ADR is a proposal, not a description of the system. None of the
> machinery below is implemented.** `IdentitySpec`, `EngineBuilder`, and
> `DEFAULT_BUILDER` do not exist anywhere in the tree — as of this branch they
> appear only in this document. The Decision section is written in the
> indicative for readability; read it as the design, not as the code. What
> *does* exist is ADR-0020's fingerprint fold, which this ADR builds on.
>
> Drafted ahead of implementation, deliberately — and landing that way, in
> PR #2200, with none of its machinery. That departs from the project
> convention that an ADR lands with a credible implementation, paired with its
> implementing PR. The reason is sequencing, not doubt about the decision:
> phases 1 and 2 are unblocked work with no owning stack entry yet, and phase 3
> threads per-engine specs through vendored-ibis backend construction — a
> high-conflict surface that should not be touched while the REST backend
> entries are still in flight. The document lands now so the design stays
> reviewable beside ADR-0020's fold, which it builds on.
>
> **Acceptance gate:** phases 1 and 2 of the Phases section land — the frozen
> `IdentitySpec` / `EngineBuilder` / `DEFAULT_BUILDER` values with no hash
> change (phase 1), and the fingerprint-agreement assert at the tokenize-time
> normalization chokepoint (phase 2, as revised by amendment 1). Phase 3
> (per-engine spec threading) is explicitly *not* part of the gate. Tracked as
> [#2202](https://github.com/xorq-labs/xorq/issues/2202) — the gate has an
> owner, so "Proposed" here is a scheduled state, not an open-ended one.

## Context

ADR-0020 landed the fingerprint fold (rule-set drift is now a build-identity
change) and committed the partition: compilers are per-engine; normalize/
tokenize rules must stay globally consistent within a graph. It scoped "the
full spec construction" — an engine whose behavior is a declarative value —
as follow-up. This ADR designs that construction.

The open question 0020 left is *how the sharing of the cross-engine rules is
enforced*. Today it is ambient: `HASHER` is a module global, and
`normalization_context` / `_current_hasher` swap the active hasher per call
(the `SnapshotStrategy` seam). Ambient sharing is correct but has costs 0020
itself records: swapping rules is a global operation, testing identity
machinery means monkeypatching module state, and "engines agree on identity
rules" is a fact about imports rather than about values you can construct
and inspect. The naive fix — make the rule tables per-engine constructor
arguments — was rejected in 0020: identity rules that vary per engine
corrupt cache identity silently within one multi-backend graph.

The resolution discussed and adopted here: everything becomes a constructor
argument, but the *cross-engine* part is passed to a **builder**, not to
engines — so agreement is structural, not ambient, and per-engine variation
of identity rules stays inexpressible.

## Decision drivers

- Preserve ADR-0020's grain: fingerprints fold into the **build** hash;
  cache hashes stay rule-set-neutral. Nothing here may reintroduce a
  per-expr-tokenize fold.
- Per-engine variation of identity rules must remain inexpressible, not
  merely discouraged (the 0020 partition).
- Rehydration paths (`Profile.get_con`, entry points) construct backends
  with no builder in hand and must keep working unchanged.
- `tokenize` is called from utility code with no engine in scope; identity
  is a property of the session executing a graph, not of any single engine.
- Reuse the existing seams (`Hasher.override`, `normalization_context`,
  `_current_hasher`) rather than inventing parallel mechanisms.

## Decision

### Two-level construction

```
IdentitySpec (cross-engine, frozen)      # hasher rule table, normalize_method
  │                                      # registry, declared strategy layers
  ▼
EngineBuilder(identity_spec)             # the sharing point
  │
  ├─ build(rest_spec)       → rest engine        # per-engine: compiler,
  ├─ build(datafusion_spec) → datafusion engine  # paginators, auth appliers
  └─ build(duckdb_spec)     → duckdb engine
```

Engines built from one builder share the `IdentitySpec` by construction:
there is no per-engine slot for identity rules to sit in, so the divergence
0020 rejected is inexpressible where the module global merely made it
invisible. `IdentitySpec` composes the existing primitives — the
`xorq_dasher.Hasher` value and the by-name `normalize_method` registry — it
does not rehome them (0020 deferred that deliberately).

This is ADR-rest-config-contract-identity-folded-residence-either's move
applied one level up: that ADR made an API a declarative config passed to a
shared backend class; 0021 makes an engine a declarative config passed to a
shared builder, itself constructed from the declarative cross-engine spec.

### The builder owns the fingerprints and the ambient hasher

The builder computes its spec's fingerprints (ADR-0020's
`dasher.rules_fingerprint` / `normalize_registry.rules_fingerprint`) once at
construction, and is the value that supplies them to the build-hash fold.
It also *installs* the ambient hasher through the existing
`normalization_context` / `_current_hasher` seam — utility-code `tokenize`
keeps working with no engine in hand, but the ambient value now has an
owner. Strategy-specific variation (`snapshot_hasher()`-style overrides)
becomes a **declared layer on the spec** — uniform across the graph within
its context and visible in the fingerprint — the sanctioned form of
variation, as opposed to per-engine divergence, which stays inexpressible.

### Mixed-builder graphs fail loudly at composition

Two builders can now coexist — the failure mode the module global
prevented. Engines carry their builder's spec fingerprint; graph-composition
points (`into_backend`, cache-key computation, RemoteTable/tee assembly)
assert all participants match and raise on mismatch.

This guard is **load-bearing, not belt-and-suspenders**: because ADR-0020
deliberately keeps cache hashes rule-set-neutral, a mixed-builder graph
that slipped past the guard could still collide in the *cache* under
divergent normalization. The build-hash fold catches cross-process build
identity; only the composition-time assert protects cache identity. The
guard must therefore cover every composition point before phase 2 ships —
its completeness is the acceptance criterion, not an optimization.

### `DEFAULT_BUILDER`: the global, demoted

Profiles and build artifacts rehydrate backends via entry points with no
builder in hand, so a canonical `DEFAULT_BUILDER` (today's `HASHER`-shaped
module state, wrapped) must exist. The win is not eliminating the global
but demoting it from *the mechanism* to *the default value of an explicit
parameter*: tests construct toy-spec builders without monkeypatching;
production code that never mentions builders behaves exactly as today.

### Phases (implementation gate for Accepted status)

1. **The builder value** — `IdentitySpec` + `EngineBuilder` as frozen
   values; `DEFAULT_BUILDER` wraps today's `HASHER` + registry; builder
   installs the ambient hasher via the existing seam. No behavior change,
   no hash change.
2. **The composition guard** — engines carry the spec fingerprint;
   `into_backend`/cache-key/tee assembly assert agreement. Gate: an audit
   that every composition point is covered.
3. **Per-engine spec threading** — compilers, paginators, auth appliers
   become `build()` inputs; includes namespacing the base paginator/auth
   names (the ADR-rest-config-contract-identity-folded-residence-either
   append-only hazard) before the catalog grows. This
   phase touches vendored-ibis backend construction — a high-conflict
   surface — and should wait until in-flight backend work settles.

## Alternatives considered

### Per-engine identity rules

Rejected because (restating ADR-0020's line, which this ADR exists to keep):
- Divergent normalizer sets within one graph corrupt cache identity
  silently. The two-level split exists precisely so "everything is a
  constructor argument" does not imply "identity rules vary per engine."

### Status quo: ambient module global only

Rejected because:
- Agreement-by-import is untestable without monkeypatching, uninspectable,
  and unconstructible; legitimate variation stays an ad-hoc contextvar
  dance rather than a declared, fingerprint-visible layer. (The ambient
  *mechanism* is retained — demoted to the builder-installed default.)

### A session/context object instead of a builder

Identity is arguably a property of the session executing a graph, so the
spec could live on a session object consulted at execution, with engines
left unbound.

Deferred because:
- It pushes agreement checks to every execution instead of binding them at
  construction, and gives rehydration no natural home for the default. The
  builder composes with a future session concept (a session would *hold* a
  builder); nothing here precludes it.

### A mutable `register()` extension API

Rejected because:
- Import-order-dependent identity: what a fingerprint means would depend on
  which modules registered first. Frozen composition (`.override()`
  returning a new value) keeps the spec a value with a stable fingerprint.

## Consequences

### Positive

- The 0020 partition becomes structural: identity-rule divergence is
  inexpressible per engine, declared-and-fingerprinted per strategy layer.
- Identity machinery becomes testable with toy-spec builders — no module
  monkeypatching.
- The spec is a value: constructible, inspectable, pinnable, and
  reconstructible — the north star 0020 named.
- Per-engine registries (compilers, paginators, auth) get a uniform,
  declarative home in phase 3, closing the extension-point audit that
  started this series.

### Negative

- The composition guard's completeness is a correctness obligation (cache
  hashes are deliberately fingerprint-neutral, so the guard is the only
  cache-side defense against mixed builders). Phase 2 cannot ship partial.
- `DEFAULT_BUILDER` remains a global — demoted, not eliminated; rehydration
  depends on it.
- Phase 3 touches vendored-ibis construction, historically high-conflict;
  sequencing after in-flight backend work is load-bearing.
- Two ways to hold a hasher exist during migration (ambient default vs
  builder-installed); phase 1 must make them the same object to avoid a
  drift window.

## References

- ADR-0020 (the fingerprint fold and partition this builds on; its
  "Per-engine rule swapping is not yet wired" negative is what phases 1–3
  retire), ADR-rest-config-contract-identity-folded-residence-either
  (declarative config precedent; paginator/auth
  namespacing hazard), ADR-0015 (build vs cache hash grain)
- `xorq_dasher.Hasher` (`core.py:84`); `dasher/__init__.py`
  (`_current_hasher`, `snapshot_hasher`); `caching/strategy.py`
  (`SnapshotStrategy.normalization_context`); `provenance_utils.py`
  (`get_expr_hash` fold, which calls it)
- The shim-vs-core-enabler extension-point audit behind
  ADR-rest-resource-reads-are-lazy-datafusion-tables/0020

## Amendment (2026-07-30)

The two-level construction stands; two mechanisms are revised before any
phase ships, and one gap is split out into its own ADR.

1. **The composition guard moves to a chokepoint (revises phase 2).** The
   original text rests cache-side safety on asserting spec agreement at
   every composition point (`into_backend`, cache-key computation,
   RemoteTable/tee assembly), with an audit of completeness as the
   acceptance gate. That is the same remembered-fact pattern ADR-0020
   criticized in the FQN-drift test: an audit polices code, not identity.
   Instead, the assert moves to the one place every cache-identity
   computation already passes through — backend normalization during
   tokenize. Engines carry their builder's spec fingerprint;
   `normalization_context(expr)` (which already receives the expr and can
   see `expr.ls.backends`) asserts every participating engine's fingerprint
   matches the ambient builder's, raising on mismatch. Anything that
   hashes, checks — by construction. Composition-point asserts may remain
   as earlier, friendlier errors, but they are no longer the correctness
   obligation, and phase 2's "cannot ship partial" negative dissolves.

2. **Ambient installation is scoped, not construction-time (clarifies
   phase 1).** "The builder installs the ambient hasher" left unspecified
   what happens when two builders coexist. Revised: the ambient hasher is
   a contextvar whose *default* is `DEFAULT_BUILDER`'s value, and a
   builder takes scope explicitly — `with builder.active(): ...` — setting
   and resetting the token. Coexisting builders are nested contexts with
   well-defined extents, thread-safe via contextvar propagation, and phase
   1's "two ways to hold a hasher must be the same object" holds because
   the default *is* the builder's value.

3. **The extension story is split out as ADR-0023.** This ADR gives
   rehydration `DEFAULT_BUILDER` but says nothing about how entry-point
   plugins extend it — and an import-time mutation of the default spec is
   exactly the import-order-dependent identity the "mutable `register()`"
   alternative rejects. ADR-0023 (identity-spec contributions as entry
   points, composed order-independently) owns that decision.
