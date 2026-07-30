# ADR-0023: Identity-spec contributions are entry points, composed order-independently into DEFAULT_BUILDER

- **Status:** Proposed
- **Date:** 2026-07-30
- **Deciders:** Dan Lovell

> Drafted ahead of implementation, like ADR-0021 (this branch is
> discussion-stage; the ADR-lands-with-credible-implementation discipline is
> relaxed by agreement). The named gate for Accepted status: a prototype
> re-expressing the rest plugin's identity mutations as a single declared
> contribution, with the composed fingerprint visible in a build.

## Context

ADR-0021 gives rehydration and builder-less code a canonical
`DEFAULT_BUILDER`, but is silent on how that default gets *extended*. The
gap is not hypothetical: `prototype/rest-plugin-shim` ships backends via the
`xorq.backends` entry-point group, and its identity needs (a dasher rule
override, a `normalize_registry` key) are met today by mutating module
state at package import. Under ADR-0021's own analysis that is the rejected
alternative — a mutable `register()` API whose meaning depends on which
modules imported first. Yet plugins have no other move: they are loaded by
entry point, hold no builder, and cannot be constructor arguments to a
default that exists before they load.

Left unresolved, every plugin re-derives the shim pattern (ADR-0022's
composition rule bounds the damage but does not remove the ambient
mutation), and `DEFAULT_BUILDER`'s fingerprint means "whatever happened to
be imported" — the exact fact ADR-0020 set out to make identity-visible.

## Decision drivers

- Extension must be **installed-set semantics, not import-order semantics**:
  two processes with the same installed packages must compose the same
  spec and fingerprint, regardless of import sequence.
- Contributions must be **fingerprint-visible** (ADR-0020: rule-set drift
  is a build-identity change; with the 2026-07-30 amendment, so is rule
  replacement).
- Per-engine variation of identity rules must stay inexpressible
  (ADR-0020/0021 partition) — contributing to the *shared* spec is the
  only sanctioned channel.
- Rehydration (`Profile.get_con`, entry-point backends) must keep working
  with no code changes in profiles or artifacts.
- Reuse existing primitives: frozen `Hasher.override` composition,
  by-name registries, the entry-point mechanism plugins already use for
  backends.

## Decision

### A new entry-point group: `xorq.identity_specs`

A plugin declares a **contribution** — a frozen, declarative value naming
what it adds to the shared identity spec:

```toml
[project.entry-points."xorq.identity_specs"]
rest = "xorq_rest_plugin.identity:CONTRIBUTION"
```

A contribution carries: dasher rules to add (`(fqn, normalizer)` pairs),
`normalize_registry` entries to add (`(key, fn)` pairs), and nothing else.
Both are by-name surfaces (ADR-0018/0020 discipline); a contribution cannot
carry per-engine material (compilers, paginators, auth — those are
`build()` inputs per ADR-0021 phase 3).

### Composition is eager, total, and sorted

At first construction of `DEFAULT_BUILDER`, xorq loads **all** installed
`xorq.identity_specs` entry points — not lazily, not on plugin import —
and composes them in **sorted entry-point-name order** onto the base
`IdentitySpec`. Import order becomes irrelevant: the composed spec is a
function of the installed set alone. Every contributed name lands in the
rule tables and therefore in the fingerprints (visible under the ADR-0020
amendment's `(key, normalizer-name)` digest).

### Conflicts are errors, not last-wins

Two contributions claiming the same dasher rule key or the same registry
key is a hard `ImportError`-grade failure at composition time. Last-wins
would reintroduce order-dependence through the back door (sorted order
would silently pick a winner); explicit conflict forces the two plugins to
coordinate names — the append-only discipline both tables already state.
A contribution may not override a *base* rule either: overriding core
normalization from a plugin is ADR-0022 shim territory (a stop-gap with a
deletion path), never a durable registration.

### Constructed builders are unaffected

Contributions extend `DEFAULT_BUILDER` only. An explicitly constructed
`EngineBuilder(identity_spec)` gets exactly the spec it was handed —
tests and hermetic embedders opt out of the installed world by
construction. A constructor flag (`with_contributions=True`) can opt a
custom builder in.

## Alternatives considered

### Plugins mutate the default spec at import (status quo of the shim)

Rejected because:
- Import-order-dependent identity — ADR-0021 already rejected this shape
  for core; a plugin doing it via package import side effects is the same
  mechanism wearing an entry point as a trigger.

### Lazy composition (compose when a plugin's backend first connects)

Rejected because:
- The spec — and its fingerprint — would change mid-process the first time
  a plugin backend is touched, so two expressions in one session could hash
  under different regimes. Eager totality keeps one regime per process.

### Last-wins conflict resolution in sorted order

Rejected because:
- Deterministic is not the same as intentional: sorted-order tie-breaking
  silently privileges alphabetically-earlier package names and hides real
  coordination failures. Names are the contract; contested names are a bug.

### Contributions may override base rules

Rejected because:
- A durable extension that changes core normalization changes what every
  existing build's hash *means*. That power stays confined to ADR-0022
  stop-gaps, which are pinned, loud, and named for deletion.

## Consequences

### Positive

- Plugins get a sanctioned, order-independent identity channel; the rest
  plugin's dasher/registry shims become one declared contribution, deleted
  per its README's graduation plan.
- `DEFAULT_BUILDER`'s fingerprint becomes a statement about the installed
  environment — reproducible, diagnosable (recorded in provenance per the
  ADR-0020 amendment), and comparable across processes.
- The ADR-0020/0021 partition survives extension: contributions land in
  the shared spec; per-engine variation stays inexpressible.

### Negative

- Eager loading of all `xorq.identity_specs` entry points at first
  `DEFAULT_BUILDER` construction adds import cost for installed-but-unused
  plugins (mitigable: contributions should be tiny modules that import no
  heavy dependencies; the backend entry point stays separate and lazy).
- A broken contribution breaks `DEFAULT_BUILDER` construction for the whole
  process — loud by design, but a bad neighbor can take down sessions that
  never use it.
- Installed-set semantics makes the plugin set itself identity-bearing:
  installing or removing a contributing plugin changes `DEFAULT_BUILDER`'s
  fingerprint — and therefore build hashes — even for pipelines that never
  touch that plugin's rules. This is the honest reading of ADR-0020 (the
  regime in force *is* different), but it means environment churn shows up
  as build-identity churn; the provenance-recorded fingerprint is what
  makes such churn diagnosable rather than mysterious.
- One more entry-point group to document and keep stable.

## References

- ADR-0021 (the builder this extends; its 2026-07-30 amendment names this
  ADR as the owner of the extension gap), ADR-0020 + amendment
  (fingerprint visibility of contributed and replaced rules), ADR-0022
  (the stop-gap composition rule contributions graduate from), ADR-0018
  (by-name registries; append-only hazard)
- `prototype/rest-plugin-shim` — `shims._patch_dasher` /
  `_patch_normalize_registry`, the two mutations a contribution replaces
- `notes/rest-api-source-registration-threads.md` — Thread B
  (registration as packaging) meets Thread D (registrations as
  identity-folded values); this ADR is their intersection
