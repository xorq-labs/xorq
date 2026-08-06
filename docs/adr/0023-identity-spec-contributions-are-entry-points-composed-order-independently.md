# ADR-0023: Identity-spec contributions are entry points, composed order-independently into DEFAULT_BUILDER

- **Status:** Proposed
- **Date:** 2026-07-30
- **Deciders:** Dan Lovell

> **This ADR is a proposal, not a description of the system. None of the
> machinery below is implemented.** The `xorq.identity_specs` entry-point group
> is declared in no `pyproject.toml` and loaded by no code; there is no
> contribution type, no composition step, and no conflict check. Nor does the
> `DEFAULT_BUILDER` this extends exist — it is proposed by ADR-0021, which is
> itself unimplemented. As of this branch every name below appears only in this
> document. The Decision section is written in the indicative for readability;
> read it as the design, not as the code.
>
> Drafted ahead of implementation, like ADR-0021, and landing unimplemented in
> the same PR (#2200) — the same departure from the
> ADR-lands-with-implementation convention, for a sharper reason: this ADR's
> gate cannot be met at this head at all. It has nothing to extend until
> ADR-0021 phase 1 makes `DEFAULT_BUILDER` a value, and its prototype
> re-expresses a REST plugin that reaches `main` in a later stack entry. What
> is *not* deferred is the obligation to name the predecessors this would
> dissolve — see "The transitional duck-typed protocols this is meant to
> dissolve" below.
>
> **Acceptance gate:** ADR-0021 phase 1 lands (this ADR has nothing to extend
> until `DEFAULT_BUILDER` is a value — tracked as
> [#2202](https://github.com/xorq-labs/xorq/issues/2202)), and a prototype
> re-expresses the rest plugin's two identity mutations as a single declared
> contribution, with the composed fingerprint visible in a build.
>
> One section below is *not* proposal: "The transitional duck-typed protocols
> this is meant to dissolve" describes seams that really exist. Two of the
> three do, at the file:line references given there — see the note on the
> third.

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

### The transitional duck-typed protocols this is meant to dissolve

Three source-side protocols already carry identity or transport across the
core/backend seam by duck typing rather than by declaration. They are
**transitional predecessors**, named here so the entry-point channel does not
quietly become a *second*, competing extension protocol alongside them:

- **`read_identity_parts`** — a method looked up with `getattr` on a read's
  source (`normalize_read_source_identity`,
  `python/xorq/common/utils/file_utils.py`); its only producer is the rest
  backend in this stack. It lets a source append identity parts of its own to
  a path-less `Read`.
- **`read_to_pyarrow_batches`** — a method looked up the same way on the
  source (`_maybe_streaming_read_reader`, `python/xorq/expr/api.py`), by which
  a backend opts a bare `Read` into page-wise streaming. Transport, not
  identity — but it is the same unnamed-protocol shape, and a plugin
  discovers it only by reading core. **Not present on this branch:** this seam
  was described from the prototype; core carries no
  `_maybe_streaming_read_reader`. It is named here for the shape, which the
  other two do exhibit, and because any future streaming opt-in should be a
  declared capability rather than a third `getattr` protocol.
- **`read.source._profile`** — the private reach-in inside
  `normalize_read_source_identity`, where core takes a backend's profile out
  of a private attribute to hash it. Identity depends on an attribute no
  contract names.

All three are transitional for one reason: they are **unnamed and
per-instance**. Presence is decided by whether a particular object happens to
carry an attribute, so nothing about them reaches the fingerprint regime this
ADR and ADR-0020 build — a source that starts or stops contributing identity
parts, or a private attribute that is renamed, changes what a build hash means
with no visible rule-set change. Names are the contract; these have no names
in any table, and so cannot be fingerprinted, conflict-checked, or diffed
across processes.

The intent is that each dissolves into declared identity contributions: the
identity parts a source wants folded become a contribution (or a declared
per-read contribution keyed by name) rather than a method core hopes to find,
and the profile hash becomes something the source *states* instead of
something core extracts. Streaming, being transport, should end up as a
declared capability on the backend rather than a second `getattr` protocol.
Until then they stay as they are — pinned by ADR-0022's stop-gap discipline,
named here for deletion.

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
- The transitional protocols above: `normalize_read_source_identity`
  (`python/xorq/common/utils/file_utils.py:138` for the `read_identity_parts`
  lookup, `:123` for the `_profile` reach-in). The third,
  `_maybe_streaming_read_reader` / `read_to_pyarrow_batches`, is a prototype
  shape with no counterpart in core on this branch.
- `notes/rest-api-source-registration-threads.md` — Thread B
  (registration as packaging) meets Thread D (registrations as
  identity-folded values); this ADR is their intersection
