# ADR-0020: Engine behavior is an immutable, identity-folded declarative spec

- **Status:** Proposed
- **Date:** 2026-07-30
- **Deciders:** Dan Lovell

## Context

xorq's behavior is governed by several rule tables: the dasher normalize/
tokenize rules (`dasher._EXTRA_RULES` → `HASHER`), the by-name
`normalize_method` registry (`ibis_yaml/normalize_registry.py`), the REST
paginator/auth registries (ADR-0018), and per-backend compilers. Reviewing
the REST streaming work surfaced a recurring shape: some of these are clean,
composable extension points and some accrete shims, and the difference is
learnable (see the shim-vs-core-enabler analysis behind ADR-0019).

Two of these tables — `_EXTRA_RULES` and `_NORMALIZE_RULES` — are *literally
the same type* (`tuple[tuple[str, object], ...]`) with the *same* append-only
discipline comment, and both are **identity-bearing**: the set of
normalize/tokenize rules in force determines what data a build's hashes
identify. Yet that dependency is implicit. `_NORMALIZE_RULES` is a fixed
module global; the dasher rule set is policed only by a hand-maintained
FQN-drift test (`test_dasher.py:1101`). Under ADR-0015 ("anything that changes
what data comes back must change the build hash"), a change to the rule set is
exactly such a change — but nothing folds it into identity. A contributor who
adds or reorders a rule shifts every build's meaning with no hash to catch it.

Separately, the tables are module globals. That is correct for tokenize/
normalize (see the partition below) but blocks the natural end state — an
engine whose full behavior is a declarative value you can construct, inspect,
pin, and reconstruct.

## Decision drivers

- The identity-determining rule set must be a first-class, hashable build
  input (ADR-0015), not a fact policed by a test that must be remembered.
- Rules must stay serializable *by name*, never as captured/pickled callables
  (the #2155 lesson that motivated `normalize_registry` in the first place).
- Tokenize/normalize must remain globally consistent within a single
  multi-backend expression graph (an `into_backend` graph spans backends;
  divergent normalization would corrupt cache identity mid-graph).
- Reuse the existing composable primitive rather than inventing another
  registry abstraction.

## Decision

### The frozen composable registry already exists — adopt it as the model

`xorq_dasher.Hasher` is `@frozen` with an inspectable `rules` tuple and
`.override()` / `.without()` returning new instances (`core.py:84`). It is
exactly the "named, append-only, immutable, composable rule table" the other
registries reinvent piecemeal. `normalize_registry` is a *different-semantics*
table (by-name selection of a chosen `normalize_method`, not MRO type
dispatch), so it is not merged into `Hasher`; instead it adopts the same
discipline — a content fingerprint over its stable keys.

### Fold the rule-set fingerprint into the build hash

Both tables expose a stable fingerprint:

- `dasher.rules_fingerprint(hasher)` — digest of the **ordered** rule-name
  tuple (order is identity-bearing: earliest match wins on MRO ties).
- `normalize_registry.rules_fingerprint()` — digest of the **sorted** key set
  (lookup is by name; order is not identity-bearing here).

Both are folded into `get_expr_hash` (`provenance_utils.py`), the **build**
hash — *not* per-expr `tokenize`. This is the load-bearing granularity choice:
the build hash answers "was this pipeline, under these rules, built?"; the
cache hash must stay rule-set-neutral so in-run and cross-version cache reuse
is unaffected. One global fingerprint means a multi-backend graph is salted
consistently.

The fingerprint is sensitive to rules **added, removed, or reordered** and
deliberately **insensitive to a rule's implementation body** under an
unchanged name — names are the contract, not pickled callables (#2155).
Implementation-body changes remain out of identity scope, the same line
ADR-0017 draws for `fetch_override`.

Verified by prototype (this branch): fingerprint deterministic and stable
across cold processes; build hash changes iff the rule set changes, with no
edit to the expression; `test_dasher` (106), `test_provenance_utils`,
`test_cache_pin`, and REST `into_backend`/cache roundtrips (55 total) all pass
— the suites assert build-hash *properties*, not golden literals, so folding
shifts absolute values without breaking asserted relationships.

### North star: engine behavior as a constructed spec (partitioned)

The end state is an engine constructed from an immutable spec bundling its
compiler, normalize/tokenize rules, paginators, and auth. The seams already
exist — `Hasher.override` composes rule sets, and `normalization_context`
(`provenance_utils.py:19`, used by `SnapshotStrategy`) already swaps the
active hasher per call. What this ADR commits to is the **partition** any such
spec must respect:

- **Per-engine**: compilers (backend-local; no cross-object consistency
  requirement).
- **Globally consistent, or folded into identity**: tokenize/normalize rules.
  A per-engine normalizer set that diverged within one graph would corrupt
  cache identity silently; folding the fingerprint into the build hash makes
  any divergence a *detectable* identity change rather than a silent one.

The full spec construction is scoped as follow-up; this ADR lands the
fingerprint fold that makes it safe.

## Alternatives considered

### Fold the fingerprint into per-expr `tokenize`

Rejected because:
- It changes every cache key whenever any rule is added, breaking in-run cache
  reuse and all cross-version cache sharing. The build hash is the correct
  ADR-0015 grain; the cache hash must stay rule-set-neutral.

### Keep the hand-maintained FQN-drift test as the only guard

Rejected because:
- A test polices *code*, not *identity*. Two builds under different rule sets
  would still collide in the catalog. The drift test remains useful as an
  early, human-readable warning, but it is not an identity mechanism.

### Per-engine normalize/tokenize rules without folding

Rejected because:
- An `into_backend` graph spanning two engines with different normalizer sets
  would tokenize the same sub-expression inconsistently — a silent cache-
  identity corruption. Identity rules must be shared or folded; configurability
  there buys nothing but risk.

### Invent a new `Registry` abstraction for all tables

Deferred because:
- `Hasher` already is the frozen composable registry. The normalize registry
  needs only the fingerprint, not a rehoming. A broader unification can follow
  once the spec construction is built.

## Consequences

### Positive

- Rule-set drift is now a build-identity change caught by the hash, not only
  by a remembered test — the ADR-0015 line extended to the engine's own
  machinery.
- The fingerprint reuses `Hasher`'s existing composability; no new abstraction.
- Establishes the partition (compiler per-engine; identity rules shared/folded)
  that makes a future declarative EngineSpec safe to build on the existing
  `Hasher.override` / `normalization_context` seams.

### Negative

- **One-time build-hash migration**: existing build artifacts get new hashes
  once, as their now-folded rule set becomes identity-bearing. A graduation
  cost, exactly ADR-0018's framing; no golden-literal test churn was observed,
  but catalog artifacts rebuild.
- **Body-blind fingerprint**: a rule whose implementation changes under an
  unchanged name is invisible to identity (same #2155 line). Accepted; names
  are the contract.
- **Per-engine rule swapping is not yet wired**: swapping `HASHER` today is a
  global operation. True per-engine specs need the `normalization_context` /
  `_current_hasher` seam threaded through construction — scoped as follow-up.

## References

- ADR-0015 (build vs cache hash), ADR-0017, ADR-0018, ADR-0019
- `xorq_dasher.Hasher` (`core.py:84`) — the frozen composable registry
- `test_dasher.py:1101` — the hand-maintained FQN-drift check this promotes
- Prototype (this branch): `dasher.rules_fingerprint`,
  `normalize_registry.rules_fingerprint`, the `get_expr_hash` fold, and
  `test_rules_fingerprint.py`

## Amendment (2026-07-30)

The decision stands; four implementation details of the fingerprint are
revised, prompted by the `prototype/rest-plugin-shim` experience (a plugin's
most natural mutation — `Hasher.override` on an existing rule key — replaces
in place and was invisible to the fingerprint as originally specified).

1. **Fingerprints digest `(rule-key, normalizer-name)` pairs, not keys
   alone.** `Hasher.override` on an existing key preserves the key tuple, so
   rule *replacement* — the fourth mutation verb, and the one extensions
   actually use — was invisible. Digesting the normalizer's
   `module.qualname` alongside each key makes replacement identity-visible
   while staying inside the #2155 line: still names, never pickled bodies.
   An implementation-body edit under an unchanged function name remains
   deliberately out of identity scope. The same treatment applies to
   `normalize_registry.rules_fingerprint` (sorted `(key, fn-name)` pairs).

2. **The fold uses the declared regime, not the in-context hasher.**
   `SnapshotStrategy._build_hasher` appends per-expression backend-FQN
   rules, so fingerprinting the context hasher made the "rule-set
   fingerprint" a function of the expr — detectable drift, but not a regime
   identifier comparable across builds. The strategy's static rules are now
   split into `declared_rules()` / `declared_hasher()`, and `get_expr_hash`
   folds `rules_fingerprint(strategy.declared_hasher())`. Per-expression
   derived rules are excluded: they carry no information the expr hash
   lacks.

3. **Fingerprints are recorded, not only salted.** Folding alone makes rule
   drift *detectable* (hashes differ) but not *diagnosable* (nothing says
   which regime built an artifact). `build_provenance_metadata` now stamps
   `dasher_rules_fingerprint` and `normalize_registry_fingerprint` as
   provenance fields alongside `expr_hash`.

4. **Digests are computed with `hashlib`, not `HASHER.tokenize`.** Routing
   a fingerprint through the rule table made one table's fingerprint a
   function of the other table's rules (and dasher's self-referential).
   Plain `sha256` over the delimited name sequence decouples them.

Consequence for the original text: the "body-blind fingerprint" negative
narrows — a *replaced* rule is now visible; only a body edit under an
unchanged name remains invisible, which is the intended contract.
