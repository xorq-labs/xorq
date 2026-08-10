# ADR-XXXX: REST APIs are declarative configs behind one backend; identity is always folded, config residence is a per-API packaging choice

- **Status:** Accepted
- **Date:** 2026-07-24
- **Deciders:** Dan Lovell

## Context

Phases 1–2 (ADR-build-artifacts-are-credential-free,
ADR-api-relations-are-pathless-read-ops) made one API (Mixpanel) a
profile-carrying backend whose resources are path-less `Read` ops. The Phase 3
target is a broad catalog (tens of APIs). Writing a hand-rolled backend per API
does not scale; adopting dlt wholesale conflicts with xorq's determinism
(schema evolution, stateful cursors — see plans/udxf-source-api-backend.md,
Phase 3 "superseded framing"). The mixpanel backend already contained the
answer inlined: resource schemas, a readers dict, URL maps (the config) and a
generic `_deferred_read` (the machinery).

Two questions had to be settled before generalizing (plan open question 4):
does the resource config participate in expression identity, and does it
live in code or in the profile?

## Decision drivers

- ADR-0015: anything that changes what data comes back must change the build
  hash. Editing a resource's `path` is a query change, not a refactor.
- ADR-build-artifacts-are-credential-free: credentials resolve at execution;
nothing serialized carries
  values. Any config mechanism must preserve this across N APIs.
- Broad catalog: adding a clean API must cost a config, not an engine; users
  must be able to define APIs without waiting for a release.
- Editing one resource must not invalidate builds/caches of its siblings.

## Decision

### The contract

`RestBackendConfig(base_urls, auth, resources)` /
`ResourceConfig(name, schema, path, base_url_key, record_path, paginator,
paginator_kwargs, params, residual_column, fetch_override)` /
`AuthConfig(kind, fields, secret_fields)` / `ParamSpec(name, required,
kind)` (`backends/rest/config.py`), executed by `RestBackend(PandasBackend)`
(`backends/rest/__init__.py`): resources are path-less `Read` ops
(`method_name="fetch_resource"`, the resource name in `read_kwargs`);
`fetch_resource` runs at the `make_dt` boundary through an explicit engine
seam (`backends/rest/engines.py`): an `Engine` protocol whose default,
`NativeEngine`, drives native paginators (`backends/rest/paginators.py`:
header_link, json_link, offset, page_number, single_page — original
implementations; the strategy interface is shaped after dlt's
`BasePaginator`, no code copied) and whose `FetchOverrideEngine` adapts
`fetch_override` callables. Engines carry the extension registries:
backends extend paginators and auth kinds by merging class-level mappings
(`paginators`, `auth_appliers`) over the base registries — declaration,
not private-method override. The engine-equivalence obligation: same
config, any engine, same rows.

Three deliberate omissions vs dlt's `RESTAPIConfig` (each load-bearing):
no incremental/cursor state (ranges are explicit `ParamSpec(kind="range")`
params in `read_kwargs`; chunking = constructing multiple reads); auth names
profile *fields*, never values; `fetch_override` as the code-fallback for
bespoke resources (Mixpanel's NDJSON export and session_id engage — the
all-override backend proving the escape hatch, with `mixpanel/client.py` as
the hand-rolled conformance baseline).

### Identity: always folded

`normalize_read_source_identity` folds **two** hashes, both derived from the
attrs declaration (`identity_field_names`) rather than hand-listed:

- the **per-resource** `ResourceConfig.content_hash` (declarative fields only;
  `fetch_override` excluded — code stays refactorable per
  ADR-api-relations-are-pathless-read-ops's line). Editing a resource's
  path/params/paginator changes build and cache hashes; editing a sibling
  resource does not (verified for config-in-code; config-in-profile
  deliberately trades this away since the profile covers the whole config).
- the **API-wide** `RestBackendConfig.content_hash` — the resolved `base_urls`
  and the whole auth shape.

The second is not decoration. The rule is *the resolved endpoint is
identity-bearing, not just the name of the route to it*: a resource folds
`base_url_key` ("default", "data"), which is a key into a mapping the resource
cannot see. A curated profile carries credentials only, so with only the
per-resource hash, repointing `base_urls` from prod to staging changed nothing
any read hashed on — and cached data from the old host was served as current
data from the new one. `resources` is excluded from the API-wide hash precisely
so sibling independence survives: whole-config folding would undo the first
bullet.

Membership in both hashes is derived from `attrs.fields`, so a field added to
either class is identity-bearing by default and excluding one takes an explicit
`metadata={"identity": False}` annotation. The direction of that default is the
point: a hand-written tuple drifts toward *under*-hashing, whose failure is a
stale cache hit; deriving makes the worst case a spurious miss.

### Residence: either, per API

- **Curated** (config in code): a subclass with a class-level `config` and
  its own entry point (`github`, `mixpanel`). Profiles stay
  credentials-shaped.
- **Self-service** (config in profile): the base ships as the `rest` entry
  point; `do_connect(config=<plain dict>, **credentials)` — the config is
  yaml-safe data captured by `Profile.from_con`, so a saved profile carries
  the API definition and rehydrates without any code release.
  `fetch_override` cannot ride this path (callables don't ride in yaml) —
  config-expressible APIs only.

### Dynamic secret keys

The `rest` backend's secret keys live inside the `config` kwarg being
checked, so a static `_secret_keys` cannot express them in full: the backend
declares the classmethod `_get_secret_keys(kwargs)`, which reads
`config.auth.secret_fields`. `check_for_exposed_secrets` **unions** that hook
with the unconditional `("password",)` and the static
`con_name_to_secret_keys` mirror, so every tier can only widen the check.

Resolution reads `sys.modules` first and otherwise imports the backend. An
earlier draft inspected `sys.modules` alone, to keep a heavy backend out of
`Profile.save`; that made the check's answer depend on the importing history
of the process, so a hand-authored profile naming an auth field outside the
mirror was rejected in one process and saved its credential in the clear in
another. The import is confined to that cold path — anything holding a live
connection is already in `sys.modules` and pays nothing.

The mirror still matters, for a different case: a backend whose extra is not
installed cannot be imported at all, the hook cannot answer, and the union
falls back to `("password",)` — a key that matches none of these backends'
fields. So all three REST-family names carry mirror entries: `github` and
`mixpanel` mirror their configs' `secret_fields` exactly, and `rest`, whose
field names are config-defined and therefore not statically knowable, carries
a deliberate **floor** of the conventional credential kwarg names that the
hook widens per config.

## Alternatives considered

### dlt as *the* extraction engine (no xorq-native config layer)

Deferred, demoted to one engine behind the contract for long-tail APIs: a
`ResourceConfig` may compile to a stateless dlt `rest_api` source. Gated on
the spike criteria (hash-stable determinism; no credential capture; lossless
compilation) plus the general engine-equivalence obligation: same config,
any engine, same rows.

### Config-in-profile only (no curated subclasses)

Rejected because:
- Profiles become program-sized (the whole API definition in every
  profiles.yaml) and every config tweak is a new backend identity,
  invalidating caches of untouched sibling resources.

### Config-in-code without the identity fold (Phase 2's implicit state)

Rejected because:
- Editing `path`/`record_path` changed returned data without changing any
  hash — a silent violation of ADR-0015.

### A per-API hand-rolled backend (Phase 1/2 shape, times N)

Rejected because:
- The mixpanel backend's own structure showed the config/machinery split;
  N copies of the machinery is the re-invention Phase 3 exists to avoid.

## Consequences

### Positive

- A clean API costs a config: the github backend is config-only, zero
  override code, and live-verified (161 issues across 4 header-link pages,
  cross-checked against the repo record's `open_issues_count`).
- Users define APIs without releases; self-service profiles round-trip
  yaml → `get_con` → live fetch, with raw secrets rejected via the union of
  the dynamic config-derived keys and the static mirror.
- Build hashes reproducible across processes, per-resource sensitive, and
  sensitive to the resolved endpoint. (An earlier draft pinned a literal hash
  here as evidence; the value moved when the API-wide hash was folded in, so
  the claim is stated as the property the tests assert rather than a number
  that decays.)

### Negative

- **Graduation cost**: the same logical read has different identity via
  self-service (`rest` + config-in-profile) vs a later curated subclass
  (different con_name and profile) — promotion invalidates caches and
  build hashes, one-time and expected.
- Self-service profiles are only as portable as their configs: a schema
  typo is a profile edit (new identity), not a code fix.
- `PandasBackend` memory semantics now apply to every config'd API; large
  reads want `.cache()` to parquet or param-partitioned reads
  (ADR-api-relations-are-pathless-read-ops's trade-off, at catalog scale).
- Paginator names and `ResourceConfig` field names are identity-bearing
  and therefore effectively append-only; backend-registered paginator and
  auth-kind names share that property and should be namespaced
  (`"mixpanel.session_id"`) to keep the base set unambiguous.
- Deriving identity membership from `attrs.fields` means **adding any field to
  either config class moves every rest read hash once**, even at its default
  value, and even for resources that never set it. Accepted deliberately: the
  alternative default — new fields excluded until someone remembers — fails as
  a stale cache hit on data, while this fails as a one-time recompute.
- Override-only resources have near-empty declarative identity: with
  `fetch_override` excluded from `content_hash` (code stays refactorable), a
  resource that is 100% override code — mixpanel's, deliberately — is
  identified by schema and params alone, so changing what the override fetches
  keeps cache hits. Accepted: the same line
  ADR-api-relations-are-pathless-read-ops draws, at its sharpest.

## References

- ADR-0015, ADR-build-artifacts-are-credential-free,
ADR-api-relations-are-pathless-read-ops
- plans/udxf-source-api-backend.md — Phase 3 and open question 4's
  resolution ("identity: always folded; residence: either")
- dlt `rest_api` / `RESTAPIConfig` (https://dlthub.com/docs) — the shape the
  omissions are defined against

## Amendment (2026-08-07)

The decision stands. Two consequences are added from an audit that classified
every identity failure on this branch as *conservative* (identity too sensitive
→ spurious cache miss → duplicated work, acceptable) or *non-conservative*
(identity too coarse → two different things collide → a stale or wrong cache
hit). The audit's finding was that this branch's deliberate choices push
failures conservative wherever a derivation can do it — the `attrs`-derived
membership above is the clearest instance — and that what remains
non-conservative clusters where something meaning-bearing lives *outside* the
folded set. These two are the ones this ADR owns.

### 1. Curated artifacts record their config fingerprint but never verify it

A third failure class the original text does not distinguish: a
**build-hash-integrity** failure, where the hash lies about what an artifact
will do but no cache ever serves wrong rows.

`read_identity_parts` reads `self.current_config` **live**
(`backends/rest/__init__.py`), and in curated residence that config lives in
installed code (`backends/github/__init__.py`), not in the artifact. So an
artifact built under version N and rehydrated under N+1 — where a curated
config changed `path`, `record_path` or `paginator` but not the schema —
executes against a different endpoint while still named by its old build hash.

Two things bound this, and both matter:

- **It is not a cache collision.** Cache keys are recomputed live at run time
  from the same `current_config`, so a drifted rehydration gets *moved* keys: a
  conservative miss returning correct, new-config rows. The hash lies; the
  cache does not. It is dangerous only to consumers that key data reuse or
  equivalence on the **build hash** — catalog dedup, cross-run comparison, and
  any future result reuse keyed on build hash. That list is the risk register.
- **The artifact already records enough to detect it.** `register_node`
  (`ibis_yaml/common.py`) writes `snapshot_hash = content_hash(node)` for every
  node, and for a rest `Read` that hash folds the api-wide and per-resource
  content hashes via the path-less snapshot branch. Nothing ever reads it back
  to compare — it is consumed only as a lineage node handle. So the remedy is
  *verify at load*, not *record then verify*: recompute and compare, refusing on
  mismatch and warning for artifacts written before the check existed.

Note the inversion this exposes in the residence choice above. Residence is
presented as a neutral per-API packaging decision with costs on both sides, and
that framing holds — but on *this* axis they are not symmetric.
**Self-service pins its config and curated does not**, because a self-service
config rides in the profile, and the profile is serialized. The "graduation
cost" negative already notes that promotion changes identity; this adds that
promotion also changes *which* copy of the config an artifact is bound to.

Deliberately not fixed here. Verify-at-load is cheap and is the recommended
shape if it is; serializing the curated config is rejected, because it would
freeze config bugs and security fixes into every old artifact and erase the
distinction this ADR chose on purpose.

### 2. The override-identity acceptance is withdrawn as a standing position

The last negative above accepts that a 100%-`fetch_override` resource is
identified by schema and params alone, so *changing what the override fetches
keeps cache hits*. Under the conservative/non-conservative split that is a
non-conservative acceptance — a stale hit on data, which is the failure
direction every other decision in this ADR is arranged to avoid. It is also
sharper in practice than the text admits: mixpanel's real regional data and
query URLs live in `mixpanel/client.py`, not in the folded `base_urls`, so this
ADR's own rule — *the resolved endpoint is identity-bearing, not just the name
of the route to it* — is de facto void for that backend.

The acceptance is therefore scheduled for retirement rather than left standing.
The mechanism is a required `override_version` field on `ResourceConfig`:
identity-bearing by the derived-membership rule above, and rejected at config
assembly when `fetch_override` is set without it. Its contract is that the
author bumps it when the override's *meaning* changes and leaves it alone
across refactors — the ADR-api-relations-are-pathless-read-ops refactorability
line kept intact, with a name to hold accountable instead of an invisible
exclusion. A useful side effect: today an override resource and a config-driven
resource with identical declarative fields hash identically, because the
opt-out erases even the *presence* of an override; a non-`None`
`override_version` makes override-ness itself identity-visible.

Rejected strengthener, recorded so it is not re-proposed: folding
`fetch_override`'s `module.qualname` into `content_hash`. It would make
"which function" identity-visible, but rename-and-move is exactly the refactor
the exclusion exists to protect, and the name is not the contract here the way
a registry key's name is.

**Gate:** the negative flips to retired when `override_version` lands and
mixpanel's `events`/`engage` declare one. Until then the negative stands as
written and this amendment is the record that it is not endorsed.

Adding the field moves every rest read hash once, at its default value — the
priced cost of derived membership, stated in the negative above. It should ride
one batched, adjudicated migration together with the other identity-moving
follow-ups rather than paying that cost twice.
