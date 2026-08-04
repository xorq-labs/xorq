# ADR-0018: REST APIs are declarative configs behind one backend; identity is always folded, config residence is a per-API packaging choice

- **Status:** Accepted
- **Date:** 2026-07-24
- **Deciders:** Dan Lovell

## Context

Phases 1–2 (ADR-0016, ADR-0017) made one API (Mixpanel) a profile-carrying
backend whose resources are path-less `Read` ops. The Phase 3 target is a
broad catalog (tens of APIs). Writing a hand-rolled backend per API does not
scale; adopting dlt wholesale conflicts with xorq's determinism (schema
evolution, stateful cursors — see plans/udxf-source-api-backend.md, Phase 3
"superseded framing"). The mixpanel backend already contained the answer
inlined: resource schemas, a readers dict, URL maps (the config) and a
generic `_deferred_read` (the machinery).

Two questions had to be settled before generalizing (plan open question 4):
does the resource config participate in expression identity, and does it
live in code or in the profile?

## Decision drivers

- ADR-0015: anything that changes what data comes back must change the build
  hash. Editing a resource's `path` is a query change, not a refactor.
- ADR-0016: credentials resolve at execution; nothing serialized carries
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

`normalize_read_source_identity` folds the **per-resource**
`ResourceConfig.content_hash` (declarative fields only; `fetch_override`
excluded — code stays refactorable per ADR-0017's line). Editing a
resource's path/params/paginator changes build and cache hashes; editing a
sibling resource does not (verified for config-in-code; config-in-profile
deliberately trades this away since the profile covers the whole config).

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
checked, so a static `_secret_keys` cannot express them:
`get_dynamic_secret_keys` prefers a backend classmethod
`_get_secret_keys(kwargs)` (reads `config.auth.secret_fields`) over the static
`con_name_to_secret_keys` mirror, falling back to the mirror, then
`("password",)`. It consults the hook without importing anything — only an
already-imported backend is inspected — which the `rest` path satisfies:
having a rest config in hand means the backend module is loaded.

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
  yaml → `get_con` → live fetch, with raw secrets rejected via dynamic keys.
- Build hashes reproducible across processes (`f619195575ac` twice) and
  per-resource sensitive.

### Negative

- **Graduation cost**: the same logical read has different identity via
  self-service (`rest` + config-in-profile) vs a later curated subclass
  (different con_name and profile) — promotion invalidates caches and
  build hashes, one-time and expected.
- Self-service profiles are only as portable as their configs: a schema
  typo is a profile edit (new identity), not a code fix.
- `PandasBackend` memory semantics now apply to every config'd API; large
  reads want `.cache()` to parquet or param-partitioned reads (ADR-0017's
  trade-off, at catalog scale).
- Paginator names and `ResourceConfig` field names are identity-bearing
  and therefore effectively append-only; backend-registered paginator and
  auth-kind names share that property and should be namespaced
  (`"mixpanel.session_id"`) to keep the base set unambiguous.
- Override-only resources have near-empty declarative identity: with
  `fetch_override` excluded from `content_hash` (code stays refactorable),
  a resource that is 100% override code — mixpanel's, deliberately — is
  identified by schema and params alone, so changing what the override
  fetches keeps cache hits. Accepted: the same line ADR-0017 draws, at its
  sharpest.

## References

- ADR-0015, ADR-0016, ADR-0017
- plans/udxf-source-api-backend.md — Phase 3 and open question 4's
  resolution ("identity: always folded; residence: either")
- dlt `rest_api` / `RESTAPIConfig` (https://dlthub.com/docs) — the shape the
  omissions are defined against
