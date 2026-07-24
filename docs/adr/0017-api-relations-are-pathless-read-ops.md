# ADR-0017: API-backed relations are path-less Read ops; their identity is the registered source-identity normalizer

- **Status:** Accepted
- **Date:** 2026-07-24
- **Deciders:** Dan Lovell

## Context

Phase 1 of API-as-Backend (ADR-0016, `plans/udxf-source-api-backend.md`)
exposed Mixpanel resources as `flight_udxf` relations: a cloudpickled
exchanger carrying an HTTP client whose fields are env var references. It
worked and satisfied the credential invariant, but running it end to end
surfaced three costs:

1. **Opacity.** The client rides inside base64-encoded pickle bytes in
   `expr.yaml`. The credential invariant could only be enforced *before*
   serialization (construction-time checks), never audited after — a leaked
   raw value would be invisible to grep.
2. **No profile participation.** The udxf relation registers into the default
   backend, so the mixpanel backend's profile never reached the build's
   `profiles.yaml`; the source of the data was not a first-class, rehydratable
   build participant (`hydrate_cons` never saw it).
3. **Hash fragility.** The udxf command is `tokenize(process_df)` — pickled
   closure bytes — so innocuous refactors of fetch code change expression
   identity.

Meanwhile the `Read` op (ADR-0006) already models declarative deferred
ingestion — `method_name` + `read_kwargs` + a by-name-registered
`normalize_method` (see `normalize_registry.py`) — but every consumer of
`Read` identity assumed a file path: dasher tokenization
(`_relations.py`, `_opaque.py`), snapshot cache keys
(`caching/strategy.py`), and the build dumper's normalize-method override
(`compiler.py::_sanitize_generated_names`) all reached for
`read_kwargs["hash_path"]`.

## Decision drivers

- Build artifacts must be auditable, not just constructed carefully
  (ADR-0016's invariant, enforced *and* greppable).
- The data source's profile must rehydrate from `profiles.yaml` like any SQL
  backend's.
- Expression identity must be declarative — stable across sessions, fetch-code
  refactors, and credential rotation — and must distinguish source config
  (different project → different data).
- Existing path-based reads must tokenize byte-identically (ADR-0015: hash
  changes are correctness-relevant).

## Decision

**An API-backed relation is a `Read` op with no `hash_path`. A path-less
Read's identity comes entirely from its registered `normalize_method`, which
receives the op itself (not a path) and returns declarative identity: the
source profile's content hash (idx excluded), the `method_name`, and the
read kwargs minus the unstable `table_name`.**

### The registered normalizer

`normalize_read_source_identity` (`common/utils/file_utils.py`), registered
as `read_source_identity` in the append-only `_NORMALIZE_RULES`. Profile
identity uses the content hash, not `hash_name`, because the idx suffix is
session-global (ADR-0002).

### Path-less branches at every Read-identity consumer

Each site that assumed `hash_path` gets an explicit path-less branch that
delegates to `read.normalize_method(read)`:

- `dasher/_relations.py::_normalize_read_xorq` (build/cache tokenization);
  raises if a path-less read has no normalize_method.
- `dasher/_opaque.py` Read case (structural SQL placeholder names).
- `caching/strategy.py::snapshot_normalize_read` (snapshot cache keys — the
  source-identity result is already stat-free, which is what snapshot wants).
- `ibis_yaml/compiler.py::_sanitize_generated_names` keeps a path-less read's
  own normalize_method instead of overriding with the dumper-wide *path*
  normalizer (mirroring the existing relocatable-read exemption).

Reads with `hash_path` take exactly the code paths they took before; the
branches key on key-absence, which previously crashed (`KeyError`), so no
existing build or cache hash changes.

### Execution: `fetch_*` methods on a served backend

`Read.make_dt` calls `getattr(source, method_name)(**kwargs)` at the
execution boundary. The mixpanel backend now subclasses the pandas backend:
`fetch_events`/`fetch_engage` perform the HTTP calls (credentials resolved
from env at that moment), land the DataFrame in `self.dictionary`, and return
a served table. `read_events`/`read_engage` construct the deferred `Read`
(and still reject raw secrets at construction, per ADR-0016 — a profile with
raw values would otherwise be serialized plaintext into `profiles.yaml`).

## Alternatives considered

### Cloudpickled `flight_udxf` relations (Phase 1 implementation)

Ran live end to end (350-row export, multi-page engage, leak-grep-clean
builds). Superseded because of the three costs in Context: unauditable
pickle payloads, no `profiles.yaml` participation, pickle-bytes hashing.
Remains the right seam for *user-defined* transforms (that is what UDXFs are
for) and in the userland template.

### Synthesize a fake `hash_path` for API reads

Rejected because:
- Every consumer treats `hash_path` as a filesystem path (stat, md5sum,
  existence checks, relocation candidacy); a sentinel value would need
  guards at the same set of sites anyway, without saying what it means.

### A new op (`ApiRead`) instead of extending `Read`

Rejected because:
- `Read` already carries exactly the right fields and the yaml
  translator/`profiles.yaml` integration; a parallel op would duplicate the
  serialization, relocation-exemption, and cache plumbing for no expressive
  gain.

## Consequences

### Positive

- `expr.yaml` for an API read is fully declarative — human-readable, no
  pickle — and the mixpanel profile lands in `profiles.yaml` with env refs,
  rehydrating via `hydrate_cons` like any SQL backend.
- Build hashes are reproducible across sessions (verified: two separate
  processes produced identical build hash `d25dcc5e30a0`) and stable under
  fetch-code refactors.
- Snapshot caching works with declarative identity (no stat calls).

### Negative

- The fetched result is served from the pandas-based backend's memory; very
  large API reads inherit pandas memory behavior (mitigate with `.cache()`
  to parquet, or param-partitioned reads).
- `normalize_method` for path-less reads is load-bearing at three dasher/cache
  sites; a future consumer of Read identity must remember the path-less
  branch (the KeyError-on-absence failure mode makes forgetting loud, not
  silent).
- Identity includes the profile *content* hash: connections differing only in
  env-ref names (not values) hash differently even when the refs resolve to
  the same account — declarative identity, consistent with profile semantics
  elsewhere.

## References

- ADR-0002 (sequential id normalization), ADR-0006 (read-kwargs hash-path
  split), ADR-0015 (hash participation rule), ADR-0016 (credential-free
  build artifacts)
- plans/udxf-source-api-backend.md — Phase 2
- xorq-labs/xorq#2182 — Phase 1 PR (cloudpickled-udxf implementation this
  supersedes in core)
