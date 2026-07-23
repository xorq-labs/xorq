# ADR-0016: Build artifacts are credential-free; Profiles are the sole credential carrier

- **Status:** Accepted
- **Date:** 2026-07-23
- **Deciders:** Dan Lovell

## Context

A xorq expression that reads from an authenticated source (postgres, snowflake,
and now REST APIs like Mixpanel) must be serializable into a build artifact
that is shareable, cacheable, and cataloged by content hash. Credentials create
two hazards there:

1. **Leakage.** Anything a build captures — `profiles.yaml`, SQL source YAML,
   cloudpickled callables inside a `FlightUDXF` — is distributed with the
   artifact. A resolved password embedded anywhere in that payload is a secret
   at rest in every copy of the build.
2. **Hash instability.** If credential *values* participate in expression
   identity, rotating a password changes build and cache hashes for pipelines
   whose semantics did not change.

The machinery to avoid both already exists but the rule was never stated
(compare ADR-0015, which stated the previously implicit hashing rule):

- `BaseBackend.__init__` substitutes `${VAR}` references for execution while
  `Profile.from_con` preserves the *unsubstituted* references
  (`vendor/ibis/backends/__init__.py:866-875`).
- Builds serialize connections as profiles and rehydrate them via
  `Profile.get_con` (`ibis_yaml/compiler.py` `dehydrate_cons`/`hydrate_cons`).
- `check_for_exposed_secrets` rejects saving a profile whose secret values are
  not env var references (`vendor/ibis/backends/profiles.py`).

What was missing: the secret-key list was a hardcoded dict covering only
postgres and snowflake, and nothing prevented a callable captured in an
expression (e.g. a `flight_udxf` `process_df` fetching an API) from closing
over resolved credentials — the enforcement point existed only at
`Profile.save`.

## Decision drivers

- A secret embedded in a build artifact is a silent security defect; it cannot
  be revoked by deleting the source expression.
- Credential rotation must not invalidate build or cache identity.
- New backends (REST APIs among them) must get the same guarantees without
  editing vendored profile code.
- Interactive use (raw credentials in a REPL) should still work for
  execution-only paths.

## Decision

**Nothing serialized — YAML, cloudpickle payloads, cache keys, build zips —
may contain credential values. Serialized artifacts carry credential
*identity* only: a profile `hash_name` and/or unresolved env var references.
Values resolve from the executing machine's environment at execution time.**

Three mechanisms implement this:

### Backends declare their own secret keys

`Backend._secret_keys` (a tuple of kwarg names) replaces the hardcoded
allowlist. `check_for_exposed_secrets` consults the declared keys via
`get_declared_secret_keys` (entry-point load), falling back to the legacy
`con_name_to_secret_keys` dict when the backend module is not importable
(e.g. optional extras absent), then to `("password",)`. postgres and snowflake
now declare their keys; the dict remains as fallback only.

### Env var references are the wire format for secrets

Connections are made with `secret="${MIXPANEL_SERVICE_ACCOUNT_SECRET}"`-style
references. The profile keeps the reference; `do_connect` receives the
substituted value; anything intended for serialization is built **from the
profile**, never from the live connection's resolved state (see
`xorq.backends.mixpanel.Backend._expr_client` / `MixpanelClient`, whose fields
hold references and resolve per request via `maybe_substitute_env_var`).

### Enforcement at every serialization doorway

- `Profile.save` rejects raw secret values (existing behavior, now driven by
  declared keys).
- Expression construction that captures a credential-bearing callable rejects
  raw secret values too: `Backend._expr_client` calls
  `check_for_exposed_secrets` before the client is closed over, because a
  cloudpickled closure inside `expr.yaml` is just as distributed as
  `profiles.yaml` — base64-encoded pickle bytes are not greppable, so this
  leak class must be prevented, not audited.

## Alternatives considered

### Encrypt secrets into build artifacts

Ship credentials encrypted in the build, decrypt at run time with a key.

Rejected because:
- It converts a no-secret design into a key-management problem and makes every
  artifact copy a target.
- Hash stability would still break on rotation.

### Central secret-manager integration (vault, keyring)

Resolve named secrets from a manager instead of env vars.

Deferred because:
- Env var references already compose with every secret manager (they all can
  export to env), and the reference syntax leaves room for other schemes later.

### Keep the hardcoded per-backend dict

Extend `con_name_to_secret_keys` for each new backend.

Rejected because:
- The knowledge belongs to the backend, not to vendored profile code; the dict
  demonstrably lagged (only 2 of 13 backends covered) and cannot cover
  out-of-tree backends installed via entry points.

## Consequences

### Positive

- Builds of authenticated sources are shareable and cataloged without secret
  hygiene review; credential rotation never invalidates hashes.
- New backends opt in by declaring one tuple; out-of-tree backends get the
  same enforcement.
- The mixpanel backend ships as the reference implementation: profile-carried
  auth, env-ref-only expressions, verified empty leak-grep of built artifacts.

### Negative

- Executing a rehydrated build requires the env vars to be present on the
  executing machine; a missing var fails at execution (KeyError), not at load.
- Raw-credential connections cannot build serializable expressions (by
  design); users must move credentials into env vars to build.
- `get_declared_secret_keys` imports the backend module at save time; for
  heavy backends this adds latency to `Profile.save` (bounded by the existing
  fallback for unimportable modules).

## References

- ADR-0006 (read-kwargs hash-path/read-path split), ADR-0010 (normalize op
  data vs structure), ADR-0015 (every op modifies the build hash)
- plans/udxf-source-api-backend.md (API-as-Backend design and phasing)
- xorq-labs/xorq-template-mixpanel-fetcher (Phase 0: fetcher-in-userland)
