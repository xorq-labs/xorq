# ADR-2213: Build artifacts are credential-free; Profiles are the sole credential carrier

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

`check_for_exposed_secrets` checks the **union** of three tiers
(`get_secret_keys`), not a fallback chain:

1. `default_secret_keys` — `("password",)`, unconditional;
2. the static keys: the `con_name_to_secret_keys` mirror, keyed by connection
   name, topped up from an already-imported backend's `_secret_keys`;
3. the backend's declared `_secret_key_sources` — static class data naming
   *where in the connection kwargs* the secret-key names live — resolved
   against the kwargs by a pure data walk.

Unioning is what makes the gate monotone: an empty, narrower, raising or
unresolvable tier leaves the others intact, so a tier can only ever *widen*
what is checked. A precedence chain cannot promise that -- the tier that
answers first decides, and the one that knows least can silence the rest. That
is fail-open in a security gate, and it is why this is a union.

Tier 2's mirror is the import-free floor, and the only input that answers in a
process which never imported the backend -- validating a saved profile, a CLI
audit. The class reads are the widening step for an out-of-tree backend, which
can never appear in the in-tree mirrors: its static `_secret_keys` covers the
fixed names, its `_secret_key_sources` covers names stored in the kwargs data.

No backend-authored code executes inside the gate. Every declaration is
static class data, read with `inspect.getattr_static` (a descriptor cannot
fire) and resolved by type checks and unbound-builtin reads; the names come
back as exact `str` copies. A callable hook was tried and replaced (#2184):
plugin code running inside a security check, handed the exact data the check
protects, needed more lines of defensive guards than feature, with one leak
shape unclosable by construction.

For an in-tree backend, `Backend._secret_keys` is pinned against the tier-2
mirror by tests in both directions, so the declaration beside the backend and
the mirror cannot drift.

### Env var references are the wire format for secrets

Connections are made with `secret="${MIXPANEL_SERVICE_ACCOUNT_SECRET}"`-style
references. The profile keeps the reference; `do_connect` receives the
substituted value; anything intended for serialization is built **from the
profile**, never from the live connection's resolved state. The supported
surface for that is `BaseBackend.expr_safe_profile_kwargs()`: it returns the
profile's kwargs (references preserved) after rejecting raw secrets, so a
backend -- in-tree or plugin -- never reaches into the profile internals to
build a capture-safe client (see `xorq.tests.fixture_backend`, whose client
fields hold references and resolve per request via
`maybe_substitute_env_var`).

### Enforcement at every serialization doorway

- `Profile.save` rejects raw secret values (existing behavior, now driven by
  declared keys).
- Expression construction that captures a credential-bearing callable rejects
  raw secret values too: `expr_safe_profile_kwargs` runs
  `check_for_exposed_secrets` before handing out the kwargs a client is built
  from and closed over, because a cloudpickled closure inside `expr.yaml` is
  just as distributed as `profiles.yaml` — base64-encoded pickle bytes are not
  greppable, so this leak class must be prevented, not audited.
- Prevention is nonetheless pinned by a test that *audits*: it builds an
  expression with fake credentials, decodes every base64 payload in the
  artifact, and asserts the resolved values are absent and the env-var
  references present. Prevention that no test can fail is indistinguishable
  from luck — handing the deferred read the resolved client instead of the
  reference-holding one leaks every credential into every artifact built from
  it, and passes every other test in the suite.
- Credentials that *are* resolved in memory — the client a live connection
  holds — are declared with `secret_field`, so `repr` and everything built on
  it (log lines, tracebacks, debugger frames, attrs validator errors) cannot
  print the plaintext. That covers the process-local surface the serialization
  rules say nothing about, and the two mechanisms are cross-checked: every
  profile-enforced key must also be repr-suppressed.

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
- An in-tree backend opts in with a mirror entry plus the matching
  `_secret_keys` declaration the drift tests require; an out-of-tree backend,
  which cannot be mirrored, opts in by declaring static `_secret_keys` for its
  fixed names (read from the imported class) and `_secret_key_sources` when
  the names live in the kwargs data.
- The plugin contract is pinned in-tree by `xorq.tests.fixture_backend`, an
  API-shaped backend installed as an entry point mid-test: profile-carried
  auth, class-declared static secret keys with no mirror entry, env-ref-only
  expressions, verified empty leak-grep of built artifacts. The reference
  *integration* is out-of-tree: `xorq-labs/xorq-mixpanel` consumes exactly
  that contract, which keeps vendor connectors -- an unbounded population
  whose churn follows vendor APIs, not xorq releases -- out of `backends/`
  and off the static mirror the alternatives section below rejects.

### Negative

- Executing a rehydrated build requires the env vars to be present on the
  executing machine; a missing var fails at execution (KeyError), not at load.
- Raw-credential connections cannot build serializable expressions (by
  design); users must move credentials into env vars to build.
- The class reads inspect only an already-imported backend, so a process that
  never imported it is checked by the default and the mirrors alone. For an
  in-tree backend the mirrors cover that; an out-of-tree backend has no mirror
  entry, so its declarations are silent until something imports it. Resolution
  deliberately does not import the backend itself: importing to read a
  declaration runs the plugin's module body, and a plugin that mutates
  identity registries at import would move build hashes as a side effect of
  saving a profile.
- A raw value that begins with `$` reads as an env-var reference
  (`compiled_env_var_substitution_re`) and passes this gate, then fails at
  execution when no such variable exists -- after the value is at rest in
  `profiles.yaml`. Pre-existing for every backend's `password` and not fixed
  here; it bounds the invariant's claim to values that do not look like
  references.

## References

- ADR-0006 (read-kwargs hash-path/read-path split), ADR-0010 (normalize op
  data vs structure), ADR-0015 (every op modifies the build hash)
- the udxf-source API-as-Backend design plan (an untracked working document
  outside this repository)
- xorq-labs/xorq-template-mixpanel-fetcher (Phase 0: fetcher-in-userland)
- xorq-labs/xorq-mixpanel (the out-of-tree reference integration)
- xorq.tests.fixture_backend / tests/test_build_artifacts_credential_free.py
  (the in-tree pin of the plugin contract)
