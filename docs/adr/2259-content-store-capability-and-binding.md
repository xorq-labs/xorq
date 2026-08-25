# ADR-2259: Content-store capabilities and hosted bindings

- **Status:** Proposed
- **Date:** 2026-08-07
- **Deciders:** TBD — proposed for review
- **Related:** ADR-0011 (catalog supports a single git remote)

## Context

The hosted (`presigned`) content store landed alongside the existing
`directory` and `s3` stores. Reviewing it surfaced sixteen findings, and
sorting them by cause rather than by symptom produced two clusters that
account for most of them.

### Cluster A — the `ContentStore` ABC models one store family

`ContentStore` was designed around "a key→bytes map the client administers".
The hosted store is a different thing: a mediated, spec-addressed transfer
channel whose blob lifecycle belongs to the server. The mismatch shows up as:

- Three of five abstract methods (`exists`, `delete`, `list_keys`) raise
  `ContentStoreCapabilityError` for the hosted store, and `get` raises unless
  given arguments the base signature does not have.
- A `client_managed_lifecycle` class boolean exists purely to let callers ask
  "which family is this?", plus a `GitPointerBackend._client_managed_lifecycle`
  property that re-derives the same answer.
- `gc_content_store` checks *both* an `isinstance` against the hosted config
  *and* the boolean.
- `backend.py` carries **seven** `isinstance(self._config,
  PresignedContentStoreConfig)` checks.

Every one of those is the same fact — "this store is not administrable" —
re-expressed at a different call site. A fourth store type multiplies them
again.

### Cluster B — validation is re-derived instead of carried

Remote-binding validation returns `None` and raises on failure. It produces no
value, so nothing downstream can tell whether it already ran. Consequences:

- `xorq catalog init` validates the same remote URL **three** times: once in
  the CLI, once in `Catalog.set_remote`, once when materialising the store.
- `GitPointerBackend.content_store` costs **three** git subprocesses per access
  (`Remote.urls`, then `git remote get-url` twice for fetch and push).
  `fetch_content` now touches it once per batch.
- `Catalog._validated_git_remotes` re-validates on every property access, so
  what reads as a cheap accessor spawns two subprocesses.
- `CatalogBackend` grew five validation hooks, three of them empty defaults
  needing `# noqa: B027`.
- Two separate sites wrap failures with `hosted_remote_error`.

Three independent parses of one string is not only waste — it is three chances
for the checks to drift apart.

## Decision

### S1 — split the store interface by capability, not by flag

```python
class ContentStore(abc.ABC):
    """Spec-addressed transfer. All the catalog backend needs."""

    @abc.abstractmethod
    def ensure_present_many(
        self, objects: Iterable[tuple[ContentSpec, Path]]
    ) -> set[str]: ...

    @abc.abstractmethod
    def get_many(
        self, objects: Iterable[tuple[ContentSpec, Path]]
    ) -> set[str] | None: ...


class ManagedContentStore(ContentStore):
    """A store whose blob lifecycle the client administers."""

    @abc.abstractmethod
    def exists(self, key: str) -> bool: ...

    @abc.abstractmethod
    def delete(self, key: str) -> bool: ...

    @abc.abstractmethod
    def list_keys(self, prefix: str = "") -> Iterator[str]: ...
```

`DirectoryContentStore` and `S3ContentStore` become `ManagedContentStore`.
`PresignedContentStore` stays a plain `ContentStore`.

Deletes outright: `client_managed_lifecycle`, `_client_managed_lifecycle`,
`_unsupported`, and the five hosted stub methods.

`gc_content_store` reduces to one check against a *capability* rather than a
vendor:

```python
store = self.content_store
if not isinstance(store, ManagedContentStore):
    raise ContentStoreCapabilityError(
        "blob garbage collection needs a client-administered content store; "
        f"{type(store).__name__} delegates blob lifecycle to its service"
    )
```

`stage_unlink`'s reference-counted delete follows the same shape.

### S2 — make the validated binding a value

```python
@frozen
class HostedBinding:
    """Proof that a Git remote is bound to this catalog's hosted service.

    Cannot be constructed invalid: `parse` is the only validation path, and
    holding an instance *is* the evidence that validation passed.
    """

    service_url: str
    catalog_id: str
    remote_url: str

    @classmethod
    def parse(
        cls, config: PresignedContentStoreConfig, remote_url: str
    ) -> HostedBinding:
        _validate_remote_binding(config.service_url, remote_url)
        return cls(config.service_url, config.catalog_id, remote_url)

    def still_binds(self, current_url: str) -> bool:
        return current_url == self.remote_url
```

`PresignedContentStore` takes a `HostedBinding` instead of three loose strings,
so an unvalidated store is unconstructible. The CLI parses once and passes the
binding down. Revalidation on later operations degrades from *parse + three
subprocesses* to *one `git remote get-url` + a string compare*.

This collapses `validate_remote_url`, `validate_remote`, `bound_remote_url`,
and `preflight_content_write` toward a single `rebind()` on the backend.

## Consequences

Findings retired by S1: the capability-error conflation, the dead key-only
`get` path, seven isinstance checks, two lifecycle booleans.

Findings retired by S2: triple validation on `init`, per-property-access
revalidation, batch-multiplied revalidation, split error-annotation sites.

Not addressed by either, and tracked separately: pointer-read TOCTOU, absence
of upload retry/multipart at the 5 GB ceiling, the `upload_id` protocol
coupling for already-verified objects, and the `_token` format check that
raises when the caller asked for an optional token.

Migration is mechanical and can land in two independent PRs; S1 does not
depend on S2. Third-party `ContentStore` implementations that only move bytes
keep working — `get_many` returning `None` means "verified nothing", the safe
default.

## Invariants

These are the properties to enforce so the clusters cannot re-form:

1. The catalog backend addresses content only by `ContentSpec`, never by bare
   key. Adding a mediated store requires no backend change.
2. `backend.py` contains zero `isinstance` checks against config classes.
   Mechanically checkable; worth a `xorq-check-style` rule.
3. Every byte on the cache volume is a named cache entry or lives in the
   staging directory, where it is counted against the budget and reaped by age.
4. Bytes are hashed once, at the boundary they cross. A store that hashed in
   transit says so; nobody re-reads to be sure.
5. A validated binding is a value. Validation happens exactly at construction.
6. Constants that must agree are derived, not co-declared, and tests assert the
   relationship rather than the number.
7. A store's capabilities are its type, not a boolean.
