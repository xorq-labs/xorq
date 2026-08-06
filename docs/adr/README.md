# Architecture Decision Records

Every ADR in this directory, with its status. Start a new one from
[`template.md`](template.md).

Nothing here is published: `docs/_quarto.yml` excludes `adr/**` from the built
site, so these documents are a repo-internal record and this file is the only
listing of them.

## Index

| # | Title | Status |
|---|-------|--------|
| [0002](0002-normalize-sequential-ids-in-build.md) | Normalize sequential IDs during expression builds for deterministic hashing | Accepted |
| [0003](0003-optional-git-annex-backend.md) | Make git-annex optional via a CatalogBackend abstraction | Accepted |
| [0004](0004-uv-as-sole-packaging-and-execution-runtime.md) | uv as the sole packaging and execution runtime for the sdist pipeline | Superseded by 0008 |
| [0005](0005-expr-builder-from-tagged-registry.md) | ExprBuilder — registry-driven domain object recovery from tagged expressions | Accepted |
| [0006](0006-read-kwargs-hash-path-read-path-split.md) | Split `read_kwargs` path into `hash_path` (identity) and `read_path` (location) | Accepted |
| [0007](0007-datafusion-plan-path-canonicalization.md) | Canonicalize catalog-extract tempdir in DataFusion DT tokens | Accepted |
| [0008](0008-wheel-based-packaging-pipeline.md) | Wheel-based uv packaging and execution pipeline | Accepted (supersedes 0004) |
| [0009](0009-bucket-fileprefix-name-uuid-namespace.md) | Namespace S3 bucket layout by remote name and UUID | Accepted |
| [0010](0010-split-normalize-op-data-from-structure.md) | Split data-dependent tokens out of expression normalization | Amended |
| [0011](0011-catalog-single-git-remote.md) | Catalog supports a single git remote | Accepted |
| [0012](0012-click-option-decorators-and-runner-hierarchy.md) | Shared Click option decorators and runner class hierarchy for CLI parity | Accepted |
| [0013](0013-batchcorder-stream-cache-for-remote-table-fan-out.md) | Replace SafeTee with batchcorder StreamCache for RemoteTable fan-out | Accepted |
| [0014](0014-teenode-deferred-writes.md) | TeeNode, deferred write as a side effect | Amended |
| [0015](0015-build-hash-cache-hash-split.md) | Every op modifies the build hash; cache-hash neutrality is the exception | Accepted |
| [0016](0016-table-driven-opaque-descent-with-registration-tripwires.md) | Table-driven opaque descent with registration tripwires | Accepted |
| [0017](0017-canonical-hash-forms-not-serializer-bytes.md) | Hash identity comes from xorq-owned canonical forms, never dependency-serializer bytes | Proposed |
| 0018 | REST APIs are declarative configs behind one backend; identity is always folded, config residence is a per-API packaging choice | **Reserved — lands in a later stack entry** |
| 0019 | REST resource reads register as lazy DataFusion tables on an owned connection | **Reserved — lands in a later stack entry** |
| [0020](0020-engine-behavior-as-immutable-identity-folded-spec.md) | Engine behavior is an immutable, identity-folded declarative spec | Accepted |
| [0021](0021-engine-construction-is-two-level-identityspec-feeds-enginebuilder.md) | Engine construction is two-level — a cross-engine IdentitySpec feeds an EngineBuilder; per-engine specs feed `build()` | **Proposed — unimplemented** |
| 0022 | Out-of-core patches compose — delegate, conjoin, or fork behind a source tripwire | **Reserved — lands in a later stack entry** |
| [0023](0023-identity-spec-contributions-are-entry-points-composed-order-independently.md) | Identity-spec contributions are entry points, composed order-independently into DEFAULT_BUILDER | **Proposed — unimplemented** |
| 0024 | Build artifacts are credential-free; Profiles are the sole credential carrier | **Reserved — lands in a later stack entry** |
| 0025 | API-backed relations are path-less Read ops; their identity is the registered source-identity normalizer | **Reserved — lands in a later stack entry** |

Five rows above are not files in this directory yet, deliberately:

- **0018, 0019, 0024 and 0025 land in stack entry 4**, and **0022 in entry 5**
  (which must merge after 4). All five are real, written ADRs on branches that
  exist; none of them is speculative. They are listed unlinked rather than
  omitted so the numbers read as taken: a gap in this table is an invitation
  for the next ADR to claim one of them and collide, which is exactly the
  failure this index exists to prevent (see below). Each row gets a link when
  its file lands. Nothing here links to a file that does not exist at this
  head.
- **0021 and 0023 are Proposed and unimplemented.** Their statuses are not
  bookkeeping lag — none of the machinery they describe (`IdentitySpec`,
  `EngineBuilder`, `DEFAULT_BUILDER`, the `xorq.identity_specs` entry-point
  group) exists in the tree. Each document opens by saying so, says why it is
  landing ahead of its implementation anyway, and names the gate that would
  flip it to Accepted. Those gates have an owner:
  [#2202](https://github.com/xorq-labs/xorq/issues/2202) implements ADR-0021
  phases 1–2, which is also what unblocks ADR-0023. ADR-0017 is Proposed for
  the ordinary reason: the decision is not yet implemented.

There is no ADR-0001; numbering starts at 0002.

## The numbering convention

**A number records claim order, not landing order.** You take the next free
number when you *write* the ADR. Nothing renumbers it afterwards, so the
sequence does not tell you what merged first, and a low number does not imply
an older decision — 0016 is dated after 0021.

The cost of that convention is collisions: a long-lived branch and `main` both
take "the next number" from the same starting point, and both are right until
they meet. That has already happened here. This stack drafted its own 0016 and
0017 while `main` independently claimed 0016 and 0017 after the merge-base —
different filenames, so git merges both sides without conflict and `main` would
have ended up with two 0016s and two 0017s. Nothing reported it; it was caught
by hand during landing, and this stack's pair was renumbered to 0024 and 0025,
which cost fifteen citation updates including two code sites. That renumber is
why the identity thread reads out of order.

The current split: `main` holds 0002–0017; this stack claims 0018–0025.

So: before claiming a number, check this table *and* the ADR directory on any
branch you know is in flight — a distinct filename is not protection, because
distinct filenames are exactly what lets the collision merge cleanly. If you
are drafting on a branch that will be open for a while, say in the ADR which
number you claimed and when, so a collision is visible in a diff.

## Suggested reading order: the identity and caching thread

Most of these ADRs are one argument about content-addressed identity — what a
hash is allowed to depend on. Read in this order rather than by number. Numbers
without links are the reserved rows above; they read in sequence once the rest
of the stack lands.

**The grain.** [0015](0015-build-hash-cache-hash-split.md) first: the build
hash answers "was this pipeline built?", the cache hash answers "is this result
reusable?", and the two are allowed to move independently. Almost everything
later is stated in its vocabulary.
[0010](0010-split-normalize-op-data-from-structure.md) and
[0006](0006-read-kwargs-hash-path-read-path-split.md) are the same move applied
inside normalization and inside read kwargs: separate what identifies data from
what merely locates or carries it.
[0002](0002-normalize-sequential-ids-in-build.md) and
[0007](0007-datafusion-plan-path-canonicalization.md) are two early instances
of the same discipline.

**What a hash may depend on.** [0017](0017-canonical-hash-forms-not-serializer-bytes.md)
(Proposed) draws the outer line: identity comes from xorq-owned canonical
forms, never from bytes a dependency's serializer happened to emit.
[0016](0016-table-driven-opaque-descent-with-registration-tripwires.md) is how
descent into unknown objects stays honest without a hand-maintained list.

**Sources with no path.** ADR-0025: an API-backed relation has no file path to
hash, so its identity is a registered normalizer. ADR-0024 is the constraint
that shapes it — artifacts carry env-var *references*, never credential values,
so identity must be built from things that are safe to write down.

**REST as the worked example.** ADR-0018 turns an API into a declarative config
behind one backend, with identity folded from the config itself; ADR-0019
replaces the eager pandas substrate under it with lazy DataFusion tables, and
leans on [0013](0013-batchcorder-stream-cache-for-remote-table-fan-out.md)'s
StreamCache to make a one-shot reader survive a multi-scan plan.

**Turning the machinery itself into identity.**
[0020](0020-engine-behavior-as-immutable-identity-folded-spec.md) closes the
loop: the *rule set* that computes hashes is itself identity-bearing, so its
fingerprint folds into the build hash. From there the thread becomes design
work — [0021](0021-engine-construction-is-two-level-identityspec-feeds-enginebuilder.md)
makes engine construction two-level so identity rules cannot vary per engine,
and [0023](0023-identity-spec-contributions-are-entry-points-composed-order-independently.md)
gives plugins an order-independent way to extend them. Read both as proposals;
neither is implemented, and each says so in its own opening. ADR-0022
(reserved, above) is the stop-gap discipline that bounds patches which cannot
yet be expressed that way.

## Follow-up: a docs-lint for this index

Keeping this table and the ADRs' cross-references honest wants a lint — check
every `ADR-NNNN` citation resolves to a file, and that every file appears here
with a matching status. That is a post-landing follow-up, not part of this
stack, because a citation lint would fail on every branch in it. Tracked as
[#2203](https://github.com/xorq-labs/xorq/issues/2203).

This is a property of a stack that adds ADRs which reference each other, not a
defect. Cross-references are written against the ADR set as it will exist once
the whole stack lands, so intermediate heads are legitimately incomplete. At
this head, six citations do not resolve:

| citing ADR | target | lines |
|---|---|---|
| 0020 | 0018 | 12, 159, 174 |
| 0020 | 0019 | 15, 174 |
| 0021 | 0018 | 94, 147, 218 |
| 0021 | 0019 | 224 |
| 0023 | 0018 | 86, 229 |
| 0023 | 0022 | 50, 108, 152, 190, 228 |

Five of the six resolve when entry 4 lands 0018 and 0019; the last (0023→0022)
resolves only at entry 5. The set becomes self-consistent after all five
entries merge — which is the moment such a lint can start passing, and
therefore the moment to add it.
