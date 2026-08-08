# ADR-XXXX: Derive ADR numbers from pull request numbers

- **Status:** Proposed
- **Date:** 2026-08-08
- **Deciders:** dlovell

## Context

ADR numbers were allocated by the author: pick the next number after the highest one in `docs/adr/`. That read of "highest existing" is taken against the author's branch, not against the state of every other open branch, so two concurrent pull requests routinely pick the same number.

Git makes this worse than an ordinary conflict. Two branches that each add `0019-<different-slug>.md` touch disjoint paths, so a merge produces no conflict and no warning — both files land, both claim number 0019, and nobody notices until someone greps for it. The failure is silent by construction, which is why it keeps recurring rather than getting fixed once.

The repository shows three instances:

- ADR-0006 and ADR-0007 landed out of order. `0007-datafusion-plan-path-canonicalization.md` merged in #1832 before `0006-read-kwargs-hash-path-read-path-split.md` merged in #1838, so the lower number describes the later decision, and ADR-0007's "Related: ADR-0006" pointed at a file that did not yet exist on `main`.
- `0001-git-annex-over-git-lfs.md` exists only on an unmerged branch. It claimed 0001 while 0002 through 0017 shipped past it.
- `0018-content-store-capability-and-binding.md` is in flight on another unmerged branch. Any third branch is free to claim 0018 today.

A fourth pattern is not itself a collision but is what produces them. A stacked series of pull requests routinely needs an early entry to cite a decision recorded in a later one. When a number is the only way to cite an ADR, the author has to allocate the whole range before any of it reaches a forge, and then publish the reservation so nobody else takes it. That is how the largest block of contended numbers in this repository came to exist: eight numbers reserved by one stack, two of which a branch elsewhere had already claimed.

A constraint shapes the fix: roughly sixty comments across `python/xorq` cite ADRs by number — `_paths.py`, `write_through.py`, `graph_utils.py`, `_canonical.py`, `catalog/*`, and many test files. Renumbering existing ADRs would either break those citations or require a sweep across production code to fix a documentation problem.

## Decision drivers

- No coordination between concurrent branches, and no shared file to serialize on.
- Collisions must be impossible, or failing that, detected by a machine rather than a reader.
- Existing numbers must not change, because code comments cite them.
- Must not add a human allocator to the path of writing an ADR.
- The identifier stays short enough to cite in a comment.
- An ADR must be citable before it has a number, so a stack can reference its own later entries without reserving a range.

## Decision

An ADR's number is the number of the pull request that adds it.

| Number | Meaning |
|--------|---------|
| Below 1000 | Legacy sequential number, frozen permanently |
| 1000 and above | The number of the pull request that added the ADR |

GitHub allocates pull request numbers from a single counter, so the number is unique, monotonic, and already assigned by the time anyone could collide over it. There is nothing to reserve and nothing to check.

The split at 1000 needs no registry to interpret. Legacy ADRs stop at 0017 and pull request numbers were already past 2200 when this landed, so the ranges cannot meet. Legacy ADRs keep their numbers and their citations keep resolving.

### Why the number arrives late

The number does not exist until the pull request does, so an ADR is authored as `XXXX-<slug>.md` and renamed once the pull request is open:

1. `just adr-new <slug>` copies the template to `docs/adr/XXXX-<slug>.md`.
2. Write it, then open the pull request.
3. `just adr-rename` reads the number from `gh pr view`, renames the file, and rewrites the `# ADR-XXXX:` heading.

`XXXX` rather than Rust's `0000` because a placeholder should not look like a number: `0000` parses as an integer, sorts as one, and a half-finished rename leaves something that reads as a real ADR. `XXXX` cannot be mistaken for a number or accidentally cited, and CI can distinguish "not renamed yet" from "wrong number".

### Referring to an ADR that has no number yet

A number that does not exist cannot be cited. That is the whole difficulty for a stacked series of pull requests, where an early entry needs to reference a decision recorded in a later one. If a number is the only citation form, the author must allocate ahead of the forge and publish the reservation — which is precisely the collision mode this decision exists to remove.

So an ADR has two citation forms, and both resolve permanently:

| Form | Example | Resolves against |
|------|---------|------------------|
| Numbered | `ADR-0011` | the number, for an ADR that has one |
| Named | `ADR-catalog-single-git-remote` | the slug, whether or not the ADR is numbered yet |

The two are lexically disjoint — a number begins with a digit, a slug with a letter — so nothing is needed to tell them apart.

The slug is the ADR's identity and is fixed when the file is created; the number is an alias that arrives later. `just adr-rename` rewrites named references to the ADR it numbers, so landed prose settles on the short numeric form. Because the named form never stops resolving, a sweep that is partial or lags behind cannot break the build — which is what makes it safe to run across branches.

A named reference to an ADR not yet in the tree — the forward reference — is reported as a warning rather than an error, since the target may legitimately live on an unlanded branch. A dangling *numeric* reference stays an error. The asymmetry is the point: a bare `ADR-NNNN` pointing at an unlanded decision tells a reader nothing and CI nothing until it lands, while `ADR-rest-apis-are-declarative-configs` is legible immediately and resolves by itself the moment its file appears.

This works because ADRs here already ride along with their implementing pull request — #2196 carried ADR-0016, #2192 carried ADR-0017, #1899 carried ADR-0011. The pull request number is therefore not an arbitrary identifier but a direct pointer to the code that implemented the decision and the discussion that shaped it.

### Enforcement

`scripts/adr_check.py`, run by `.github/workflows/ci-adr.yml`, fails a pull request that leaves a placeholder in a filename, adds an ADR whose number is not the pull request number, adds a second ADR to the same pull request, duplicates an existing number or slug, disagrees between filename and heading, or makes a numeric reference to an ADR that does not exist. Unresolved *named* references are listed as warnings and do not fail the run.

The script is stdlib-only and needs no `uv sync`, so the workflow runs the runner's `python3` directly. It carries no `paths:` filter: a path-filtered workflow reports no status at all on pull requests that miss the filter, which makes it unusable as a required check. The script is a sub-second no-op instead.

Two of the checks are retroactive. Duplicate detection and reference resolution run over the whole directory on every pull request, so they also cover the legacy range — and reference resolution closes a real gap, since ADRs are excluded from the rendered site and the existing `lychee` link check only walks `docs/_site`.

A short allowlist in the script exempts the legacy-numbered ADRs that were in flight when this landed, so their branches merge without renumbering. Each entry names the branch it covers and is deleted when that branch merges; the list only ever shrinks.

CI reports; it does not rewrite. `.pre-commit-config.yaml` sets `autofix_prs: false`, and a check that silently renames a file the author is still editing would fit that convention badly.

## Alternatives considered

### Central assignment by a maintainer, as Python PEPs do

A PEP author submits with a `pep-9999` placeholder and an editor assigns the real number at review. IETF RFCs work the same way, with the RFC Editor assigning at publication.

Rejected because the mechanism depends on a standing group of editors and a submission rate measured in weeks. It puts a human round trip in the path of every ADR and still needs somewhere to record which numbers are taken. The useful half of the idea — the document carries a placeholder while in flight and the number is assigned by whoever has global knowledge — is what this decision keeps. It substitutes GitHub for the editor.

### Date-prefixed filenames

Name files `2026-08-08-<slug>.md`. Rails migrations moved from `001_create_users.rb` to timestamps for exactly this reason.

Rejected because it breaks citation. `ADR-0011` appears in code comments across the catalog; `ADR-2026-08-08` is not a usable substitute, and dropping to bare dates gives no stable short identifier at all. Same-day collisions also remain possible, so it trades a common failure for a rare one rather than eliminating it.

### No numbers, slugs only

Kotlin KEEPs and arc42 identify decisions by title alone.

Rejected *as the sole identifier*, for the same citation reason and more sharply: it would require rewriting all sixty-odd existing references and would leave nothing short to write in a comment.

Its core claim is nonetheless correct — a slug is a better identity than a number, because it is stable from the moment the file exists and carries meaning on its face. So it is adopted alongside rather than instead: the slug identifies, the number abbreviates. Keeping both is what allows an ADR to be cited before a forge has numbered it, at the cost of two citation forms to learn.

### An index file that reserves numbers

Keep sequential numbers, and have each pull request append its claim to a shared `README` table.

Rejected, though it is closer to workable than it looks. Its virtue is that concurrent claims collide on the same line range, so git *detects* what filenames hide. Its vice is that this happens on every concurrent ADR, turning a rare silent bug into a frequent noisy one, and it adds a second place where the number is recorded and can drift. Deriving the number from the forge gets the detection for free without the conflicts.

An index also serves a second purpose that is easy to miss: reserving a number is the only way to cite an unlanded ADR when numbers are the sole citation form. That function is real, and rejecting the index without replacing it would have removed something load-bearing. Named references replace it directly, and better — a reservation records that a number is taken, while a named reference records *which decision is meant*, and it needs no shared file to do so.

### Content-addressed identifiers with a head check

Alembic gives each migration a random hash and a parent pointer, then fails when the graph has more than one head.

Deferred as over-engineered for prose documents. The insight — detect the collision instead of trying to avoid it — is adopted directly in `scripts/adr_check.py`. The machinery around it exists because migrations must execute in a total order, which ADRs do not.

### Renumber at merge time

Ask whoever merges to fix the number.

Rejected: this is the current de facto process, and all three incidents above happened under it. It requires one person to hold global state that git will not show them.

## Consequences

### Positive

- Collisions become impossible for new ADRs. The number is allocated by GitHub before any branch can contend for it.
- No registry, index, allocator, or maintainer round trip.
- Every new ADR gains a backlink to the pull request that introduced it, and usually to the implementing code.
- Existing numbers and their sixty-odd code citations are untouched.
- Dangling `ADR-NNNN` references now fail CI across the whole directory, including the legacy range, which was previously unchecked.
- A stack can cite its own unlanded entries by name, so it no longer has to reserve a range of numbers — removing the mechanism that produced the largest cluster of contended numbers in this repository.
- An ADR is citable from the moment it is written rather than from the moment its pull request opens.

### Negative

- Numbers are sparse and jump. A four-digit number in the 2000s no longer tells a reader that it is the nineteenth decision. Chronology moves to `git log docs/adr/`, which was always the more reliable source — see ADR-0006 and ADR-0007.
- The directory sorts by pull request number, which is chronological but leaves visible gaps.
- Two numbering conventions coexist permanently. The floor at 1000 makes the boundary unambiguous, but it is a rule a new contributor has to read.
- Two citation forms coexist permanently, which is genuine added surface. A reader meeting `ADR-catalog-single-git-remote` and `ADR-0011` has to know they are the same document. The rename sweep keeps the long form mostly confined to unlanded prose, but it does not eliminate it.
- Unresolved named references are only a warning, so a misspelled slug survives CI until someone reads it. This is the deliberate price of allowing forward references at all; a `- **Pending:** <slug>` header line would let the guard harden it later by erroring on undeclared unresolved names.
- One extra step: authors run `just adr-rename` after opening the pull request. CI fails until they do, which is the intent, but it does mean a red check on the first push of any pull request carrying an ADR.
- Closing a pull request and opening a replacement strands the number and requires a second rename. The check catches it.
- One ADR per pull request. A change that warrants two decision records needs two pull requests.
- `ci-adr` must be added to branch protection as a required check. That is repository configuration, not something this pull request can carry.

## References

- [README.md](README.md) — the numbering rule and the authoring workflow
- [template.md](template.md)
- `scripts/adr_check.py`, `.github/workflows/ci-adr.yml`
- Rust RFCs, which name each file after its pull request number: <https://github.com/rust-lang/rfcs>
- Kubernetes enhancement proposals, numbered by their tracking issue: <https://github.com/kubernetes/enhancements>
- PEP 1, on placeholder numbers and editor assignment: <https://peps.python.org/pep-0001/>
- [ADR-0006](0006-read-kwargs-hash-path-read-path-split.md) and [ADR-0007](0007-datafusion-plan-path-canonicalization.md) — the out-of-order pair
- [ADR-0004](0004-uv-as-sole-packaging-and-execution-runtime.md) and [ADR-0008](0008-wheel-based-packaging-pipeline.md) — the supersession pattern this keeps
