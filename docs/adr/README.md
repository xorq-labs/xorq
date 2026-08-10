# Architecture decision records

This directory records architecture decisions: choices where reasonable people could disagree, where the reasoning outlives the diff, and where a future reader needs to know what else was on the table. If the decision is "fix the obvious bug," it belongs in a commit message instead.

Start from [template.md](template.md).

## Numbering

| Number | Meaning |
|--------|---------|
| Below 1000 | Legacy sequential number, frozen |
| 1000 and above | The number of the pull request that added the ADR |

A new ADR takes the number of the pull request that adds it. GitHub allocates pull request numbers, so two branches can't claim the same ADR number and you never have to check which number is free.

ADR-0002 through ADR-0017 predate this rule and keep their sequential numbers permanently. Around fifty comments across `python/xorq` cite them by number, so renumbering would either break those citations or require a risky sweep of them. Pull request numbers are already far above 1000, so the two ranges can't collide. A small allowlist in `scripts/adr_check.py` covers the legacy-numbered ADRs that were still in flight when this rule landed; it only shrinks.

Numbers are sparse and non-consecutive under this rule. That's the trade: the number tells you which pull request introduced the decision rather than how many decisions came before it. For chronology, use `git log docs/adr/`.

## Citing an ADR

There are two forms, and both keep working forever:

| Form | Example | Use it when |
|------|---------|-------------|
| Numbered | `ADR-0011` | the ADR has a number — the short form, preferred once one exists |
| Named | `ADR-catalog-single-git-remote` | the ADR has no number yet, or you're citing across branches |

The slug is an ADR's real identity: it's fixed when the file is created and never changes. The number is an alias that arrives when the pull request does.

The named form is what lets you cite an ADR that isn't numbered yet — including one that lands in a *later* pull request, which is common in a stack. That's why no one has to reserve a block of numbers ahead of time.

`scripts/adr_rename.py` rewrites named references to the ADR it numbers, so prose settles on the short form. Because the named form never stops resolving, a sweep that misses something is harmless.

CI treats the two asymmetrically. A numbered reference that doesn't resolve is an **error** — the forge allocated that number, so a missing one is a mistake. A named reference that doesn't resolve is a **warning**, because the ADR it names may still be on a branch that hasn't landed.

## Writing one

1. `python3 scripts/adr_new.py my-decision` copies the template to `docs/adr/XXXX-my-decision.md`.
2. Write it. The `XXXX` placeholder stays in the filename until a pull request exists.
3. Open the pull request. An ADR usually rides along with the code that implements it, which is what makes the pull request number a useful identifier.
4. `python3 scripts/adr_rename.py` reads the pull request number, renames the file, updates the heading, and rewrites any named references to it. Pass a slug — `python3 scripts/adr_rename.py my-decision` — if more than one ADR is in flight.
5. Push.

The `ci-adr` check fails while a filename still contains `XXXX`, so an unnumbered ADR can't merge.

Write one ADR per pull request. The number comes from the pull request, so a second ADR in the same pull request has no number to take. Open another pull request — an ADR reviews fine on its own, and splitting a set of related decisions costs nothing, because each is citable as `ADR-<slug>` from the moment it is written. Cite the others by slug; don't wait for their numbers, and don't reserve any.

The allowlist in `scripts/adr_check.py` is the one exception: a branch that predates this rule can land the legacy-numbered ADRs it already wrote, together, because their numbers were claimed before the rule existed. Each entry names one file, by number *and* slug, and is deleted when its branch merges.

## What CI checks

`scripts/adr_check.py` runs from [ci-adr.yml](../../.github/workflows/ci-adr.yml) on every pull request and on pushes to `main`. It verifies that:

- filenames match `NNNN-slug.md`, where the slug is lowercase words joined by single hyphens
- no two ADRs share a number, and no two share a slug
- a newly added ADR's number equals its pull request number
- the `# ADR-NNNN:` heading agrees with the filename
- every `ADR-NNNN` mention and every same-directory Markdown link resolves to a real ADR
- every `ADR-<slug>` mention resolves, or is reported as a warning if it doesn't

Unnumbered ADRs are checked too — a file still carrying `XXXX` has its heading and its references validated like any other.

Run the directory checks locally at any time:

```bash
python3 scripts/adr_check.py
```

Add the new-ADR checks by naming a base and a pull request number:

```bash
python3 scripts/adr_check.py --base main --pr 2211
```

## Superseding an ADR

Don't edit a decision that was reversed. Mark the old ADR `Superseded by ADR-NNNN` with a link, and mark the new one `Accepted, supersedes ADR-NNNN`. ADR-0004 and ADR-0008 show the pattern, including the note at the top of the superseded file.

When the decision still holds but the implementation details changed, set the status to `Amended` and append a dated `## Amendment` section rather than rewriting the original text.

## Why there is no index file

An index listing every ADR would be a second place that records each number, and every concurrent pull request would append to the same line range and conflict — reintroducing exactly the coordination this scheme removes. Use `ls docs/adr/`, `git log docs/adr/`, or grep. Status and supersession live in each ADR's own header, which is the only place they can't go stale.

An index has one other use worth naming: when numbers are the only way to cite an ADR, reserving a row is the only way to point at one that hasn't landed. Named references cover that directly, and say more while doing it — a reservation records that a number is taken, a named reference records which decision you meant.

A curated reading order is a different thing from an index and is genuinely useful. If one is added, it should say plainly that it is a reading guide and not the source of truth for numbers.

## Why these files aren't on the docs site

ADRs are internal engineering history, not user documentation. `docs/_quarto.yml` excludes `adr/**` from the rendered site, `.vale.ini` turns off prose style checks for this directory, and `docs/lint.sh` skips it. Write for the next engineer to touch the code, not for a public audience.
