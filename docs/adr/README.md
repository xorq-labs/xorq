# Architecture decision records

This directory records architecture decisions: choices where reasonable people could disagree, where the reasoning outlives the diff, and where a future reader needs to know what else was on the table. If the decision is "fix the obvious bug," it belongs in a commit message instead.

Start from [template.md](template.md).

## Numbering

| Number | Meaning |
|--------|---------|
| Below 1000 | Legacy sequential number, frozen |
| 1000 and above | The number of the pull request that added the ADR |

A new ADR takes the number of the pull request that adds it. GitHub allocates pull request numbers, so two branches can't claim the same ADR number and you never have to check which number is free.

ADR-0002 through ADR-0017 predate this rule and keep their sequential numbers permanently. Roughly sixty comments across `python/xorq` cite them by number, so renumbering would either break those citations or require a risky sweep of them. Pull request numbers are already far above 1000, so the two ranges can't collide. A small allowlist in `scripts/adr_check.py` covers the legacy-numbered ADRs that were still in flight when this rule landed; it only shrinks.

Numbers are sparse and non-consecutive under this rule. That's the trade: the number tells you which pull request introduced the decision rather than how many decisions came before it. For chronology, use `git log docs/adr/`.

## Writing one

1. `just adr-new my-decision` copies the template to `docs/adr/XXXX-my-decision.md`.
2. Write it. The `XXXX` placeholder stays in the filename until a pull request exists.
3. Open the pull request. An ADR usually rides along with the code that implements it, which is what makes the pull request number a useful identifier.
4. `just adr-rename` reads the pull request number, renames the file, and updates the heading.
5. Push.

The `ci-adr` check fails while a filename still contains `XXXX`, so an unnumbered ADR can't merge.

Write one ADR per pull request. The number comes from the pull request, so a second ADR in the same pull request has no number to take. Open another pull request — an ADR reviews fine on its own.

## What CI checks

`scripts/adr_check.py` runs from [ci-adr.yml](../../.github/workflows/ci-adr.yml) on every pull request and on pushes to `main`. It verifies that:

- filenames match `NNNN-slug.md`, where the slug is lowercase words joined by single hyphens
- no two ADRs share a number
- a newly added ADR's number equals its pull request number
- the `# ADR-NNNN:` heading agrees with the filename
- every `ADR-NNNN` mention and every same-directory Markdown link resolves to a real ADR

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

## Why these files aren't on the docs site

ADRs are internal engineering history, not user documentation. `docs/_quarto.yml` excludes `adr/**` from the rendered site, `.vale.ini` turns off prose style checks for this directory, and `docs/lint.sh` skips it. Write for the next engineer to touch the code, not for a public audience.
