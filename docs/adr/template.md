# ADR-NNNN: <title — the decision, not the problem>

<!--
Save this file as docs/adr/NNNN-<slug>.md (e.g. 0012-click-option-decorators.md),
where NNNN is the next sequential number after the highest existing ADR in
this directory.

The title should name the decision or design choice, not the bug or symptom.
Good:  "Make git-annex optional via a CatalogBackend abstraction"
Bad:   "Fix catalog dependency issue"

Delete all guidance comments (HTML comments like this one) before merging.
-->

- **Status:** Proposed | Accepted | Amended | Rejected | Deprecated | Superseded by ADR-NNNN
- **Date:** YYYY-MM-DD <!-- when the decision was last updated -->
- **Deciders:** <names>

<!--
Status guide:
- Amended: the decision still holds, but implementation details changed.
  Append a dated "## Amendment" section; do not alter the original text.
- Superseded: the decision itself was reversed or replaced by a new ADR.
  Mark this ADR "Superseded by ADR-NNNN" and write the new ADR.
-->

## Context

<!--
What situation or problem prompted this decision? Include enough background
that a reader unfamiliar with the current code can understand why a decision
is needed. If there was a specific incident (CI hang, production bug), link
to it.

This section should make clear that reasonable people could disagree on the
right approach. If the "decision" is simply "fix the obvious bug," it
belongs in a commit message, not an ADR.
-->

## Decision drivers

<!--
Optional — a bullet list of the forces, constraints, or quality attributes
that shape the decision (e.g. "must not add a hard dependency on git-annex",
"startup time under 200ms", "compatible with existing catalog on disk").

Useful when alternatives need to be scored against the same criteria. If
the drivers are obvious from Context, omit this section.
-->

- <driver>

## Decision

<!--
What did we decide to do, and how does it work? This is the core of the ADR.

Structure as prose with subsections (###) for distinct design choices.
Use tables for comparisons, code blocks for key interfaces or signatures.
Explain the reasoning inline — why this approach, not just what it does.

Avoid pasting full implementations; reference file:line instead. Include
short code blocks only when a signature or interface is central to the
decision and would be hard to understand from the file alone.
-->

## Alternatives considered

<!--
What else was considered and why was it deferred or rejected?
Each alternative gets a ### heading, a brief description, and the reason
it was not chosen.

"Deferred" means it may be revisited; "Rejected" means it was ruled out.
State which, so future readers know whether the door is open.
-->

### <Alternative name>

<Description.>

<Deferred | Rejected> because:
- <reason>

## Consequences

### Positive

- <outcome>

### Negative

- <outcome — include known risks, migration costs, or fragility>

## References

<!--
Links to PRs, issues, incidents, related ADRs, external docs, or prior
discussions. Use a flat bullet list.
-->

- <link>
