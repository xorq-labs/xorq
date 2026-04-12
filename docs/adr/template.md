# ADR-NNNN: <title — the decision, not the problem>

<!--
The title should name the decision or design choice, not the bug or symptom.
Good:  "Make git-annex optional via a CatalogBackend abstraction"
Bad:   "Fix catalog dependency issue"
-->

- **Status:** Proposed | Accepted | Superseded by ADR-NNNN
- **Date:** YYYY-MM-DD
- **Context area:** `path/to/primary/file.py`

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

## Rationale

<!--
Optional — use when the "why" is complex enough to warrant its own section
rather than being woven into Decision. Useful for justifying a non-obvious
choice, e.g. "Why graph rewrite, not mutation?" (ADR-0002) or "Why an ABC,
not a flag?" (ADR-0003).

If the reasoning fits naturally in Decision, omit this section.
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
