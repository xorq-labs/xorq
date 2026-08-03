---
name: review-retro
description: After sequential independent review rounds on a change, run a retrospective that groups ALL findings by generating cause and converts causes into structural fixes. Use when two or more review rounds have completed, before merge — or whenever review findings feel like whack-a-mole.
---

# Review retrospective: from findings to generators

Individual review findings are symptoms. This skill finds the process that
*generated* them, so the fix kills the class, not the instance.

## Method

1. **Collect the full corpus.** Findings from ALL review rounds — including
   declined ones (they calibrate cost/benefit norms) and operational hiccups
   (auth failures, hook surprises, wrong paths, CI misattribution). Code
   findings alone miss half the causes.
2. **Group by generating cause, not chronology or severity.** For each
   finding ask: "what process would have produced this?" Many findings
   collapsing into few causes is the signal you are looking for.
3. **For each cause, seek a class-killer, not a finding-fixer.** Is there a
   mechanism — impossible-by-construction, import-time validation, or a
   co-located test — that makes the whole class impossible or loud?
   Enumeration-to-derivation is the most common answer: wherever coverage is
   defined by a hand-written list, look for a rule to close over instead.
4. **Read the convergence curve.** Findings-per-round and their severity
   trend tells you when another review round stops paying. A curve like
   10 → 5 (one high) → 5 (zero blocking) → suggestions-only means converged;
   flat or rising means the generators are still live.
5. **Route outputs to their durable homes.** Mechanisms → code + tests;
   design decisions → an ADR; process rules → CONTRIBUTING (with team
   consent — norms are not riders on feature PRs); environment facts →
   memory; unresolved causes → named follow-ups with owners.
6. **Apply cause #1 below to the new mechanisms themselves.** Every checker
   added gets a planted-violation test (prove it catches a real violation)
   and a stated blind spot (what it does NOT catch). Enforcement layers
   reliably exhibit the disease they treat.

## Known cause taxonomy (prior, not fence)

Start from these — they have recurred — but explicitly hunt for causes that
do not fit; a taxonomy that only confirms itself is another enumeration.

1. **Claims without enforcement** — invariants living in prose (docstrings,
   comments, PR bodies) with no tier that makes them fail when violated.
   Includes checkers overclaiming their own coverage.
2. **Coverage by enumeration instead of closure** — a mechanism covers the
   cases its author listed; each review round finds the complement.
3. **Knowledge far from its point of use** — contract tests in another
   module's test file, conventions discoverable only by collision, docs that
   drift because they restate facts instead of citing their source.
4. **Environment assumptions untested until they fail** — credentials,
   hooks, CI quirks, branch policies. Cheapest fix is usually memory, not
   machinery.
5. **Single-perspective review saturation** — each independent cold review
   finds a different character of issue; sequential rounds converge
   measurably (see step 4).

## Scope discipline

This is a thinking scaffold, not an orchestration: it needs no fan-out and
no verification machinery. Its outputs are promotions up the ladder
(memory → skill prior → CONTRIBUTING/ADR), never a new document trail of
its own.
