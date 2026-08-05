---
name: review-retro
description: The retrospective step of xorq's sequential-review policy for shared-infrastructure changes (CONTRIBUTING § "Review policy") — group ALL findings from the review rounds by generating cause and convert causes into structural fixes. Use when two or more review rounds have completed, before merge — or whenever review findings feel like whack-a-mole.
---

# Review retrospective: from findings to generators

Individual review findings are symptoms. This skill finds the process that
*generated* them, so the fix kills the class, not the instance.

## When this runs (repo policy)

Per CONTRIBUTING § "Review policy: shared-infrastructure changes", changes to
shared traversal/hashing/serialization infrastructure get two or more
sequential independent cold reviews; when the rounds produce findings, this
retrospective runs before merge. The convergence curve (step 4) is the stop
criterion for the review loop itself. ADR-0016 records the #2177/#2196 case
study this practice was distilled from.

## Method

1. **Collect the full corpus.** Findings from ALL review rounds — including
   declined ones (they calibrate cost/benefit norms) and operational hiccups
   (auth failures, hook surprises, wrong paths, CI misattribution). Code
   findings alone miss half the causes.
2. **Group by generating cause, not chronology or severity.** For each
   finding ask: "what process would have produced this?" Many findings
   collapsing into few causes is the signal you are looking for.
   Then **test each cluster for over-unification**: findings that share a
   *signature* may not share a mechanism. Ask where the information is
   actually lost. A family of silent-degradation findings split three ways —
   at a boundary the code itself controls (a return type conflating absence,
   emptiness and failure), between wire and frame (a default swallowing a
   distinction the wire still carried), and on the wire itself (two states
   genuinely indistinguishable) — with three different cures, one of which
   does not exist. Claiming a single cure for a member that has none is
   manufacturing architecture. Argue the split; do not assert the unity.
3. **For each cause, seek a class-killer, not a finding-fixer.** Is there a
   mechanism — impossible-by-construction, import-time validation, or a
   co-located test — that makes the whole class impossible or loud?
   Enumeration-to-derivation is the most common answer: wherever coverage is
   defined by a hand-written list, look for a rule to close over instead.
   When elimination is not available, **invert the failure direction**. A
   hand-written identity-field list failed toward *stale cache hit* — silent
   wrong data. Deriving the list from the type's declared fields, with
   exclusion as an explicit annotation, made the same mistake fail toward
   *spurious cache miss* — wasteful and safe. Ask of every proposed
   mechanism: when this is gotten wrong, which way does it break? A cheap fix
   that flips the direction often beats an expensive one that narrows the
   probability.
4. **Read the convergence curve.** Findings-per-round and their severity
   trend tells you when another review round stops paying. A curve like
   10 → 5 (one high) → 5 (zero blocking) → suggestions-only means converged;
   flat or rising means the generators are still live.
5. **Route outputs to their durable homes.** Mechanisms → code + tests;
   design decisions → `docs/adr/` (see `template.md` there); process rules →
   CONTRIBUTING § "Removing or changing behavior" (with team consent — norms
   are not riders on feature PRs); agent-facing summaries → AGENTS.md;
   environment facts → memory; unresolved causes → named follow-ups with
   owners (ours: declaration-site spec registration → XOR-363).
6. **Apply cause #1 below to the new mechanisms themselves.** Every checker
   added gets a planted-violation test (prove it catches a real violation)
   and a stated blind spot (what it does NOT catch). Enforcement layers
   reliably exhibit the disease they treat.

## Two techniques worth reaching for

**Run the retro at two altitudes, blind.** One pass clusters the per-review
findings into causes (steps 2–3); a second reviews the change as a whole for
problems no single review could see, *without* the findings corpus. Neither
sees the other's output. Where they converge independently, the agreement is
evidence; where only one fires, you learn which altitude sees what. A
composition hazard is the standard example: a guard that is merely latent in
the PR that adds it becomes live in the PR that adds its first caller, and
each single-PR reviewer ranks it correctly for their own PR and wrongly for
the change as a whole. The cheap version of this is one extra pass whose input
is "guarantees each part *provides*" × "guarantees each later part *assumes*".

**Isolate before you diagnose.** A causal story that explains the evidence is
not the cause. Apply the one suspected commit alone to the base and check the
result reproduces byte-identically; if it does not, the story is wrong however
well it fits. This costs minutes and repeatedly overturns confident diagnoses
— including diagnoses that would have produced the wrong fix. The same
skepticism applies to *reported* facts: re-run the counts and re-read the
trees yourself, especially when a number is load-bearing for a decision.

## Known cause taxonomy (prior, not fence)

Distilled from two cases. Causes 1–5 come from the #2177/#2196 review rounds
(28 findings, four rounds; ADR-0016 records the code-side outcome). Cause 6,
the over-unification and failure-direction notes in steps 2–3, and the two
techniques above come from the rest-backend landing-stack retro of 2026-08-05
(~35 findings across five per-part reviews plus two blind structural passes;
ADRs 0018–0025 record the code-side outcomes). Cause 2 recurred there as the
highest-severity finding, which is the main reason to start from this list
rather than from scratch. Start from these — they have recurred — but
explicitly hunt for causes that do not fit; a taxonomy that only confirms
itself is another enumeration. Amend this list when a retro finds a new
cause.

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
6. **Documents written during the work, describing it as done** — an ADR or
   module docstring drafted mid-change states intent in the perfect tense
   ("the workaround is deleted", "verified: all N tests pass"), and landing
   invalidates it. Cheap to write, expensive to trust: a design record that
   misdescribes the shipped system sends the next reader to implement from a
   spec whose load-bearing mechanics were never true. What promotes this from
   a slip to a generator is that it *recurs after diagnosis* — in the case
   this was distilled from, two further documents reproduced the pattern
   within hours of it being named, because writing-as-you-go naturally
   produces present-tense claims and nothing in the workflow catches them.
   The rule: prose describes the tree it lands in; intended work goes under
   Consequences; a status of Proposed also requires naming what does not exist
   yet and the gate for accepting it. Grep candidates when reviewing docs:
   "verified", "now", "currently", "this branch".

## Reviewing against a document that may itself be wrong

When a change implements a design doc, "does this match the spec?" is the
wrong pass condition if the spec is the thing under suspicion. Charter the
review against the *goal* — correctness, the property the change exists to
win — and say explicitly that concluding "this should not land" is a valid
outcome. A spec whose evidence cannot be reproduced from the repo (a cited
prototype that was never committed, a test count no one can re-run) should be
treated as a lead, not an authority; verify its mechanics before building on
them.

## Scope discipline

This is a thinking scaffold, not an orchestration: it needs no fan-out and
no verification machinery. Its outputs are promotions up the ladder
(memory → skill prior → CONTRIBUTING/ADR), never a new document trail of
its own.
