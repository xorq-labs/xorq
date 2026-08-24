# AGENTS.meta.md — how to write this project's AGENTS.md

This file is a template, not a rulebook: it names the *shapes* of agent
directive that recur across well-run software projects, independent of
language, tooling, or domain. A project's actual `AGENTS.md` should
instantiate each category below with concrete, tool-specific rules. Where a
category doesn't apply to a given project (no comparable surface exists),
say so explicitly in that project's `AGENTS.md` rather than silently
omitting it — an absence should read as "considered, not applicable," not
"forgotten."

## Apex principle

Never let a convenient or weak signal substitute for the real thing you
actually need: real evidence, real consent, real verification at a trust
boundary, or a real, undistorted record of what happened.

The four examples above are the most common shapes this takes, not an
exhaustive taxonomy — categories 4, 5, 7, and 8 below instantiate the same
apex principle without mapping onto one of those four nouns directly (an
inaccurate label, an exposed secret, an unreviewed scope-widening, and a
duplicated fact are all still "a weak signal standing in for the real
thing," they just aren't evidence, consent, a boundary crossing, or
edited history).

Every category below is a specific instance of this. When a new situation
doesn't fit an existing category, ask which weak signal is being proposed
as a stand-in for the real one, and write the rule from there.

## The categories

**1. Preserve a truthful record of accepted work; don't rewrite it after the fact.**
Once a change, decision, or artifact is accepted and shared, its history
should reflect what actually happened. This is a behavioral rule, not a
claim about the storage medium — the record may well be technically
editable; the obligation is simply not to edit it once it's accepted.
This doesn't forbid cleanup of *your own* not-yet-shared work
(squashing local WIP commits, rewriting a draft before review) — the line
is whatever "shared/accepted" means in this project's workflow (merged to
the trunk branch, tagged, published). Decisions that are revised later get
appended to (amendments, follow-up entries), not rewritten in place.

**2. Require real evidence before asserting correctness or completion.**
Claims of "done," "fixed," or "verified" must be backed by something a
reader could independently check — a reproduced failure, a passing test
that would have failed before the change, a named contract the change
satisfies. Don't write a test that only proves the code contains certain
text or calls a certain function; prove behavior, not string presence.
Closing/resolving a tracked unit of work is a stronger claim than
commenting on it — if the evidence isn't there yet, downgrade to a comment
or a "needs review" state instead of closing.

This extends to language itself, not just process. A comment or docstring
that declares a guaranteed contract — an invariant, a precondition or
postcondition, an absolute claim about behavior — is making a claim of
correctness in miniature, and deserves the same evidence-gating as any
other one: the contract should be backed by an assertion or test that
would actually catch a violation. But ordinary prose uses the same words
("always," "never," "must") without meaning to assert anything formal, so
a declared contract needs its own unambiguous marker — a small closed
vocabulary of tags, or a dedicated section in structured documentation —
so the obligation falls only on claims meant as contracts, and ordinary
language stays exempt. Where practical, prefer expressing the contract as
an executable assertion at its own site over a prose claim plus a
separate test elsewhere; that also avoids creating a second copy for
category 8 to worry about.

**3. Gate irreversible or high-blast-radius actions behind explicit, current authorization.**
Some actions are expensive or impossible to undo: schema/data migrations,
deleting persisted state, publishing a release, rotating or revoking
credentials, invoking a tool whose side effects reach outside the local
workspace. For these, state the exact action and its consequences and wait
for an explicit go-ahead addressed to that specific action. A standing
policy, an approval of something else, or silence does not count. Adjacent
context — a green test suite, a feature request, an unreleased/pre-alpha
status — is not consent and should not be treated as implying it.

**4. Represent work accurately to external or future audiences.**
Don't manufacture visibility that wasn't asked for (unsolicited comments,
notifications, broadcasts), and don't frame routine or expected work as a
discovered failure, a gap, or someone else's mistake. Both are the same
underlying move: manufactured visibility implies something happened that
warrants attention, and an unrequested "verified" or "testing" section
implies a review process the reader has no way to check — both overstate
what actually occurred, the same way a mischaracterized routine change
does. If the project's
tooling mechanically derives artifacts from your descriptions (a changelog
built from commit messages, release notes built from PR titles), treat
getting that description right as a correctness requirement, not a style
preference — an inaccurate label there corrupts a downstream artifact, not
just the narrative.

**5. Minimize exposure of sensitive or identifying data.**
Real names, credentials, customer data, internal hostnames, or other
non-public identifiers shouldn't end up in tests, docs, examples, fixtures,
issue reports, or generated artifacts unless there's an explicit,
deliberate reason to preserve them. Default to neutral placeholders.

**6. Trust is scoped to a boundary; assume trust within it, verify every crossing.**
Identify the actual boundaries in this project — process boundaries,
network origins, authentication principals, tenant/workspace boundaries,
trusted-vs-untrusted input sources — and write down what's assumed safe
inside each one versus what must be independently checked whenever
authority, a credential, or data crosses from one side to the other. A
capability that a trusted actor already has by design is not a
vulnerability; a credential or session reaching a boundary it wasn't
scoped for always is. If a project genuinely has no such boundaries (a
purely local, single-user tool with no network-facing or multi-principal
surface), say that plainly instead of inventing a threat model it doesn't
need.

**7. Grant or assume only the minimum authority, access, or footprint actually needed.**
This applies at whatever granularity the project has: import/dependency
surface, filesystem access, network reach, environment inheritance,
process permissions. Prefer the narrowest default and make any deliberate
widening (an optional dependency, an inherited credential, a broadened
scope) an explicit, reviewable exception rather than an accident.

**8. Keep every fact in exactly one authoritative place.**
When the same fact would need to live in two places — two tracked items
covering the same work, a comment restating what the code it sits next to
already expresses, a doc restating a decision recorded elsewhere — either
delete the redundant copy or replace it with a pointer to the
authoritative one, rather than maintaining both. Duplicated copies degrade
quietly: nothing fails when one goes stale, so the stale one keeps being
trusted anyway. This is why a comment explaining *what* code does (fully
duplicating what the code already expresses) should generally be deleted
rather than kept in sync forever, while a comment explaining *why*
(context not otherwise recoverable from the code) is a genuine, singular
source of information rather than a duplicate. It's also why, before
opening a new tracked item (issue, ticket, doc, ADR), you check whether
one already covers it and update that one instead. A fragmented trail is a
subtly untruthful one, even when every fragment is individually accurate.

## Using this file

A project's `AGENTS.md` should, for each category above, either:

- name the concrete mechanism that instantiates it (the actual command,
  file, convention, or workflow step), or
- state explicitly that the category doesn't apply here and briefly say
  why (e.g., "no persisted schema exists," "no network-facing service
  exists").

Categories should not be silently dropped, and this meta file should stay
free of any single project's tool names, commands, or file paths — those
belong in the particular `AGENTS.md`.

These categories are overlapping lenses on the same underlying concern,
not a mutually exclusive partition. A single project rule can satisfy more
than one at once — a rule about widening a credential's scope, for
instance, can be simultaneously an instance of category 3's
consent-gating and category 7's minimal-footprint default. Don't force a
rule into exactly one category, and don't treat that overlap as a defect
in either one.
