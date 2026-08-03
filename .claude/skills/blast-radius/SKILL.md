---
name: blast-radius
description: Before removing or renaming behavior, compute the repo-wide blast radius — grep for removed symbols, error strings, and config keys across code, tests, and docs; report every hit that must be updated. Use when deleting code paths (even dead ones), renaming public names, or changing error messages.
---

# Blast radius of a removal or rename

Given the working diff (or a described removal/rename):

1. Extract from the removed or renamed lines:
   - function/class/variable names (public AND private — tests import private names),
   - string literals: error messages, log strings, config keys,
   - CLI flags, dotted module paths, entry-point names.
2. For each token, `grep -rn` the whole repo — including `**/tests/**`, `docs/`,
   and `.github/` — for the exact token AND for distinctive substrings of string
   literals (error messages are often asserted via `pytest.raises(match=...)`
   with only a fragment of the message).
3. Also search for the *module name itself* in other modules' test files:
   a test that does `monkeypatch.setattr(other_module, ...)` is in the blast
   radius even when no removed symbol matches textually.
4. Classify every hit:
   - (a) already updated by this diff,
   - (b) must be updated — **blocker**, fix before proceeding,
   - (c) safe to leave (changelog, historical notes).
5. Report the classification before making the removal. If there are zero hits,
   say exactly what was searched so the absence is verifiable.

Rules of thumb:

- Dead code can still have live tests. "Unreachable by construction" does not
  mean "unreferenced".
- When retiring a test that guarded a now-impossible drift, replace it with a
  structural invariant test co-located with the owning module, and leave a
  pointer comment at the old site.
- After the removal, re-run the test modules of every file that appeared in
  category (b), not just the module you edited.
