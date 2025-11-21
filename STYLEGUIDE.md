# Xorq Documentation Style Guide

Write for data engineers who are smart but busy. Keep it clear, scannable, and consistent.

This guide builds on the [Google developer documentation style guide](https://developers.google.com/style). When something isn't covered here, follow Google's rules. When that's still unclear, pick the option that's easier to read, scan, and understand for non-native English speakers.

---

## Product name and terminology

**Xorq** is always written with a capital X in body text, headings, and UI copy.

In code, commands, and config, match the API/CLI syntax exactly. Don't override what the code expects.

| Context | Correct | Wrong |
|---------|---------|-------|
| Body text, headings | Xorq | xorq, XORQ, XOR |
| Code, commands, config | Match API/CLI syntax exactly | Don't "fix" casing in code |

Use official names for core concepts (build, node hash, backend, profile). Define each term once in a Concepts or Glossary page, then use it consistently everywhere.

### Latin abbreviations

Don't use them. Write "for example" instead of "e.g." Write "that is" instead of "i.e." ([Google guidance](https://developers.google.com/style/abbreviations#latin-abbreviations))

### People and inclusivity

Use "people with disabilities", not "the disabled" or "disabled people". ([Google guidance](https://developers.google.com/style/accessibility))

---

## Voice and tone

Write like you're talking to a professional data engineer who knows their field but doesn't have time for marketing speak.

Be direct. Be helpful. Skip the hype.

| Good | Bad |
|------|-----|
| Use Xorq to build a pipeline that runs on DuckDB. | Xorq is an innovative next-generation orchestration engine that revolutionizes data workflows. |
| Configure the backend in the profile file. | Leverage our cutting-edge configuration paradigm to unlock backend power. |

Focus on what the user is trying to do, not on how clever Xorq is.

---

## Person, tense, and pronouns

Use second person ("you") to talk to the reader. Use present tense unless you're describing a past event.

| Good | Bad |
|------|-----|
| You configure the backend in the profile file. | The user will configure the backend. |
| Xorq builds the pipeline and writes output to the builds directory. | Xorq will build the pipeline and write output. |
| Run the command, then check the logs if you see an error. | After you have run the command, which might take several minutes depending on your environment, you should then proceed to inspect the logs in the event that any errors occurred. |

Avoid "we" unless you clearly mean the Xorq team (for example, release notes or roadmap statements).

---

## Sentence and paragraph style

One main idea per sentence. Keep paragraphs short (2–4 sentences). Avoid nested clauses when a simpler structure works.

Write so people can quickly scan the docs and find what they need.

---

## Headings and capitalization

Use sentence case for all headings (H1–H6). Make headings descriptive, not cute.

| Good | Bad |
|------|-----|
| Run a build with DuckDB | Run A Build With DuckDB |
| Serve a pipeline with xorq serve-unbound | Let's go live |
| Configure the profile file | Profile magic happens here |

**Every page must have:**
- One H1 that states the topic clearly
- Logical nested headings (H2, H3) to break content into scannable sections

---

## Content types

Use consistent topic types across the docs. Each page should have one clear purpose. Don't mix tutorial content with reference material on the same page.

| Type | Purpose | Example |
|------|---------|---------|
| **Quickstart** | Get to a first successful build/run as fast as possible | "Get started with Xorq in 5 minutes" |
| **Tutorial** | Step-by-step, learning-focused walkthrough | "10-minute tour of Xorq" |
| **How-to guide** | Task-based instructions for specific goals | "How to serve a pipeline with Xorq and DuckDB" |
| **Reference** | API, CLI, config, and schema details | "xorq build command reference" |
| **Concept** | Explanations of ideas and architecture | "Deferred execution", "Multi-engine", "Profiles" |

---

## Lists, steps, and tables

### Lists

Use bullet lists for unordered information. Use numbered lists for sequences and procedures.

**Example procedure:**

1. Install the required backend.
2. Configure your profile.
3. Run `xorq build` with your script.
4. Serve the generated build.

### Tables

Use tables for parameter lists, flags, and comparisons. Don't use tables for layout.

Keep headers short and descriptive.

| Flag | Description | Default |
|------|-------------|---------|
| `--builds-dir` | Directory where build artifacts are stored | `builds` |
| `--engine` | Execution engine to use | `expr` |

---

## Code, commands, and output

Code and CLI examples are critical in Xorq docs. Follow these rules:

Use fenced code blocks with the correct language hint (`bash`, `python`, `sql`, etc.).

**Separate input and output:**
- Show the command in one block
- Show the output in another block, labeled as output

**Good example:**

```bash
xorq build penguins_analysis.py -e expr --builds-dir builds
```

**Output:**

```
Build complete. Wrote manifest to builds/45f3bdbdb521/manifest.json
```

Don't put long comments inside commands users will copy-paste. Explain the command above or below the code block instead.

### Placeholders

Use clear placeholders in angle brackets and explain them once per page:

```bash
xorq serve-unbound builds/<build_id> --host localhost --port 8001
```

Then explain: "Replace `<build_id>` with your actual build directory name (for example `45f3bdbdb521`)."

### Platform differences

If commands differ on Windows, macOS, or Linux, show each explicitly and label them. Don't assume only bash or zsh.

---

## Links

Make link text descriptive. Don't use "click here" or raw URLs. ([Google guidance](https://developers.google.com/style/link-text))

| Good | Bad |
|------|-----|
| See the [CLI reference for xorq build](#). | Click [here](#). |
| Read about [deferred execution](#) in the Concepts guide. | More info at https://docs.xorq.dev/concepts |

**Link to:**
- Related tutorials and how-to guides
- Concept pages that explain jargon
- Reference pages for flags, options, and API details

Every major page should have "Next steps" links at the end.

---

## Images and screenshots

Use screenshots when they clarify an important step or result (for example, showing the builds folder and manifest, or a successful served pipeline).

Add alt text that describes what the user should notice, not just "screenshot." ([Google guidance](https://developers.google.com/style/images))

| Good alt text | Bad alt text |
|---------------|--------------|
| Screenshot of the builds directory showing manifest.json in the 45f3bdbdb521 folder. | Screenshot |
| Terminal output showing successful build completion with node hash. | Image of terminal |

Don't use images as the only way to convey critical information.

---

## Accessibility

Write and structure docs so they work for people with disabilities and for people on different devices. ([Google guidance](https://developers.google.com/style/accessibility))

- Use clear headings and logical structure
- Use proper list markup for steps and bullets
- Write meaningful link text (not "here" or "this")
- Include alt text for images
- Avoid overly complex tables and deep nested lists when a simpler structure works

---

## Inclusive and bias-free language

Avoid terms that stereotype or exclude. Use neutral, precise language.

Prefer "you" over "the user" when speaking to the reader.

| Good | Bad |
|------|-----|
| developers, data engineers | rockstar developers, ninja coders |
| people with disabilities | the disabled, the handicapped |
| example data, test data | dummy data for idiots |

When in doubt, check the [Google style guide's word list](https://developers.google.com/style/word-list) and [accessibility guidance](https://developers.google.com/style/accessibility).

---

## Numbers, dates, and units

### Numbers

Use numeric form for 10 and above. Spell out one through nine, unless it's a unit or parameter.

**Examples:**
- "You can run up to 3 builds in parallel."
- "The cluster has 16 workers."

### Dates

Use unambiguous formats in docs.

| Good | Bad |
|------|-----|
| 21 November 2025 | 11/21/25 |
| 2025-11-21 (in code/logs) | 11/21/2025 |

### Units

Use SI units and be consistent (for example, MB vs MiB) when it matters.

---

## UI text and messages

When you document UI labels, flags, or error messages:

Match the UI text exactly (including case).

**Use formatting to make it clear:**
- **Buttons:** bold → "Click **Run build**."
- **Flags/keys:** code → "Set `builds-dir` in the config."

For error messages, quote them exactly, then explain what they mean and what to do next.

**Error:**

```
Error: Backend not found for engine 'duckdb'
```

**What it means:** The DuckDB backend isn't installed or configured in your environment.

**What to do:** Install the backend with `pip install "xorq[duckdb]"` and verify your profile configuration.

---

## File and navigation structure

Every page should answer one clear question:

| Page type | Question it answers |
|-----------|-------------------|
| Quickstart | How do I get my first pipeline running? |
| How-to | How do I serve a pipeline? |
| Reference | What flags does `xorq build` support? |
| Concept | What is deferred execution and why does it matter? |

Keep sidebar hierarchy shallow and intuitive. Avoid deep nesting. Avoid duplicate or near-duplicate topics with small differences.

---

## Xorq-specific word list

Use these terms consistently across all Xorq docs. When in doubt, refer to the [Google word list](https://developers.google.com/style/word-list) and this table.

| Term | Usage | Notes |
|------|-------|-------|
| Xorq | Always capital X | Except in code where syntax demands otherwise |
| Quickstart | One word, capital Q | When used as a title ("Xorq Quickstart") |
| backend | Lowercase in running text | Unless part of a formal name |
| profile | Lowercase in running text | Use code formatting when referring to a key or field |
| build | Lowercase | As in "run a build" |
| builds directory | Lowercase | The directory where builds are stored |
| node hash | Lowercase | Unique identifier for a node in the execution graph |
| manifest | Lowercase | The JSON file containing build metadata |

Define each term once in the Concepts section, then use it the same way everywhere.

---

## How to use this guide

When you write or review Xorq docs:

1. Start with this style guide and the Xorq-specific decisions
2. If something isn't covered here, follow the [Google developer documentation style guide](https://developers.google.com/style)
3. If it's still unclear, choose the option that is easier to read, scan, and understand for non-native English speakers

**Consistency and clarity beat cleverness.**