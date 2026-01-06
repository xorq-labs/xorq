# Xorq Documentation Style Guide

This guide builds on the [Google developer documentation style guide](https://developers.google.com/style). When something isn't covered here, follow Google's guidance. If it's still not clear, choose the option that is easier to read, scan, and understand.


## Product name and terminology

Xorq is always written with a capital **X** in body text, headings, and UI copy.

In code, commands, and config, match the API/CLI syntax exactly. Don't override what the code expects.

| Context | Correct | Wrong |
|---------|---------|-------|
| Body text, headings | Xorq | xorq, XORQ, XOR |
| Code, commands, config | Match API/CLI syntax exactly | Don't "fix" casing in code |

Use official names for core concepts: **build**, **node hash**, **backend**, **profile**.

Define each term once in a Concepts or Glossary page, then use it consistently everywhere.


#### Latin abbreviations

Don't use them.

- Write **"for example"** instead of **"e.g."**
- Write **"that is"** instead of **"i.e."** ([Google guidance](https://developers.google.com/style/abbreviations#latin-abbreviations))


#### People and inclusivity

Use **"people with disabilities"**, not **"the disabled"** or **"disabled people"** ([Google guidance](https://developers.google.com/style/accessibility)).


## Voice and tone

Write like you're talking to a professional data engineer who knows their field but doesn't have time for marketing speak.

Be direct. Be helpful. Skip the hype.

| Good | Bad |
|------|-----|
| Use Xorq to build a pipeline that runs on DuckDB. | Xorq is an innovative next-generation orchestration engine that revolutionizes data workflows. |
| Configure the backend in the profile file. | Leverage our cutting-edge configuration paradigm to unlock backend power. |

Focus on what the user is trying to do, not on how impressive Xorq is.



## Person, tense, and pronouns

Use second person (**"you"**) to talk to the reader.

Use present tense unless you're describing something that happened in the past.

| Good | Bad |
|------|-----|
| You configure the backend in the profile file. | The user will configure the backend. |
| Xorq builds the pipeline and writes output to the builds directory. | Xorq will build the pipeline and write output. |
| Run the command, then check the logs if you see an error. | After you have run the command, which might take several minutes depending on your environment, you should then proceed to inspect the logs in the event that any errors occurred. |

Avoid **"we"** unless you clearly mean the Xorq team (for example, in release notes or roadmap statements).



## Sentence and paragraph style

One main idea per sentence.

Keep paragraphs short — two to four sentences.

Avoid nested clauses when a simpler structure works.

Write so people can quickly scan the docs and find what they need.



## Headings and capitalization

Use sentence case for all headings (H1 through H6).

Make headings descriptive, not cute.

| Good | Bad |
|------|-----|
| Run a build with DuckDB | Run A Build With DuckDB |
| Serve a pipeline with xorq serve-unbound | Let's go live |
| Configure the profile file | Profile magic happens here |

Every page must have one H1 that states the topic clearly, plus logical nested headings (H2, H3) to break content into scannable sections.



## Content types

Use consistent documentation types across the docs. Each page should have one clear purpose. Don't mix tutorial content with reference material on the same page.

| Type | Purpose | Example |
|------|---------|---------|
| Quickstart | Get to a first successful build or run as fast as possible | "Get started with Xorq in 5 minutes" |
| Tutorial | Step-by-step, learning-focused walkthrough that takes the reader by the hand | "10-minute tour of Xorq" |
| How-to guide | Task-based instructions for specific goals that assume some existing knowledge | "How to serve a pipeline with Xorq and DuckDB" |
| Reference | API, CLI, config, and schema details | "xorq build command reference" |
| Concept | Explanations of ideas and architecture | "Deferred execution", "Multi-engine", "Profiles" |

Tutorials are learning-oriented and assume the reader is a beginner. You're responsible for deciding what the reader needs to know and what they should do. The tutorial takes them by the hand through a series of steps to complete a meaningful project.

How-to guides are goal-oriented and assume the reader already has some experience. They answer questions that only someone with existing knowledge would ask: "How do I…?". The reader knows what they want to achieve but doesn't yet know how.



## Guide structure

Every production guide (how-to guide) must follow this structure:

### Required sections

1. **Introduction** — State what the reader will accomplish by the end of the guide. Be specific about the outcome.

   Example: "By the end of this guide, you will have a CI/CD pipeline that automatically builds, tests, and deploys Xorq pipelines to production."

2. **Prerequisites** — List what the reader needs before starting. Include:
   - Required software and versions
   - Configuration that must be in place
   - Knowledge or completed tutorials
   - Access requirements (API keys, credentials)

   Example:
   - Xorq 0.3.4 or later installed
   - GitHub repository with Actions enabled
   - Completed the "Build reproducible environments" guide

3. **Step-by-step instructions** — Walk through the task with numbered steps. Each step should:
   - Have one clear action
   - Include code examples or commands
   - Explain why the step matters (briefly)
   - Show expected output when relevant

4. **Production considerations** — Cover real-world concerns:
   - Performance implications
   - Security best practices
   - Cost considerations
   - Scaling guidance
   - When to use this approach vs alternatives

5. **Troubleshooting** — Document common errors specific to this guide:
   - Error message (quoted exactly)
   - What it means
   - How to fix it
   - What to check

   Format each issue as:
   ```markdown
   ### Issue: Server won't start
   **Error:** `Address already in use`
   **Cause:** Port 8001 is already occupied by another process.
   **Solution:** Check running servers with `xorq ps`, then kill the process or choose a different port.
   ```

6. **Next steps** — Point to related guides or next logical tasks. Give the reader clear direction on where to go from here.

### Guide structure example

```markdown
# Deploy models to production

Learn how to deploy trained Xorq models as production-ready prediction endpoints using Apache Arrow Flight.

By the end of this guide, you will have a model serving predictions with sub-100ms latency.

## Prerequisites

- Xorq 0.3.4 or later
- Trained model from "Train your first model" tutorial
- Understanding of Flight protocol (see "Understand Flight protocol" tutorial)

## Step-by-step instructions

### 1. Build your model

Create a build with your trained model:

\`\`\`bash
xorq build train_model.py --builds-dir builds
\`\`\`

Expected output:
\`\`\`
Build complete: builds/45f3bdb/
\`\`\`

### 2. Serve the model

Start the Flight server:

\`\`\`bash
xorq serve-unbound builds/45f3bdb --port 8001
\`\`\`

[Continue with remaining steps...]

## Production considerations

**Performance:** Use connection pooling for high-traffic endpoints. See "Optimize model serving" for latency tuning.

**Security:** Enable TLS in production. Never expose Flight servers directly to the internet without authentication.

**Monitoring:** Integrate with your observability stack using OpenTelemetry. See "Monitor production deployments."

## Troubleshooting

### Issue: Server won't start
**Error:** `Address already in use`
**Cause:** Port 8001 is already occupied.
**Solution:** Check running servers with `xorq ps`, then kill the process or choose a different port with `--port 8002`.

### Issue: Predictions return wrong schema
**Error:** `ArrowTypeError: Expected schema X, got Y`
**Cause:** Model output schema doesn't match client expectations.
**Solution:** Verify model output with `xorq catalog info <build>` and update client schema.

## Next steps

- [Optimize model serving](optimize-model-serving.md) for production latency requirements
- [Monitor production deployments](monitor-production.md) to track system health
- [Version and promote models](version-promote-models.md) to manage multiple model versions
```

This structure ensures every guide is complete, production-ready, and helps users succeed the first time.



## Lists, steps, and tables

### Lists

Use bullet lists for unordered information.

Use numbered lists for sequences and procedures.

Example procedure:

1. Install the required backend.
2. Configure your profile.
3. Run `xorq build` with your script.
4. Serve the generated build.

### Tables

Use tables for parameter lists, flags, and comparisons.

Don't use tables for layout.

Keep headers short and descriptive.

| Flag | Description | Default |
|------|-------------|---------|
| `--builds-dir` | Directory where build artifacts are stored | `builds` |
| `--engine` | Execution engine to use | `expr` |



## Code, commands, and output

Code and CLI examples are critical in Xorq docs. Follow these guidelines:

- Use fenced code blocks with the correct language hint (`bash`, `python`, `sql`, and so on).
- Separate input and output. Show the command in one block, then show the output in another block, labeled as output.

Good example:

```bash
xorq build penguins_analysis.py -e expr --builds-dir builds
```

Output:

```
Build complete. Wrote manifest to builds/45f3bdbdb521/manifest.json
```

Don't put long comments inside commands users will copy and paste. Explain the command above or below the code block instead.

### Placeholders

Use clear placeholders in angle brackets and explain them once per page:

```bash
xorq serve-unbound builds/<build_id> --host localhost --port 8001
```

Then explain: "Replace `<build_id>` with your actual build directory name (for example, 45f3bdbdb521)."

### Platform differences

If commands differ on Windows, macOS, or Linux, show each explicitly and label them. Don't assume only bash or zsh.



## Links

Make link text descriptive. Don't use "click here" or raw URLs. ([Google guidance](https://developers.google.com/style/link-text))

| Good | Bad |
|------|-----|
| See the CLI reference for xorq build. | Click here. |
| Read about deferred execution in the Concepts guide. | More info at https://docs.xorq.dev/concepts |

Link to related tutorials and how-to guides, concept pages that explain jargon, and reference pages for flags, options, and API details.

Every major page should have "Next steps" links at the end.



## Images and screenshots

Use screenshots when they clarify an important step or result—showing the builds folder and manifest, or a successful served pipeline, for example.

Add alt text that describes what the user should notice, not just "screenshot." ([Google guidance](https://developers.google.com/style/images))

| Good alt text | Bad alt text |
|---------------|--------------|
| Screenshot of the builds directory showing manifest.json in the 45f3bdbdb521 folder. | Screenshot |
| Terminal output showing successful build completion with node hash. | Image of terminal |

Don't use images as the only way to convey critical information.



## Accessibility

Write and structure docs so they work for people with disabilities and for people on different devices. ([Google guidance](https://developers.google.com/style/accessibility))

- Use clear headings and logical structure.
- Use proper list markup for steps and bullets.
- Write meaningful link text (not "here" or "this").
- Include alt text for images.
- Avoid overly complex tables and deep nested lists when a simpler structure works.



## Inclusive and bias-free language

Avoid terms that stereotype or exclude. Use neutral, precise language.

Prefer "you" over "the user" when speaking to the reader.

| Good | Bad |
|------|-----|
| developers, data engineers | rockstar developers, ninja coders |
| people with disabilities | the disabled, the handicapped |
| example data, test data | dummy data for idiots |

When in doubt, check the [Google style guide's word list](https://developers.google.com/style/word-list) and [accessibility guidance](https://developers.google.com/style/accessibility).



## Numbers, dates, and units

### Numbers

Use numeric form for 10 and above. Spell out one through nine, unless it's a unit or parameter.

Examples:

- "You can run up to 3 builds in parallel."
- "The cluster has 16 workers."

### Dates

Use unambiguous formats in docs.

| Good | Bad |
|------|-----|
| 21 November 2025 | 11/21/25 |
| 2025-11-21 (in code or logs) | 11/21/2025 |

### Units

Use SI units and be consistent (MB versus MiB, for example) when it matters.



## UI text and messages

When you document UI labels, flags, or error messages, match the UI text exactly (including case).

Use formatting to make it clear:

- Buttons: bold → "Click **Run build**."
- Flags or keys: code → "Set `builds-dir` in the config."

For error messages, quote them exactly, then explain what they mean and what to do next.

Error:

```
Error: Backend not found for engine 'duckdb'
```

**What it means:** The DuckDB backend isn't installed or configured in your environment.

**What to do:** Install the backend with `pip install "xorq[duckdb]"` and verify your profile configuration.



## File and navigation structure

Every page should answer one clear question.

| Page type | Question it answers |
|-----------|---------------------|
| Quickstart | How do I get my first pipeline running? |
| How-to | How do I serve a pipeline? |
| Reference | What flags does xorq build support? |
| Concept | What is deferred execution and why does it matter? |

Keep sidebar hierarchy shallow and intuitive. Avoid deep nesting. Avoid duplicate or near-duplicate topics with small differences.

---

## Section transitions

Connect sections so the reader always knows where they are and where they're going next.

Good example:

"With header normalization all set, the next step is to make sure caching is working perfectly.

Each section should bridge to the next one. Tell readers what they just finished and what comes next. This keeps them oriented and prevents them from feeling lost.



## Words and phrases to avoid

Don't use marketing language or jargon that clutters technical documentation. Avoid these words and phrases entirely:

### Completely avoid

- Foster
- Revolutionize (use "transform" instead, except in narratives)
- Landscape (especially "in the fast-paced tech landscape...")
- Underscoring
- Seamless (avoid or reduce usage)
- Robust (avoid or reduce usage)
- Streamline
- Delve
- Dives deep
- Elevate
- Realm
- Facilitate
- Groundbreaking
- Evolve
- Emergence
- Comprehensive (avoid or reduce usage)

### Prohibited phrases

- "Dives deep"
- "This is where x comes in"
- "X is not just about; it's about"
- "X isn't just nice to have; it's..."
- "You're not alone"
- "Thrive in a fast-paced environment"

### Limited usage (2-3 instances max per page)

- Ensure (appears multiple times in articles—keep to 2-3 instances)
- Enhance (often overused—reduce to 2 per page)
- Significant(ly) / Effective(ly) (often overused—reduce to 2-3 instances)
- Essential (often overused—reduce to 2-3 per page)
- Em dashes (—) (often overused—reduce to 2-3 instances)

These words add nothing. Cut them. Write plainly instead.



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



## How to use this guide

When you write or review Xorq docs:

1. Start with this style guide and the Xorq-specific decisions.
2. If something isn't covered here, follow the [Google developer documentation style guide](https://developers.google.com/style).
3. If it's still unclear, choose the option that is easier to read, scan, and understand.


