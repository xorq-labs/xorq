# Xorq Agent Prompt Library

## Purpose

This directory now bundles the complete prompt set that powers the demo kernel agent. We ship it inside the Xorq repo so every new `xorq init --agent` flow can reference the same guidance materials without reaching into sibling repos. Prompts stay in Markdown so they can be curated, diffed, and selectively embedded into AGENTS/CLAUDE instructions.

## Bundles & Priority

We surface the most critical guides first during onboarding:

1. **Core workflow** – `planning_phase.md`, `sequential_execution.md`, `xorq_vendor_ibis.md`, and `context_blocks/xorq_core.md`
2. **Reliability aides** – troubleshooting and policy blocks under `context_blocks/` (schema checks, pandas avoidance, backend workarounds, fix_* guides)
3. **Advanced skills** – ML, optimization, plotting, deliverable guidance, and the `phase_*_completion_check.md` files

CLI helpers can read this README to know which prompts to inject at each step of the agent journey.

## Directory Structure

```
agent_prompts/
├── README.md
├── planning_phase.md
├── sequential_execution.md
├── xorq_vendor_ibis.md
├── llm_object_guide.md
├── package_exploration_guide.md
└── context_blocks/
    ├── xorq_core.md
    ├── xorq_ml_complete.md
    ├── xorq_connection.md
    ├── data_source_rules.md
    ├── must_check_schema.md
    ├── column_case_rules.md
    ├── avoid_pandas.md
    ├── transformation_patterns.md
    ├── optimization_patterns.md
    ├── plotting_patterns.md
    ├── summary_patterns.md
    ├── expression_deliverables.md
    ├── task_understanding.md
    ├── repl_exploration.md
    ├── pandas_udf_patterns.md
    ├── udaf_aggregation_patterns.md
    ├── backend_operation_workarounds.md
    ├── fix_schema_errors.md
    ├── fix_attribute_errors.md
    ├── fix_data_errors.md
    ├── fix_import_errors.md
    ├── fix_udf_backend_errors.md
    ├── phase_*_completion_check.md
    └── other supporting guides
```

All original filenames are preserved to keep the demo kernel agent compatible with this bundle.

## Using the Prompts

- **CLI auto-injection** – `xorq init --agent` should copy snippets from bundle tier 1 into the generated `AGENTS.md` and surface the rest via `xorq agent prompt list`.
- **Programmatic access** – Tools can load any prompt by reading the Markdown file and substituting `{variable}` slots (the `context_manager.py` pattern from the demo kernel still applies).
- **Manual reference** – Human contributors can browse these files for troubleshooting or to extend skills.

## Customization Workflow

1. Identify the prompt file you want to change.
2. Edit the Markdown directly (keep `{variable}` placeholders intact).
3. Update AGENT/CLAUDE docs if the change affects onboarding steps.
4. Commit alongside other doc updates so downstream consumers pick it up.

## Variables Reference

Prompts may include placeholders like `{xorq_import_instructions}`, `{var_descriptions}`, `{initial_prompt}`, `{xorq_capabilities}`, etc. Your agent runtime is responsible for substituting them before sending the final message to an LLM.

## Maintenance Tips

- Keep prompts scoped to a single purpose.
- Note the bundle tier (core/reliability/advanced) in commit messages.
- Validate rendered prompts by printing them in the runtime or via `xorq agent prompt show <name>`.
- When adding new files, update the structure table above so future maintainers can discover them quickly.

## Roadmap Ideas

- Tag files with YAML front‑matter describing bundle tier, applicable phases, and recommended CLI hook.
- Build a manifest (`prompts.json`) that the CLI can query for metadata instead of hardcoding paths.
- Track prompt usage/feedback to keep the library evolving with agent needs.
