---
name: xorq
description: >
  Compute manifest and composable tools for ML. Build, catalog, and serve deferred
  expressions with input-addressed caching, multi-engine execution, and Arrow-native
  data flow. Use for ML pipelines, feature engineering, and model serving.
version: "{{VERSION}}"
author: "Xorq Labs <https://github.com/xorq-labs>"
license: "Apache-2.0"
---

# Xorq - Manifest-Driven Compute for ML

A compute manifest system providing persistent, cacheable, and portable expressions for ML workflows. Expressions are tools that compose via Arrow.

## Codex-Specific Notes

**Tool Mapping for Codex:**
When xorq docs reference Claude-specific tools, use your Codex equivalents:
- `TodoWrite` → `update_plan` (your planning/task tracking tool)
- `Task` tool with subagents → Do the work directly (subagents not available in Codex)
- `Skill` tool → Not needed (you're reading this skill directly)
- `Read`, `Write`, `Edit`, `Bash` → Use your native tools with similar functions

{{CORE_CONTENT}}

## Best Practices (Codex-Specific)

8. **Use update_plan** - Track your tasks with Codex's planning tool

## Version

v{{VERSION}} - Codex-adapted skill with tool mapping for OpenAI Codex CLI
