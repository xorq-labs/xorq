---
name: xorq
description: >
  Compute manifest and composable tools for ML. Build, catalog, and serve deferred
  expressions with input-addressed caching, multi-engine execution, and Arrow-native
  data flow. Use for ML pipelines, feature engineering, and model serving.
allowed-tools: "Read,Bash(xorq:*),Bash(python:*)"
version: "{{VERSION}}"
author: "Xorq Labs <https://github.com/xorq-labs>"
license: "Apache-2.0"
---

# Xorq - Manifest-Driven Compute for ML

A compute manifest system providing persistent, cacheable, and portable expressions for ML workflows. Expressions are tools that compose via Arrow.

## Agent Tool Compatibility

**For non-Claude Code agents (Codex, etc.):**
When xorq docs reference Claude Code-specific tools, map to your environment's equivalents:
- `TodoWrite` → Your planning/task tracking tool (e.g., `update_plan`)
- `Task` tool with subagents → Do the work directly (if subagents not available)
- `Skill` tool → Not needed (you're reading this skill directly)
- `Read`, `Write`, `Edit`, `Bash` → Use your native tools with similar functions

{{CORE_CONTENT}}

## Version

v{{VERSION}} - Consolidated skill with CLI + Python API coverage
