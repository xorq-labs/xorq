# Xorq Skill for OpenAI Codex

This skill teaches OpenAI Codex how to use xorq for building ML pipelines and deferred data analysis.

## Installation

The skill is automatically installed when you run:

```bash
xorq agents init codex
# Or for both Claude and Codex:
xorq agents init codex,claude
```

This will:
1. Copy the skill files to your project's `.xorq/` directory
2. Add bootstrap instructions to `~/.codex/AGENTS.md`
3. Make xorq commands and patterns available to Codex

## What Gets Installed

- **~/.codex/AGENTS.md** - Bootstrap instructions added
- **Project .xorq/codex/** - Skill files (SKILL.md, scripts)

## Manual Installation

If you want to install manually:

1. Copy `skills/codex/SKILL.md` to `~/.codex/skills/xorq/SKILL.md`
2. Add the bootstrap content from `.xorq/codex/bootstrap.md` to `~/.codex/AGENTS.md`

## Using the Skill

Codex will automatically have access to xorq knowledge after installation. Ask questions like:

- "How do I build a data pipeline with xorq?"
- "Show me how to check schema and build an expression"
- "How do I catalog my builds?"

## Differences from Claude Version

The Codex version includes tool mapping since Codex has different built-in tools:

| Claude Tool | Codex Equivalent |
|-------------|------------------|
| TodoWrite | update_plan |
| Task (subagents) | Direct execution |
| Skill tool | Not needed |

## Version

v0.2.0 - Initial Codex skill based on Claude skill v0.2.0
