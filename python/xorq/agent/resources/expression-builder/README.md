# Expression Builder Skill

**Lean, editable skill for building deferred xorq expressions.**

## Structure

```
expression-builder/
├── SKILL.md              # Main skill file (edit directly)
├── skill-rules.json      # Auto-activation triggers
├── README.md             # This file
└── resources/            # Additional reference docs
    ├── TROUBLESHOOTING.md   # Common errors and solutions
    └── ml-pipelines.md      # ML patterns and sklearn integration
```

## Philosophy

This skill uses **progressive disclosure** via `<details>` blocks:
- Core concepts and quickstart are always visible
- Advanced topics (UDAFs, ML pipelines, multi-engine) are collapsed
- Reference cheat sheet at the bottom
- Additional deep-dive resources available in `resources/` folder

## Editing

Just edit `SKILL.md` directly - no generation needed!

The skill follows this structure:
1. **Mental model** - 4-word summary
2. **Non-negotiables** - Critical rules (imports, schema checks)
3. **Quickstart** - Minimal working example
4. **Common patterns** - Everyday operations
5. **Advanced** - In `<details>` blocks
6. **Reference** - Quick cheat sheet

## Installation

This skill is installed when users run:
```bash
xorq agents skill install
```

Which copies it to `~/.claude/skills/expression-builder/`
