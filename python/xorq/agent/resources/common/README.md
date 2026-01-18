# Shared Skill Content Architecture

This directory contains the **single source of truth** for xorq agent skill documentation. Agent-specific SKILL.md files are generated from these shared templates.

## Problem Solved

Previously, we had ~80% duplicated content between:
- `skills/xorq/SKILL.md` (Claude Code)
- `python/xorq/agent/resources/codex/SKILL.md` (OpenAI Codex)

When xorq was updated, both files needed manual syncing, risking content drift.

**Solution:** Extract common content to shared templates, generate agent-specific skills automatically.

## Architecture

```
python/xorq/agent/resources/
├── common/                       # Shared content (YOU ARE HERE)
│   ├── README.md                 # This file
│   ├── skill_core.md             # 90% shared xorq guidance
│   ├── claude_wrapper.md         # Claude-specific sections
│   ├── codex_wrapper.md          # Codex-specific sections
│   └── generate_skills.py        # Generation script
├── codex/
│   ├── SKILL.md                  # GENERATED - Do not edit directly
│   ├── README.md                 # Codex installation docs
│   └── bootstrap.md              # Codex bootstrap content
└── ...
```

Plus:
```
skills/xorq/
├── SKILL.md                      # GENERATED - Do not edit directly
├── README.md                     # Claude Code docs
├── CLAUDE.md                     # Maintenance guide
└── resources/                    # Deep-dive documentation
```

## File Purposes

### skill_core.md
**90% of content** - All shared xorq guidance:
- Core concepts
- Quick start
- Essential CLI commands
- Python API essentials
- Critical rules
- Session protocol
- Common expression patterns
- Agent-native features
- Multi-engine support
- Troubleshooting
- Best practices

**DO NOT include:**
- Agent-specific tool mappings
- Installation instructions
- Front matter (version, allowed-tools)

### claude_wrapper.md
**10% Claude-specific:**
- Front matter with `allowed-tools`
- Tool compatibility notes for other agents
- References to `{{CORE_CONTENT}}` placeholder
- Version footer

### codex_wrapper.md
**10% Codex-specific:**
- Front matter without `allowed-tools`
- Codex tool mapping section
- References to `{{CORE_CONTENT}}` placeholder
- Codex-specific best practices
- Version footer

### generate_skills.py
**Generation logic:**
- Reads `skill_core.md` (shared content)
- Reads agent wrapper (claude or codex)
- Substitutes `{{CORE_CONTENT}}` and `{{VERSION}}`
- Writes to destination paths

## Workflow

### When xorq is Updated

**1. Edit shared content** (skill_core.md):
```bash
vim python/xorq/agent/resources/common/skill_core.md
```

**2. Regenerate all skills:**
```bash
python python/xorq/agent/resources/common/generate_skills.py
```

**3. Verify output:**
```bash
# Check generated files
git diff skills/xorq/SKILL.md
git diff python/xorq/agent/resources/codex/SKILL.md

# Verify line counts
wc -l skills/xorq/SKILL.md python/xorq/agent/resources/codex/SKILL.md
```

**4. Commit all files:**
```bash
git add python/xorq/agent/resources/common/skill_core.md
git add skills/xorq/SKILL.md
git add python/xorq/agent/resources/codex/SKILL.md
git commit -m "Update xorq skills with <new feature>"
```

### When Adding Agent-Specific Content

**1. Edit wrapper** (claude_wrapper.md or codex_wrapper.md):
```bash
vim python/xorq/agent/resources/common/claude_wrapper.md
```

**2. Regenerate:**
```bash
python python/xorq/agent/resources/common/generate_skills.py
```

**3. Verify and commit** (as above)

### When Changing Version

Update version in `generate_skills.py`:
```python
def generate_all_skills(version: str = "0.3.0") -> dict[str, Path]:
    ...
```

Then regenerate and commit.

## Integration

### Automatic Generation

Skills are regenerated automatically when:
- `xorq agents onboard` runs
- `register_claude_skill()` is called
- `register_codex_skill()` is called

See `python/xorq/agent/onboarding.py`:
```python
def register_claude_skill() -> Path | None:
    # Generate fresh skill from shared content
    try:
        from xorq.agent.resources.common.generate_skills import generate_all_skills
        generate_all_skills()
    except Exception as e:
        print(f"⚠️  Could not regenerate skills: {e}")
    ...
```

### Manual Generation

```bash
# Generate all skills
python python/xorq/agent/resources/common/generate_skills.py

# Generate specific agent
python python/xorq/agent/resources/common/generate_skills.py claude
python python/xorq/agent/resources/common/generate_skills.py codex

# Generate with custom version
python python/xorq/agent/resources/common/generate_skills.py claude 0.3.0
```

## Testing

### Verify Generation Works
```bash
# Generate skills
python python/xorq/agent/resources/common/generate_skills.py

# Check outputs exist
ls skills/xorq/SKILL.md
ls python/xorq/agent/resources/codex/SKILL.md

# Verify content differs appropriately
diff skills/xorq/SKILL.md python/xorq/agent/resources/codex/SKILL.md | head -30
```

### Test Registration
```bash
# Test Claude registration
python -c "from xorq.agent.onboarding import register_claude_skill; print(register_claude_skill())"

# Test Codex registration
python -c "from xorq.agent.onboarding import register_codex_skill; from pathlib import Path; print(register_codex_skill(Path.cwd()))"

# Verify installed
ls ~/.claude/skills/xorq/SKILL.md
ls .xorq/codex/SKILL.md
```

## Content Guidelines

### What Goes in skill_core.md

✅ **Include:**
- xorq concepts, commands, patterns
- Python API essentials
- Workflow protocols
- Troubleshooting common to all agents
- Best practices (general)

❌ **Exclude:**
- References to specific agent tools (TodoWrite, Task, etc.)
- Installation instructions (agent-specific)
- Agent-specific front matter

### What Goes in Wrappers

✅ **Include:**
- Front matter (YAML metadata)
- Tool mapping for that agent
- Installation/setup instructions
- Agent-specific notes
- Version information

❌ **Exclude:**
- xorq-specific guidance (put in skill_core.md)

## Placeholder Reference

### {{VERSION}}
Replaced with version string (e.g., "0.2.0")

### {{CORE_CONTENT}}
Replaced with entire contents of skill_core.md

## Future Enhancements

Potential improvements:
- [ ] Add version to skill_core.md front matter
- [ ] Extract reusable sections (e.g., CLI table, Python examples)
- [ ] Add linting to check for agent-specific content in core
- [ ] CI check to ensure generated skills are up-to-date
- [ ] Template variables for common patterns

## Maintenance Checklist

When updating skills:

- [ ] Edit skill_core.md for shared changes
- [ ] Edit wrappers for agent-specific changes
- [ ] Run generate_skills.py to regenerate
- [ ] Verify generated files with git diff
- [ ] Check line counts (should be similar)
- [ ] Test registration functions
- [ ] Commit all files (core, wrappers, generated)
- [ ] Update version if needed

## Questions?

See also:
- `skills/xorq/CLAUDE.md` - Claude Code skill maintenance guide
- `python/xorq/agent/resources/codex/README.md` - Codex skill installation
- `python/xorq/agent/onboarding.py` - Registration implementation

---

**Key Principle:** skill_core.md is the single source of truth. All xorq updates go here first.
