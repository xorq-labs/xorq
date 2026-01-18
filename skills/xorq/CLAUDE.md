# Maintaining the Xorq Claude Code Skill

This guide is for maintaining and updating the xorq skill for Claude Code.

## Skill Structure

```
skills/xorq/
├── SKILL.md              # Primary skill file (~400 lines)
├── README.md             # Human documentation
├── CLAUDE.md             # This file (maintenance guide)
├── skill-rules.json      # Auto-activation triggers
└── resources/            # Progressive disclosure (deep dives)
    ├── expression-api.md      # Expression building patterns
    ├── ml-pipelines.md        # ML/sklearn integration
    ├── caching.md             # Performance optimization
    ├── udf-udxf.md            # UDFs and Flight servers
    ├── examples.md            # End-to-end examples
    ├── CLI_REFERENCE.md       # Complete CLI docs
    ├── WORKFLOWS.md           # Step-by-step patterns
    ├── TROUBLESHOOTING.md     # Common issues
    └── PATTERNS.md            # Best practices
```

## What Claude Reads

Claude reads **SKILL.md** when the skill is invoked. It contains:

1. **Front matter (YAML)** - Metadata (name, description, version)
2. **Core content** - Essential guidance for using xorq (400 lines)
3. **Resource links** - Progressive disclosure to detailed docs

**Progressive Disclosure Pattern:**
- Keep SKILL.md under 500 lines (follows Claude Code infrastructure pattern)
- Move detailed documentation to `resources/`
- Claude loads resources only when needed

## Auto-Activation System

### skill-rules.json

Defines when the xorq skill should auto-activate:

- **Prompt triggers**: Keywords + intent patterns (regex)
- **File triggers**: Path patterns + content patterns

When users run `xorq agent onboard`, this file is automatically copied to:
- `~/.claude/skills/skill-rules.json`

The system merges with existing skills, so it won't overwrite other skill configurations.

### How It Works

1. User asks: "How do I build a data pipeline?"
2. Claude Code's `skill-activation-prompt` hook runs
3. Matches "data pipeline" keyword in skill-rules.json
4. Suggests xorq skill to Claude
5. Claude loads SKILL.md and responds with xorq context

## Updating the Skill

### When xorq CLI Changes

1. **Update CLI reference** in SKILL.md essential commands table
2. **Test new commands** in Session Protocol section
3. **Add detailed docs** to resources/CLI_REFERENCE.md
4. **Update version** in front matter

### When Python API Changes

1. **Update core examples** in SKILL.md Python API section
2. **Add detailed patterns** to resources/expression-api.md
3. **Update ML patterns** in resources/ml-pipelines.md if relevant
4. **Test examples** to ensure they work

### When Agent Features Change

1. **Update agent commands** section in SKILL.md
2. **Refresh prompt list** from `xorq agent prompt list`
3. **Update templates** from `xorq agent templates list`
4. **Update onboarding steps** if workflow changes

### When Adding Resources

1. Create file in `resources/` directory (keep under 500 lines per file)
2. Add link in "Resources" table in SKILL.md
3. Update this CLAUDE.md file structure section
4. Keep resources focused on specific topics

## Testing the Skill

### Manual Testing

1. **Copy to Claude Code skills directory**:
   ```bash
   # This happens automatically via xorq agent onboard
   # But you can test manually:
   cp -r skills/xorq ~/.claude/skills/
   ```

2. **Test auto-activation**:
   - Ask Claude: "How do I build a data pipeline?"
   - Should see skill suggestion

3. **Verify Claude**:
   - References SKILL.md content
   - Uses correct command syntax
   - Follows session protocol
   - Mentions agent features

### Integration Testing

Test with real workflows:

```bash
# Initialize project (installs skill)
xorq init -t penguins

# Verify skill installed
ls ~/.claude/skills/xorq/

# Verify skill-rules.json
cat ~/.claude/skills/skill-rules.json | jq '.skills.xorq'

# Test in Claude Code:
# - Ask "help with xorq catalog"
# - Edit expr.py file (should trigger skill)
# - Ask "how to check schema"
```

## Syncing with xorq Updates

### Regular Maintenance

1. **Check xorq releases**: Monitor GitHub releases
2. **Test new features**: Try new commands locally
3. **Update skill**: Add new features to SKILL.md
4. **Update resources**: Add detailed docs to resources/
5. **Bump version**: Update version in front matter
6. **Test with Claude**: Verify skill still works

### Breaking Changes

If xorq has breaking changes:

1. **Update Session Protocol** with new workflow
2. **Mark deprecated commands** in CLI Reference
3. **Add migration notes** in SKILL.md
4. **Update examples** to use new syntax
5. **Test thoroughly** with Claude Code

## Best Practices

### Keep SKILL.md Focused (500-line rule)

- Main file should be concise
- Move detailed docs to resources/
- Use tables for quick reference
- Include essential code examples inline
- Link to resources for deep dives

### Make It Actionable

- Provide concrete commands
- Show real workflow patterns
- Include "Session Protocol" for step-by-step
- Link to agent prompts/templates

### Stay Current

- Sync with xorq documentation
- Test commands before adding
- Update examples to match current syntax
- Remove deprecated features

### Progressive Disclosure

- SKILL.md: Overview + essentials + quick reference
- Resources: Deep dives, detailed examples, troubleshooting
- Each resource file: <500 lines, focused topic

## Auto-Activation Triggers

### When to Update skill-rules.json

Add triggers when:
- New xorq CLI commands added
- New Python API patterns introduced
- New file patterns used in projects
- Common user questions identified

### Trigger Types

**Prompt triggers:**
- Keywords: Literal strings in user prompts
- Intent patterns: Regex for flexible matching

**File triggers:**
- Path patterns: Glob patterns (e.g., `**/*expr*.py`)
- Content patterns: Regex in file content

**Example:**
```json
{
  "promptTriggers": {
    "keywords": ["xorq build", "catalog"],
    "intentPatterns": ["(build|create).*?pipeline"]
  },
  "fileTriggers": {
    "pathPatterns": ["**/*expr*.py"],
    "contentPatterns": ["import xorq", "xo\\.connect"]
  }
}
```

## Common Update Scenarios

### Adding a New Command

1. Test the command locally:
   ```bash
   xorq new-command --help
   ```

2. Add to Essential CLI Commands table in SKILL.md

3. Add detailed docs to resources/CLI_REFERENCE.md

4. If commonly used, add keyword to skill-rules.json

### Adding a New Python Pattern

1. Test the pattern in a real script

2. Add concise example to SKILL.md (if essential)

3. Add detailed guide to resources/expression-api.md or resources/PATTERNS.md

4. Add content pattern to skill-rules.json if unique

### Adding a New Agent Prompt/Template

1. Verify it exists:
   ```bash
   xorq agent prompt show new-prompt
   xorq agent templates show new-template
   ```

2. Add to Agent-Native Features section in SKILL.md

3. Add keyword to skill-rules.json if commonly referenced

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-01-16 | Initial xorq skill for Claude Code |
| 0.2.0 | 2026-01-18 | Consolidated with xorq-ibis, added auto-activation, progressive disclosure |

## Auto-Installation

The skill is automatically installed when users run:

```bash
xorq agent onboard
# Or
xorq init -t <template>
```

This:
1. Copies `skills/xorq/` to `~/.claude/skills/xorq/`
2. Sets up `~/.claude/skills/skill-rules.json` (merges with existing)
3. Makes skill auto-activate on relevant prompts/files

## Future Enhancements

Potential additions:

- [ ] Video examples in resources/
- [ ] Multi-engine workflow patterns
- [ ] Serving patterns with Arrow Flight
- [ ] Performance benchmarking guides
- [ ] Integration patterns (Airflow, Prefect, etc.)

## Contact

Questions or suggestions:
- Open an issue at [github.com/xorq-labs/xorq](https://github.com/xorq-labs/xorq)
- Check [docs.xorq.dev](https://docs.xorq.dev) for latest documentation

---

**Key Principle:** SKILL.md is the source of truth for Claude. Keep it focused, actionable, and under 500 lines. Use resources for deep dives.
