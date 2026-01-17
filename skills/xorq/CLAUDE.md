# Maintaining the Xorq Claude Code Skill

This guide is for maintaining and updating the xorq skill for Claude Code.

## Skill Structure

```
skills/xorq/
├── SKILL.md         # Primary skill file (Claude reads this)
├── README.md        # Human documentation
├── CLAUDE.md        # This file (maintenance guide)
└── resources/       # Optional detailed docs
    ├── CLI_REFERENCE.md
    ├── WORKFLOWS.md
    ├── TROUBLESHOOTING.md
    └── PATTERNS.md
```

## What Claude Reads

Claude reads **SKILL.md** when the skill is invoked. It contains:

1. **Front matter (YAML)** - Metadata about the skill
2. **Core content** - The actual guidance for using xorq

Keep SKILL.md focused and concise. Move detailed documentation to `resources/`.

## Updating the Skill

### When xorq CLI Changes

1. **Update CLI reference** in SKILL.md
2. **Test new commands** in the Session Protocol section
3. **Add new patterns** if workflow changes
4. **Update version** in front matter

### When Agent Features Change

1. **Update agent commands** section
2. **Refresh prompt list** from `xorq agent prompt list`
3. **Update skills list** from `xorq agent templates list`
4. **Add new onboarding steps** if workflow changes

### When Adding Resources

1. Create file in `resources/` directory
2. Add link in "Resources" table in SKILL.md
3. Update README.md file structure section
4. Keep resources focused on specific topics

## Testing the Skill

### Manual Testing

1. **Copy to Claude Code skills directory**:
   ```bash
   cp -r skills/xorq ~/.claude/skills/
   ```

2. **Start Claude Code** and ask:
   - "How do I build a xorq expression?"
   - "Show me how to use xorq catalog"
   - "What's the xorq session protocol?"

3. **Verify Claude**:
   - References SKILL.md content
   - Uses correct command syntax
   - Follows session protocol
   - Mentions agent features

### Integration Testing

Test with real workflows:

```bash
# Initialize project
xorq init -t penguins

# Ask Claude to:
# - Build the expression
# - Catalog the result
# - Run the build
# - Show lineage

# Verify Claude uses:
# - Correct xorq commands
# - Agent prompts (xorq agent prompt)
# - Skills (xorq agent templates)
```

## Syncing with xorq Updates

### Regular Maintenance

1. **Check xorq releases**: Monitor GitHub releases
2. **Test new features**: Try new commands locally
3. **Update skill**: Add new features to SKILL.md
4. **Bump version**: Update version in front matter
5. **Test with Claude**: Verify skill still works

### Breaking Changes

If xorq has breaking changes:

1. **Update Session Protocol** with new workflow
2. **Mark deprecated commands** in CLI Reference
3. **Add migration notes** in SKILL.md
4. **Update examples** to use new syntax
5. **Test thoroughly** with Claude Code

## Best Practices

### Keep It Focused

- SKILL.md should be concise (Claude reads this directly)
- Move detailed docs to resources/
- Use tables for quick reference
- Include code examples inline

### Make It Actionable

- Provide concrete commands
- Show real workflow patterns
- Include "Session Protocol" for step-by-step guidance
- Link to agent prompts/skills

### Stay Current

- Sync with xorq documentation
- Test commands before adding
- Update examples to match current syntax
- Remove deprecated features

## Common Update Scenarios

### Adding a New Command

1. Test the command locally:
   ```bash
   xorq new-command --help
   ```

2. Add to CLI Reference section in SKILL.md:
   ```markdown
   | Command | Purpose |
   |---------|---------|
   | `xorq new-command` | Description |
   ```

3. If it's a core workflow command, add to Session Protocol

### Adding a New Agent Prompt

1. Verify it exists:
   ```bash
   xorq agent prompt show new-prompt
   ```

2. Add to Agent Commands section in SKILL.md

3. If it's commonly used, add example usage

### Adding a New Skill

1. Check it's registered:
   ```bash
   xorq agent templates list
   ```

2. Add to Skills section in SKILL.md

3. Consider adding a workflow example in README.md

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-01-16 | Initial xorq skill for Claude Code |

## Future Enhancements

Potential additions:

- [ ] CLI_REFERENCE.md - Complete command documentation
- [ ] WORKFLOWS.md - Step-by-step patterns
- [ ] TROUBLESHOOTING.md - Common error fixes
- [ ] PATTERNS.md - Best practices and recipes
- [ ] Integration examples with bd (beads)
- [ ] Multi-engine workflow patterns
- [ ] ML pipeline examples
- [ ] Serving patterns with Arrow Flight

## Contact

Questions or suggestions:
- Open an issue at [github.com/xorq-labs/xorq](https://github.com/xorq-labs/xorq)
- Check [docs.xorq.dev](https://docs.xorq.dev) for latest documentation
