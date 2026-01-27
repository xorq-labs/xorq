# Xorq Claude Code Hooks

Simple hook system for Claude Code lifecycle events.

## Available Hooks

- **session_start.py** - **Onboarding context** - Runs `xorq agents onboard` at session start
- **user_prompt_submit.py** - Triggered when user submits a prompt
- **PreToolUse (prompt-based)** - **Deferred execution guard** - Blocks eager pandas/visualization operations
- **post_tool_use_failure.py** - **Troubleshooting assistant** - Detects xorq errors and provides guidance
- **pre_compact.py** - Triggered before context compaction
- **stop.py** - **Catalog checker** - Warns about uncataloged builds when stopping
- **session_end.py** - Triggered when a Claude Code session ends

### SessionStart Hook (Onboarding Context)

Automatically runs `xorq agents onboard` when a Claude Code session begins and injects the onboarding content as context. This provides lean workflow guidance without cluttering the codebase.

**Benefits:**
- Dynamic, context-aware workflow guidance
- No need for large AGENTS.md files
- Always up-to-date instructions

### PreToolUse Hook (Deferred Execution Guard)

The PreToolUse hook uses a **prompt-based evaluation** (not a command script) to detect and block problematic eager patterns that violate xorq's deferred execution principle.

**Blocks:**
1. **Eager pandas with processing**: `.to_pandas()` followed by pandas operations (df[...], df.groupby(), etc.)
2. **Eager execute with processing**: `.execute()` followed by result manipulation
3. **Inline visualization**: `plt.`, `sns.`, `plotly.`, `.plot()`, `.show()`, `.savefig()` in expression code

**Allows:**
- Expression execution via `xorq run <alias>`
- Piping results: `.execute()` or `xorq run` when output is piped/saved without further processing
- Storing results if not further processed

When violations are detected, Claude is blocked and receives guidance to:
- Build expressions and run them separately
- Use `xorq run` to execute and pipe output
- Avoid eager pandas operations in expression definitions
- Get help with `xorq agents prime`

### PostToolUseFailure Hook (Troubleshooting Assistant)

The PostToolUseFailure hook analyzes tool failures to detect xorq-related errors and provides contextual troubleshooting guidance.

**Detects:**
- Type coercion errors (suggests `.into_backend()`)
- Expression not found in catalog
- Build failures
- File/directory not found errors
- Import errors
- Catalog operation failures
- Deferred execution issues

**Provides guidance for:**
- Fixing type coercion errors with `.into_backend()`
- Listing and cataloging expressions
- Building expressions correctly
- Using vignette templates
- Debugging common xorq workflows
- Following deferred execution patterns

When xorq errors are detected, Claude receives specific troubleshooting steps and relevant commands to resolve the issue.

### Stop Hook (Catalog Checker)

Checks for uncataloged builds in `.xorq/builds/` when Claude stops working. If uncataloged builds are found, provides a helpful reminder to catalog them AND commit to git.

**Benefits:**
- Ensures builds don't get lost
- Reminds users to catalog their work
- Enforces git commit workflow for builds and catalog
- Shows exact commands to complete the workflow

**Checks:**
- Lists all builds in `.xorq/builds/`
- Compares with `xorq catalog ls` output
- Warns about any uncataloged builds (up to 5)
- Provides 2-step workflow: catalog → git commit

**Required workflow:**
1. Catalog builds: `xorq catalog add .xorq/builds/<hash> --alias <name>`
2. Commit to git: `git add .xorq/builds/ .xorq/catalog.yaml && git commit`

⚠️ Work is not done until builds are cataloged AND committed!

## Usage

These are currently dummy hooks with placeholder implementations.
Each hook can be extended to add custom logic for the respective event.

## Installation

Copy hooks to your Claude project:

```bash
cp python/xorq/claude_hooks/*.py .claude/hooks/
```

Configure in `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/session_start.py"
      }]
    }],
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/user_prompt_submit.py"
      }]
    }],
    "PreToolUse": [{
      "hooks": [{
        "type": "command",
        "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/pre_tool_use.py"
      }]
    }],
    "PostToolUseFailure": [{
      "hooks": [{
        "type": "command",
        "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/post_tool_use_failure.py"
      }]
    }],
    "PreCompact": [{
      "hooks": [{
        "type": "command",
        "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/pre_compact.py"
      }]
    }],
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/stop.py"
      }]
    }],
    "SessionEnd": [{
      "hooks": [{
        "type": "command",
        "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/session_end.py"
      }]
    }]
  }
}
```
