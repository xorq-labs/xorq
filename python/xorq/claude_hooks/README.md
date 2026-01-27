# Xorq Claude Code Hooks

Simple hook system for Claude Code lifecycle events.

## Available Hooks

- **session_start.py** - Triggered when a Claude Code session begins
- **user_prompt_submit.py** - Triggered when user submits a prompt
- **PreToolUse (prompt-based)** - **Deferred execution guard** - Blocks eager pandas/visualization operations
- **pre_compact.py** - Triggered before context compaction
- **stop.py** - Triggered when Claude Code execution is stopped
- **session_end.py** - Triggered when a Claude Code session ends

### PreToolUse Hook (Deferred Execution Guard)

The PreToolUse hook uses a **prompt-based evaluation** (not a command script) to detect and block eager operations that violate xorq's deferred execution principle.

**Blocks:**
- `.to_pandas()` - Eager pandas execution
- `.execute()` - Eager ibis execution
- `plt.`, `sns.`, `plotly.` - Visualization libraries
- `.plot()`, `.show()`, `.savefig()` - Plotting methods

When violations are detected, Claude is blocked and receives guidance to use:
- `xorq agents prime` - Get workflow context
- `xorq agents vignette list` - Find deferred patterns

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
