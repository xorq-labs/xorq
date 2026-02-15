# Xorq Claude Code Hooks

Simple hook system for Claude Code lifecycle events.

## Available Hooks

- **session_start.py** - Triggered when a Claude Code session begins
- **user_prompt_submit.py** - Triggered when user submits a prompt
- **post_tool_use.py** - Injects xorq onboarding instructions after every tool use
- **post_tool_use_failure.py** - Provides troubleshooting guidance on tool failures
- **pre_compact.py** - Triggered before context compaction
- **stop.py** - Triggered when Claude Code execution is stopped
- **session_end.py** - Triggered when a Claude Code session ends

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
    "PostToolUse": [{
      "hooks": [{
        "type": "command",
        "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/post_tool_use.py"
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
