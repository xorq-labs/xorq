## LLM SESSION TRACKING OBJECT

You have access to a special `llm` object that tracks all LLM sessions and turns. Use this to understand task progress and make informed decisions about continuation.

### Available Properties

**Session Access:**
- `llm.current` - Current active session (if any)
- `llm.last` - Most recently completed session
- `llm.history` - List of all completed sessions
- `llm.turns` - Turns from current session

**Quick Access to Recent Activity:**
- `llm.last_turn` - The most recent turn (prompt/response/code/results)
- `llm.last_code` - Code blocks from the last turn
- `llm.last_result` - Execution results from the last turn

### Useful Methods for Task Evaluation

```python
# Check how many turns have been executed
len(llm.turns)  # Number of turns in current session

# Find turns that had errors
error_turns = llm.find_turns_with_errors()
if error_turns:
    print(f"Found {len(error_turns)} turns with errors")

# Get all code generated so far
all_code = llm.get_all_code()
print(f"Generated {len(all_code)} code blocks so far")

# Get all execution results
all_results = llm.get_all_results()

# Check session statistics
stats = llm.session_stats()
print(f"Current session: {stats['current_turns']} turns, {stats['current_code_blocks']} code blocks")

# Inspect session details
print(llm.inspect())  # Pretty-printed session information
```

### Using LLM Object for Task Reflection

When evaluating whether to continue or complete a task, you can examine the session history:

```python
# Example: Check if we've been repeating similar errors
if llm.current:
    error_count = len([t for t in llm.turns if t.has_error])
    if error_count > 3:
        print(f"Multiple errors encountered ({error_count} turns with errors)")
        # Consider alternative approach or stopping

# Example: Check if we've generated enough code
if llm.current and llm.current.total_code_blocks >= 5:
    print(f"Already generated {llm.current.total_code_blocks} code blocks")
    # Evaluate if more code is really needed

# Example: Check execution duration
if llm.current:
    duration = llm.current.duration
    if duration > 60:
        print(f"Session has been running for {duration:.1f} seconds")
        # Consider wrapping up
```

### Turn Object Structure

Each turn in `llm.turns` contains:
- `turn.index` - Turn number (0-based)
- `turn.prompt` - The prompt sent to the LLM
- `turn.response` - The LLM's response
- `turn.code_blocks` - List of code blocks generated
- `turn.results` - List of execution results
- `turn.has_error` - Boolean indicating if execution had errors
- `turn.has_code` - Boolean indicating if code was generated
- `turn.success` - Boolean indicating successful code execution

### IMPORTANT: Use for Informed Decisions

The `llm` object helps you make better decisions about:
1. Whether the original task has been completed
2. If you're making progress or stuck in a loop
3. Whether errors are preventing task completion
4. How much work has already been done

Always consider the session history when deciding whether to continue or mark a task as complete.