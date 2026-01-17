"""Dynamic workflow context for xorq agents.

Inspired by `bd prime` - provides AI-optimized workflow guidance as single source of truth.
Replaces scattered prompt files with dynamic, context-aware output.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def find_builds_dir() -> Path | None:
    """Find the builds/ directory in current project."""
    cwd = Path.cwd()
    builds = cwd / "builds"
    return builds if builds.exists() and builds.is_dir() else None


def get_recent_builds(limit: int = 3) -> list[str]:
    """Get most recently modified build directories."""
    builds_dir = find_builds_dir()
    if not builds_dir:
        return []

    # Get all build directories sorted by modification time
    build_dirs = [
        d for d in builds_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    build_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in build_dirs[:limit]]


def check_custom_prime() -> str | None:
    """Check for custom PRIME.md override in .xorq/ directory."""
    custom_paths = [
        Path(".xorq/PRIME.md"),
        Path("PRIME.md"),
    ]

    for path in custom_paths:
        if path.exists():
            return path.read_text(encoding="utf-8")

    return None


def render_prime_context() -> str:
    """Generate dynamic workflow context for xorq agents.

    This is the single source of truth for xorq workflow guidance.
    Replaces the need for scattered prompt files.
    """
    # Check for custom override first
    custom = check_custom_prime()
    if custom:
        return custom

    # Get project state
    recent_builds = get_recent_builds(limit=3)
    builds_status = _format_builds_status(recent_builds)

    context = dedent(f"""\
        # Xorq Workflow Context

        > **Single Source of Truth**: This command provides dynamic workflow guidance.
        > Run `xorq agent prime` after context compaction or at session start.

        {builds_status}

        # 🚨 SESSION CLOSE PROTOCOL 🚨

        **CRITICAL**: Before saying "done" or "complete", you MUST:

        ```
        [ ] 1. Build and catalog new expressions
        [ ] 2. Commit build manifests: git add builds/ && git commit
        [ ] 3. Push to remote: git push
        [ ] 4. Verify: git status (must show "up to date")
        ```

        **NEVER skip this.** Work is not done until pushed.

        ## Core Workflow

        ### 1. MANDATORY FIRST STEP - Check Schema

        ```python
        # ALWAYS run this before ANY operations:
        print(table.schema())
        # Then use EXACT column names from the output
        ```

        **Critical Rules:**
        - **Snowflake = UPPERCASE**: Use `_.CARAT`, `_.PRICE`, `_.COLOR`
        - **DuckDB/Postgres = lowercase**: Use `_.carat`, `_.price`, `_.color`
        - **ALWAYS match exact case from .schema() output**

        ### 2. Required Imports

        ```python
        import xorq.api as xo
        from xorq.api import _
        from xorq.vendor import ibis  # For desc/asc ordering
        ```

        ### 3. Build → Catalog → Run Pattern

        ```bash
        # 1. Build expression
        xorq build expr.py -e expr

        # 2. Catalog the build
        xorq catalog add builds/<hash> --alias my-pipeline

        # 3. Run when needed
        xorq run builds/<hash> -o output.parquet
        ```

        ### 4. Deferred Execution Rules

        **NEVER use pandas for data operations** - Use ibis expressions only:

        ```python
        # ❌ WRONG - Using pandas
        import pandas as pd
        df = table.execute()
        df_filtered = df[df['PRICE'] > 100]

        # ✅ RIGHT - Using ibis expressions
        filtered = table.filter(_.PRICE > 100)
        result = filtered.execute()  # Only at the end!
        ```

        **Common replacements:**
        - `pd.merge()` → `table1.join(table2, predicates)`
        - `df.groupby()` → `table.group_by()`
        - `df['new'] = ...` → `table.mutate(new=...)`
        - `df.sort_values()` → `table.order_by()`
        - `df.drop_duplicates()` → `table.distinct()`

        ### 5. Aggregation vs Selection

        ```python
        # Single-row statistics: use .aggregate()
        table.aggregate([_.PRICE.mean().name('avg_price')]).execute()  # → 1 row

        # Column selection: use .select()
        table.select(_.PRICE, _.CARAT).execute()  # → N rows

        # NEVER use .select() for aggregations (broadcasts to all rows!)
        ```

        ### 6. Ordering After Aggregation

        ```python
        from xorq.vendor import ibis  # ✅ CORRECT import

        grouped = table.group_by(_.COLOR).aggregate([
            _.count().name('count')
        ])

        # WRONG: grouped.order_by(_.count.desc())  # ❌ AttributeError!
        # RIGHT: grouped.order_by(ibis.desc('count'))  # ✅ Use string name
        ```

        ## 🔥 ML PREDICTIONS: ALWAYS USE STRUCT PATTERN! 🔥

        **When doing ML tasks, ALL predictions MUST use the struct pattern:**

        ```python
        test_predicted = (
            test
            .mutate(as_struct(name=ORIGINAL_ROW))
            .pipe(fitted_pipeline.predict)
            .unpack(ORIGINAL_ROW)
        )
        ```

        **Never use:** `fitted.predict(test[features])` - This breaks deferred execution!

        ## Essential Commands Reference

        **Building & Running:**
        - `xorq build expr.py -e expr` - Build expression to builds/<hash>/
        - `xorq run builds/<hash> -o output.parquet` - Execute build
        - `xorq lineage <alias>` - Show column lineage

        **Catalog Management:**
        - `xorq catalog ls` - List all registered builds
        - `xorq catalog add builds/<hash> --alias name` - Register with alias
        - `xorq catalog show <alias>` - View catalog entry

        **Agent Helpers:**
        - `xorq agent prime` - This command (workflow context)
        - `xorq agent templates list` - List available templates
        - `xorq agent onboard` - Onboarding guide

        ## Customization

        Create `.xorq/PRIME.md` to override this default output with project-specific guidance.
        """)

    return context.rstrip() + "\n"


def _format_builds_status(recent_builds: list[str]) -> str:
    """Format the current builds status section."""
    if not recent_builds:
        return dedent("""\
            ## Current Project State

            - No builds found in builds/ directory
            - Run `xorq build expr.py -e expr` to create your first build
            """).strip()

    builds_list = "\n".join(f"- builds/{b}/" for b in recent_builds)
    count_msg = (
        f"{len(recent_builds)} most recent" if len(recent_builds) >= 3 else "Recent"
    )

    return dedent(f"""\
        ## Current Project State

        {count_msg} builds:
        {builds_list}

        Run `xorq catalog ls` to see registered aliases.
        """).strip()


def agent_prime_command(args=None, export: bool = False) -> None:
    """Execute the prime command - output workflow context."""
    # Check if we're in a xorq project (has builds/ or can create it)
    # For now, just output the context - don't require specific setup

    # If export flag (for --export mode in future), ignore custom override
    if export:
        # Could implement this to export default template
        pass

    context = render_prime_context()
    print(context, end="")


if __name__ == "__main__":
    # Allow running as standalone script
    agent_prime_command()
