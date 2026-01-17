"""Enhanced prime command with task-aware guidance.

Detects ML tasks and provides targeted prompts automatically.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from xorq.agent.prime import (
    _format_builds_status,
    check_custom_prime,
    get_recent_builds,
)


def detect_task_type() -> str | None:
    """Detect likely task type from recent files and builds.

    Returns: "ml_regression", "ml_classification", "etl", "viz", or None
    """
    # Check scripts directory for ML-related files
    scripts_dir = Path("scripts")
    if scripts_dir.exists():
        for script in scripts_dir.glob("*.py"):
            content_lower = script.read_text().lower()
            if "sklearn" in content_lower or "pipeline" in content_lower:
                if "regression" in content_lower or "predict" in content_lower:
                    return "ml_regression"
                elif "classification" in content_lower or "classifier" in content_lower:
                    return "ml_classification"

    # Check for examples directory
    examples = Path("examples")
    if examples.exists():
        for example in examples.glob("*prediction*.py"):
            return "ml_regression"
        for example in examples.glob("*classification*.py"):
            return "ml_classification"

    return None


def get_ml_guidance(task_type: str) -> str:
    """Get ML-specific guidance based on task type."""
    if task_type not in ("ml_regression", "ml_classification"):
        return ""

    guidance = dedent("""\
        ## 🎯 ML Task Detected

        **CRITICAL REMINDERS for sklearn models:**

        ### 1. Use Struct Pattern (MANDATORY)
        ```python
        predictions = (
            table
            .mutate(as_struct(name="original_row"))  # Pack ALL columns
            .pipe(fitted.predict)
            .drop("target")
            .unpack("original_row")                  # Unpack ALL columns
            .mutate(predicted=_.predicted)
            .drop("predicted")
        )
        ```

        ### 2. Float64 for All Features (MANDATORY)
        ```python
        # Categorical encodings MUST be float64
        cut_score = (_.cut.case()
            .when("Fair", 0.0)   # Use 0.0 not 0
            .when("Good", 1.0)
            .end()
            .cast("float64"))    # REQUIRED!
        ```

        ### 3. Feature Columns (Numeric Only)
        ```python
        # ✅ RIGHT: Only numeric features
        feature_columns = ["carat", "cut_score", "color_score"]

        # ❌ WRONG: Including raw categoricals
        feature_columns = ["carat", "cut", "color"]  # cut/color are strings!
        ```

        ### 4. Check Examples First
        ```bash
        # Look for similar patterns
        find examples/ -name "*prediction*" -o -name "*ml*"
        ls examples/diamonds_price_prediction.py  # Reference implementation
        ```

        ### 5. Use Templates
        ```bash
        xorq agent templates scaffold sklearn_pipeline
        ```

        **Reference Documentation:**
        - Struct pattern: python/xorq/agent/context_blocks/ml_struct_pattern.md
        - Type requirements: python/xorq/agent/context_blocks/sklearn_type_requirements.md
        - Task planning: python/xorq/agent/prompts/ml_task_planning.md
        - Working example: examples/diamonds_price_prediction.py

        **Common Errors & Fixes:**
        - "Duplicate column" → Remove categoricals from feature_columns
        - "Type coercion failed" → Cast all features to float64
        - "Cannot add Field" → Use struct pattern, don't manual join

        """)

    return guidance.rstrip() + "\n\n"


def render_enhanced_prime_context(task_type: str | None = None) -> str:
    """Generate enhanced workflow context with task-specific guidance."""
    # Check for custom override first
    custom = check_custom_prime()
    if custom:
        return custom

    # Auto-detect task if not specified
    if task_type is None:
        task_type = detect_task_type()

    # Get project state
    recent_builds = get_recent_builds(limit=3)
    builds_status = _format_builds_status(recent_builds)

    # Get task-specific guidance
    task_guidance = get_ml_guidance(task_type) if task_type else ""

    context = dedent(f"""\
        # Xorq Workflow Context

        > **Single Source of Truth**: This command provides dynamic workflow guidance.
        > Run `xorq agent prime` after context compaction or at session start.

        {builds_status}

        {task_guidance}# 🚨 SESSION CLOSE PROTOCOL 🚨

        **CRITICAL**: Before saying "done" or "complete", you MUST:

        ```
        [ ] 1. Build and catalog new expressions
        [ ] 2. Validate builds run: xorq run <alias> --limit 10
        [ ] 3. Commit build manifests: git add builds/ && git commit
        [ ] 4. Push to remote: git push
        [ ] 5. Verify: git status (must show "up to date")
        ```

        **NEVER skip this.** Work is not done until pushed and validated.

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

        # 2. Catalog the build with an alias
        xorq catalog add builds/<hash> --alias my-pipeline

        # 3. View catalog (shows aliases, revisions, root tags)
        xorq catalog ls

        # 4. Run by alias or specific revision
        xorq run my-pipeline -o output.parquet
        xorq run my-pipeline@r2 -o output.parquet  # Run specific revision
        ```

        **Catalog Root Tags:**
        - The catalog automatically extracts and displays the root tag from each build
        - Root tags identify the top-level expression node (e.g., table name from Tag nodes)
        - Use `xorq catalog ls` to see all aliases with their root tags
        - This helps identify what each catalog entry produces

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

        ## Essential Commands Reference

        **Building & Running:**
        - `xorq build expr.py -e expr` - Build expression to builds/<hash>/
        - `xorq run <alias> -o output.parquet` - Execute by catalog alias
        - `xorq run <alias>@r2 -o output.parquet` - Execute specific revision
        - `xorq run builds/<hash> -o output.parquet` - Execute by build hash
        - `xorq lineage <alias>` - Show column lineage with root tag

        **Catalog Management:**
        - `xorq catalog ls` - List aliases with revisions and root tags
        - `xorq catalog add builds/<hash> --alias name` - Register with alias
        - `xorq catalog info` - Show catalog metadata

        **Agent Helpers:**
        - `xorq agent prime` - This command (workflow context)
        - `xorq agent prime --task ml` - ML-specific guidance
        - `xorq agent templates list` - List available templates
        - `xorq agent onboard` - Onboarding guide

        ## Customization

        Create `.xorq/PRIME.md` to override this default output with project-specific guidance.
        """)

    return context.rstrip() + "\n"


def agent_prime_command_enhanced(task: str | None = None) -> None:
    """Execute enhanced prime command with optional task type."""
    context = render_enhanced_prime_context(task_type=task)
    print(context, end="")


if __name__ == "__main__":
    # Test with task detection
    agent_prime_command_enhanced()
