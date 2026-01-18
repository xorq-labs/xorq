"""Dynamic workflow context for xorq agents.

Inspired by `bd prime` - provides AI-optimized workflow guidance as single source of truth.
Replaces scattered prompt files with dynamic, context-aware output.

Includes task-aware guidance that detects ML tasks and provides targeted prompts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent


def find_builds_dir() -> Path | None:
    """Find the builds/ directory in current project."""
    cwd = Path.cwd()
    # Check both .xorq/builds and builds/
    xorq_builds = cwd / ".xorq" / "builds"
    if xorq_builds.exists() and xorq_builds.is_dir():
        return xorq_builds

    builds = cwd / "builds"
    return builds if builds.exists() and builds.is_dir() else None


def get_recent_builds(limit: int = 5) -> list[tuple[str, str]]:
    """Get most recently modified build directories with timestamps.

    Returns: List of (hash, relative_time) tuples
    """
    builds_dir = find_builds_dir()
    if not builds_dir:
        return []

    # Get all build directories sorted by modification time
    build_dirs = [
        d for d in builds_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    build_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    results = []
    for d in build_dirs[:limit]:
        # Get relative time (e.g., "2 hours ago")
        try:
            mtime = d.stat().st_mtime
            import time

            seconds_ago = time.time() - mtime
            if seconds_ago < 60:
                time_str = "just now"
            elif seconds_ago < 3600:
                time_str = f"{int(seconds_ago / 60)}m ago"
            elif seconds_ago < 86400:
                time_str = f"{int(seconds_ago / 3600)}h ago"
            else:
                time_str = f"{int(seconds_ago / 86400)}d ago"
            results.append((d.name, time_str))
        except Exception:
            results.append((d.name, "unknown"))

    return results


def get_catalog_entries(limit: int = 10) -> list[dict]:
    """Get recent catalog entries using xorq catalog ls.

    Returns: List of dicts with alias, revision, hash, root_tag
    """
    try:
        result = subprocess.run(
            ["xorq", "catalog", "ls"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return []

        # Parse the output (skip header lines)
        lines = result.stdout.strip().split("\n")
        entries = []

        for line in lines[2:]:  # Skip header and separator
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                alias = parts[0]
                revision = parts[1]
                build_hash = parts[2] if len(parts) > 2 else ""
                root_tag = parts[3] if len(parts) > 3 else ""
                entries.append(
                    {
                        "alias": alias,
                        "revision": revision,
                        "hash": build_hash,
                        "root_tag": root_tag,
                    }
                )

        return entries[:limit]
    except Exception:
        return []


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


def detect_task_type() -> str | None:
    """Detect likely task type from recent files and builds.

    Returns: "ml_regression", "ml_classification", "etl", "viz", or None
    """
    # Check scripts directory for ML-related files
    scripts_dir = Path("scripts")
    if scripts_dir.exists():
        for script in scripts_dir.glob("*.py"):
            try:
                content_lower = script.read_text().lower()
                if "sklearn" in content_lower or "pipeline" in content_lower:
                    if "regression" in content_lower or "predict" in content_lower:
                        return "ml_regression"
                    elif (
                        "classification" in content_lower
                        or "classifier" in content_lower
                    ):
                        return "ml_classification"
            except Exception:
                continue

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

        **PREFERRED: Pipeline API with as_struct Pattern**

        ### 1. Create as_struct Helper
        ```python
        import toolz

        @toolz.curry
        def as_struct(expr, name=None):
            \"\"\"Pack all columns into a struct.\"\"\"
            struct = xo.struct({column: expr[column] for column in expr.columns})
            if name:
                struct = struct.name(name)
            return struct
        ```

        ### 2. Create and Fit sklearn Pipeline
        ```python
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestRegressor
        from xorq.expr.ml.pipeline_lib import Pipeline

        sklearn_pipeline = SkPipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=6))
        ])
        xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

        fitted = xorq_pipeline.fit(
            train,
            features=FEATURE_COLUMNS,  # Tuple of feature names
            target="target_column",
        )
        ```

        ### 3. Predict with Struct Pattern (MANDATORY)
        ```python
        # MUST use struct pattern to preserve all columns
        predictions = (
            test
            .mutate(as_struct(name="original_row"))  # 1. Pack ALL columns
            .pipe(fitted.predict)                     # 2. Predict
            .drop("target_column")                    # 3. Remove target if present
            .unpack("original_row")                   # 4. Unpack original columns
            .mutate(predicted_price=_.predicted)      # 5. Rename prediction
            .drop("predicted")                        # 6. Clean up
        )
        ```

        ### 4. Float64 for All Features (MANDATORY)
        ```python
        # Cast categorical encodings to float64
        cut_score = (_.cut.case()
            .when("Fair", 0.0)   # Use 0.0 not 0
            .when("Good", 1.0)
            .end()
            .cast("float64"))    # REQUIRED!
        ```

        **Working Examples:**
        - examples/diamonds_price_prediction.py - Pipeline API (PREFERRED)
        - examples/pipeline_example.py - Pipeline with SelectKBest

        **Common Errors & Fixes:**
        - "Duplicate column" → Remove categorical columns from feature_columns
        - "Type coercion failed" → Cast all features to float64
        - "Cannot add Field" → Use struct pattern (.mutate(as_struct) + .pipe + .unpack)

        """)

    return guidance.rstrip() + "\n\n"


def render_prime_context(task_type: str | None = None) -> str:
    """Generate dynamic workflow context for xorq agents.

    This is the single source of truth for xorq workflow guidance.
    Replaces the need for scattered prompt files.

    Args:
        task_type: Optional task type override ("ml_regression", "ml_classification", etc.)
                   If None, will auto-detect from project files.
    """
    # Check for custom override first
    custom = check_custom_prime()
    if custom:
        # Get project state for injection
        recent_builds = get_recent_builds(limit=5)
        catalog_entries = get_catalog_entries(limit=10)
        builds_status = _format_builds_status(recent_builds, catalog_entries)

        # Inject dynamic state into custom PRIME.md
        if "**Recent activity**" in custom:
            custom = custom.replace(
                "**Recent activity** is displayed dynamically when you run `xorq agent prime`.",
                builds_status,
            )
        return custom

    # Auto-detect task if not specified
    if task_type is None:
        task_type = detect_task_type()

    # Get project state (always needed)
    recent_builds = get_recent_builds(limit=5)
    catalog_entries = get_catalog_entries(limit=10)
    builds_status = _format_builds_status(recent_builds, catalog_entries)

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
        [ ] 3. Commit catalog and manifests: git add .xorq/catalog.yaml builds/ && git commit
        [ ] 4. Push to remote: git push
        [ ] 5. Verify: git status (must show "up to date")
        ```

        **NEVER skip this.** Work is not done until pushed and validated.
        **The catalog file (.xorq/catalog.yaml) MUST be committed** to preserve aliases and revisions.

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


def _format_builds_status(
    recent_builds: list[tuple[str, str]], catalog_entries: list[dict]
) -> str:
    """Format the current builds status section."""
    if not recent_builds and not catalog_entries:
        return dedent("""\
            ## Current Project State

            - No builds or catalog entries found
            - Run `xorq build expr.py -e expr` to create your first build
            - Then: `xorq catalog add builds/<hash> --alias my-pipeline`
            """).strip()

    output = "## Current Project State\n\n"

    # Recent builds
    if recent_builds:
        output += "**Recent Builds:**\n"
        for build_hash, time_str in recent_builds[:5]:
            output += f"- `{build_hash[:12]}...` ({time_str})\n"
        output += "\n"

    # Catalog entries
    if catalog_entries:
        output += "**Cataloged Pipelines:**\n"
        for entry in catalog_entries[:10]:
            alias = entry.get("alias", "")
            revision = entry.get("revision", "")
            root_tag = entry.get("root_tag", "")
            if root_tag:
                output += f"- `{alias}` @ {revision} → {root_tag}\n"
            else:
                output += f"- `{alias}` @ {revision}\n"
        output += "\n"
        output += "Run `xorq catalog ls` for full list.\n"
    else:
        output += "**No catalog entries yet.**\n"
        output += "- Catalog builds: `xorq catalog add builds/<hash> --alias name`\n"

    return output.strip()


def agent_prime_command(task: str | None = None, export: bool = False) -> None:
    """Execute the prime command - output workflow context.

    Args:
        task: Optional task type ("ml", "ml_regression", "ml_classification", etc.)
        export: If True, export default template (future feature)
    """
    # Map simple task names to full types
    task_map = {
        "ml": "ml_regression",  # Default ML to regression
        "regression": "ml_regression",
        "classification": "ml_classification",
    }
    task_type = task_map.get(task, task) if task else None

    # If export flag (for --export mode in future), ignore custom override
    if export:
        # Could implement this to export default template
        pass

    context = render_prime_context(task_type=task_type)
    print(context, end="")


if __name__ == "__main__":
    # Allow running as standalone script
    agent_prime_command()
