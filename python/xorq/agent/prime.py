"""Dynamic workflow context for xorq agents.

Inspired by `bd prime` - provides AI-optimized workflow guidance as single source of truth.
Replaces scattered prompt files with dynamic, context-aware output.
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


def render_prime_context() -> str:
    """Generate dynamic workflow context for xorq agents.

    This is the single source of truth for xorq workflow guidance.
    Replaces the need for scattered prompt files.
    """
    # Get project state (always needed)
    recent_builds = get_recent_builds(limit=5)
    catalog_entries = get_catalog_entries(limit=10)
    builds_status = _format_builds_status(recent_builds, catalog_entries)

    # Check for custom override
    custom = check_custom_prime()
    if custom:
        # Inject dynamic state into custom PRIME.md
        # Replace the placeholder with actual state
        if "**Recent activity**" in custom:
            custom = custom.replace(
                "**Recent activity** is displayed dynamically when you run `xorq agent prime`.",
                builds_status,
            )
        return custom

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
