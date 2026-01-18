"""Error pattern matching for common xorq build failures.

Provides helpful guidance when builds fail with known error patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass(frozen=True)
class ErrorPattern:
    """Pattern for matching and handling common errors."""

    pattern: str  # Substring to match in error message
    name: str  # Human-readable error name
    cause: str  # What causes this error
    fix: str  # How to fix it
    examples: list[str]  # Code examples
    references: list[str]  # File references


# Known error patterns
KNOWN_PATTERNS = [
    ErrorPattern(
        pattern="Duplicate column",
        name="Duplicate Column in Result Set",
        cause=dedent("""\
            Common causes:
            1. Using mutate with column names that already exist
            2. Multiple aggregations creating same column name
            3. Unpacking structs that contain duplicate columns
            4. Join operations creating duplicate column names
            """).strip(),
        fix=dedent("""\
            Solution:
            1. Use unique column names in mutate operations
            2. Use .name() to rename aggregation results
            3. Drop existing columns before recreating them
            4. Use proper join predicates to avoid duplicates
            """).strip(),
        examples=[
            dedent("""\
                # ❌ WRONG:
                table.mutate(total=_.col1 + _.col2).mutate(total=_.col3)  # Duplicate!

                # ✅ RIGHT:
                table.mutate(total=_.col1 + _.col2).drop("total").mutate(total=_.col3)
                # Or just use one mutate:
                table.mutate(total=_.col3)  # Overwrites previous
                """).strip(),
        ],
        references=[
            "scripts/astronauts_prediction_simple.py",
            "scripts/batting_avg_ml_training.py",
        ],
    ),
    ErrorPattern(
        pattern="Failed to coerce arguments",
        name="Type Coercion Error (sklearn UDF)",
        cause=dedent("""\
            sklearn models in xorq require float64 for all numeric features.

            This error occurs when you:
            1. Use int8/int16/int32/int64 for categorical encodings
            2. Forget to cast to float64
            3. Use integer literals (0, 1, 2) instead of floats (0.0, 1.0, 2.0)
            """).strip(),
        fix=dedent("""\
            Solution:
            1. Cast ALL categorical encodings to float64
            2. Use float literals in case statements: 0.0 not 0
            3. Add explicit .cast("float64") after all case statements
            """).strip(),
        examples=[
            dedent("""\
                # ❌ WRONG:
                cut_score = (_.cut.case()
                    .when("Fair", 0)
                    .when("Good", 1)
                    .end()
                    .cast("int8"))  # ❌ int8 will fail!

                # ✅ RIGHT:
                cut_score = (_.cut.case()
                    .when("Fair", 0.0)
                    .when("Good", 1.0)
                    .end()
                    .cast("float64"))  # ✅ float64 required!
                """).strip(),
        ],
        references=[
            "scripts/astronauts_prediction_simple.py:35-43",
            "python/xorq/agent/prime.py:211-217",
        ],
    ),
    ErrorPattern(
        pattern="Cannot add",
        name="Cross-Relation Field Reference",
        cause=dedent("""\
            Trying to reference columns from different table expressions.

            This happens when:
            1. Trying to mutate with columns from another table
            2. Using chained mutates with _.column syntax incorrectly
            3. Not using table.column references after mutate
            """).strip(),
        fix=dedent("""\
            Solution:
            1. In chained mutates, use table.column not _.column for new columns
            2. For ML predictions: Use predict_udf.on_expr(table) in mutate
            3. Keep all columns in same table through transformations
            """).strip(),
        examples=[
            dedent("""\
                # ❌ WRONG:
                predictions = table.mutate(predicted=predict_udf.on_expr(table))
                result = predictions.mutate(error=_.predicted - _.target)  # May fail!

                # ✅ RIGHT:
                predictions = table.mutate(predicted=predict_udf.on_expr(table))
                result = predictions.mutate(
                    error=predictions.predicted - predictions.target
                )
                """).strip(),
        ],
        references=[
            "scripts/astronauts_prediction_simple.py:82-93",
        ],
    ),
    ErrorPattern(
        pattern="table 'datafusion.public",
        name="Missing Intermediate Table",
        cause=dedent("""\
            DataFusion cannot find an intermediate table reference.

            Common causes:
            1. Complex deferred expressions with broken lineage
            2. Manually created table references
            3. Cache invalidation issues
            """).strip(),
        fix=dedent("""\
            Solution:
            1. Rebuild all dependent expressions
            2. Clear cache: rm -rf .cache/xorq/
            3. Simplify expression to avoid complex joins
            4. Use struct pattern instead of manual joins
            """).strip(),
        examples=[],
        references=[],
    ),
    ErrorPattern(
        pattern="not found in table",
        name="Column Not Found",
        cause=dedent("""\
            Trying to access a column that doesn't exist in the table.

            Common causes:
            1. Typo in column name
            2. Column case mismatch (uppercase vs lowercase)
            3. Column was dropped earlier in pipeline
            4. Trying to use predicted column before it exists
            """).strip(),
        fix=dedent("""\
            Solution:
            1. Check schema: print(table.schema())
            2. Verify column case matches (Snowflake=UPPER, DuckDB=lower)
            3. Ensure column hasn't been dropped
            4. For predictions: use correct column name after unpack
            """).strip(),
        examples=[
            dedent("""\
                # ❌ WRONG:
                # After unpack, trying to use wrong name
                .mutate(price_pred=_.predicted_price)  # ❌ Doesn't exist yet!

                # ✅ RIGHT:
                .mutate(predicted_price=_.predicted)  # ✅ 'predicted' exists after pipe
                """).strip(),
        ],
        references=[],
    ),
]


def match_error(error_message: str) -> ErrorPattern | None:
    """Match error message against known patterns."""
    for pattern in KNOWN_PATTERNS:
        if pattern.pattern.lower() in error_message.lower():
            return pattern
    return None


def format_error_help(pattern: ErrorPattern) -> str:
    """Format helpful error guidance."""
    lines = [
        f"{'=' * 70}",
        f"🔴 {pattern.name}",
        f"{'=' * 70}",
        "",
        "CAUSE:",
        pattern.cause,
        "",
        "FIX:",
        pattern.fix,
    ]

    if pattern.examples:
        lines.extend(
            [
                "",
                "EXAMPLES:",
                *pattern.examples,
            ]
        )

    if pattern.references:
        lines.extend(
            [
                "",
                "REFERENCES:",
                *[f"  - {ref}" for ref in pattern.references],
            ]
        )

    lines.append(f"{'=' * 70}")

    return "\n".join(lines)


def handle_build_error(error_message: str) -> str | None:
    """Check if error matches known patterns and return help text."""
    pattern = match_error(error_message)
    if pattern:
        return format_error_help(pattern)
    return None


if __name__ == "__main__":
    # Test with sample errors
    test_errors = [
        "Duplicate column 'cut' in result set",
        "Failed to coerce arguments to satisfy a call to 'dumps_of_inner_fit_0'",
        "Cannot add <Field object> to projection, they belong to another relation",
    ]

    for error in test_errors:
        help_text = handle_build_error(error)
        if help_text:
            print(help_text)
            print("\n")
