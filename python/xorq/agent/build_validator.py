"""Build validation hooks for xorq expressions.

Validates builds for common issues before execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    message: str
    severity: str  # "error", "warning", "info"


class BuildValidator:
    """Validates xorq build expressions for common issues."""

    def __init__(self, script_path: Path):
        """Initialize validator with script path."""
        self.script_path = script_path
        self.content = script_path.read_text()

    def validate_all(self) -> list[ValidationResult]:
        """Run all validation checks."""
        results = [
            self.check_struct_pattern(),
            self.check_float64_usage(),
            self.check_schema_check(),
            self.check_imports(),
            self.check_manual_joins(),
        ]
        return [r for r in results if r is not None]

    def check_struct_pattern(self) -> ValidationResult | None:
        """Check if ML predictions use struct pattern."""
        # Look for .pipe(fitted.predict) or .pipe(...predict)
        if ".pipe(" not in self.content or "predict" not in self.content:
            return None  # Not an ML pipeline

        has_struct = "as_struct" in self.content
        has_unpack = ".unpack(" in self.content

        if not has_struct:
            return ValidationResult(
                passed=False,
                message="ML pipeline detected but missing as_struct pattern. Use as_struct before .pipe(fitted.predict)",
                severity="error",
            )

        if not has_unpack:
            return ValidationResult(
                passed=False,
                message="ML pipeline detected but missing .unpack(). Add .unpack() after .pipe(fitted.predict)",
                severity="error",
            )

        return ValidationResult(
            passed=True,
            message="✓ Struct pattern detected for ML predictions",
            severity="info",
        )

    def check_float64_usage(self) -> ValidationResult | None:
        """Check if categorical encodings use float64."""
        # Look for .case() patterns (categorical encoding)
        if ".case()" not in self.content:
            return None

        has_float64 = (
            '.cast("float64")' in self.content or ".cast('float64')" in self.content
        )
        has_int_cast = (
            '.cast("int8")' in self.content
            or ".cast('int8')" in self.content
            or '.cast("int16")' in self.content
            or ".cast('int16')" in self.content
        )

        if has_int_cast:
            return ValidationResult(
                passed=False,
                message="Found int8/int16 cast - sklearn requires float64. Change .cast('int8') to .cast('float64')",
                severity="error",
            )

        if not has_float64 and "sklearn" in self.content.lower():
            return ValidationResult(
                passed=False,
                message="sklearn detected but no float64 casts found. Cast all numeric features to float64",
                severity="warning",
            )

        return ValidationResult(
            passed=True,
            message="✓ float64 casts detected",
            severity="info",
        )

    def check_schema_check(self) -> ValidationResult | None:
        """Check if schema is checked before operations."""
        has_schema_check = (
            "schema()" in self.content or "print(table.schema())" in self.content
        )

        if not has_schema_check:
            return ValidationResult(
                passed=False,
                message="No schema check found. Add print(table.schema()) at the start",
                severity="warning",
            )

        return ValidationResult(
            passed=True,
            message="✓ Schema check found",
            severity="info",
        )

    def check_imports(self) -> ValidationResult | None:
        """Check for correct imports."""
        has_xorq_vendor_ibis = "from xorq.vendor import ibis" in self.content
        has_direct_ibis = re.search(r"^import ibis", self.content, re.MULTILINE)

        if has_direct_ibis and not has_xorq_vendor_ibis:
            return ValidationResult(
                passed=False,
                message="Found 'import ibis' - should be 'from xorq.vendor import ibis'",
                severity="error",
            )

        if has_xorq_vendor_ibis:
            return ValidationResult(
                passed=True,
                message="✓ Correct ibis import (xorq.vendor)",
                severity="info",
            )

        return None

    def check_manual_joins(self) -> ValidationResult | None:
        """Check for manual joins that might indicate incorrect ML pattern."""
        if ".pipe(" not in self.content or "predict" not in self.content:
            return None  # Not an ML pipeline

        # Look for suspicious patterns
        has_join_after_predict = bool(
            re.search(r"\.pipe\([^)]*predict[^)]*\).*\.join\(", self.content, re.DOTALL)
        )

        if has_join_after_predict:
            return ValidationResult(
                passed=False,
                message="Found .join() after .pipe(predict) - likely incorrect. Use struct pattern instead",
                severity="warning",
            )

        return None

    def print_summary(self):
        """Print validation summary."""
        results = self.validate_all()

        errors = [r for r in results if r.severity == "error"]
        warnings = [r for r in results if r.severity == "warning"]
        infos = [r for r in results if r.severity == "info"]

        print(f"\n{'=' * 70}")
        print(f"Build Validation: {self.script_path.name}")
        print(f"{'=' * 70}\n")

        if errors:
            print("❌ ERRORS:")
            for error in errors:
                print(f"  - {error.message}")
            print()

        if warnings:
            print("⚠️  WARNINGS:")
            for warning in warnings:
                print(f"  - {warning.message}")
            print()

        if infos:
            print("✓ CHECKS PASSED:")
            for info in infos:
                print(f"  {info.message}")
            print()

        if errors:
            print("Fix errors before building.")
            return False
        elif warnings:
            print("Consider addressing warnings.")
            return True
        else:
            print("All checks passed!")
            return True


def validate_build(script_path: str | Path) -> bool:
    """Validate a build script.

    Args:
        script_path: Path to the script to validate

    Returns:
        True if validation passed, False if errors found
    """
    path = Path(script_path)
    if not path.exists():
        print(f"Error: {path} not found")
        return False

    validator = BuildValidator(path)
    return validator.print_summary()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        validate_build(sys.argv[1])
    else:
        print("Usage: python build_validator.py <script.py>")
