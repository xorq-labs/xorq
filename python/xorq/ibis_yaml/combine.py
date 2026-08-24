"""Combine build artifacts (or already-loaded exprs) via join/union.

Shared core for the ``xorq join``/``xorq union`` and ``xorq catalog
join``/``xorq catalog union`` CLI commands: `join_exprs`/`union_exprs`
operate on already-resolved `Expr` objects (used directly by the catalog
tier, whose entries resolve to exprs via `CatalogEntry.load_expr`), while
`join_builds`/`union_builds` additionally load/build from plain build-path
strings (used by the top-level tier).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from xorq.api import Expr


def _split_columns(value: str) -> tuple[str, ...]:
    return tuple(c.strip() for c in value.split(",") if c.strip())


def _build_join_predicates(
    on: str | None,
    left_on: str | None,
    right_on: str | None,
    how: str,
) -> tuple[str | tuple[str, str], ...]:
    """Turn CLI-style --on/--left-on/--right-on into ibis's `predicates` shape."""
    if on is not None and (left_on is not None or right_on is not None):
        raise ValueError("--on is mutually exclusive with --left-on/--right-on")
    if (left_on is None) != (right_on is None):
        raise ValueError("--left-on and --right-on must be given together")

    if on is not None:
        return _split_columns(on)

    if left_on is not None:
        left_cols = _split_columns(left_on)
        right_cols = _split_columns(right_on)
        if len(left_cols) != len(right_cols):
            raise ValueError(
                f"--left-on has {len(left_cols)} column(s) but "
                f"--right-on has {len(right_cols)}"
            )
        return tuple(zip(left_cols, right_cols))

    if how != "cross":
        raise ValueError(
            "must specify --on or --left-on/--right-on, unless --how=cross"
        )
    return ()


def join_exprs(
    left: "Expr",
    right: "Expr",
    *,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    how: str = "inner",
    lname: str = "",
    rname: str = "{name}_right",
) -> "Expr":
    """Join two already-resolved exprs; a thin wrapper around `Table.join`."""
    predicates = _build_join_predicates(on, left_on, right_on, how)
    return left.join(right, predicates, how=how, lname=lname, rname=rname)


def union_exprs(*exprs: "Expr", distinct: bool = False) -> "Expr":
    """Union two-or-more already-resolved exprs; a thin wrapper around `Table.union`."""
    if len(exprs) < 2:
        raise ValueError("union requires at least 2 sources")
    first, *rest = exprs
    return first.union(*rest, distinct=distinct)


def join_builds(
    left_path: str | Path,
    right_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    builds_dir: str | Path = "builds",
    relocate_reads: bool = True,
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    how: str = "inner",
    lname: str = "",
    rname: str = "{name}_right",
) -> Path:
    """Load two build artifacts, join them, and write the result as a new build."""
    from xorq.ibis_yaml.compiler import build_expr, load_expr  # noqa: PLC0415

    left = load_expr(left_path, cache_dir=cache_dir)
    right = load_expr(right_path, cache_dir=cache_dir)
    expr = join_exprs(
        left, right, on=on, left_on=left_on, right_on=right_on, how=how,
        lname=lname, rname=rname,
    )
    return build_expr(
        expr, builds_dir=builds_dir, cache_dir=cache_dir, relocate_reads=relocate_reads
    )


def union_builds(
    *paths: str | Path,
    cache_dir: str | Path | None = None,
    builds_dir: str | Path = "builds",
    relocate_reads: bool = True,
    distinct: bool = False,
) -> Path:
    """Load two-or-more build artifacts, union them, and write the result as a new build."""
    from xorq.ibis_yaml.compiler import build_expr, load_expr  # noqa: PLC0415

    exprs: Sequence[Expr] = [load_expr(path, cache_dir=cache_dir) for path in paths]
    expr = union_exprs(*exprs, distinct=distinct)
    return build_expr(
        expr, builds_dir=builds_dir, cache_dir=cache_dir, relocate_reads=relocate_reads
    )


def _read_build_library_version(build_path: str | Path) -> str:
    from xorq.ibis_yaml.enums import DumpFiles  # noqa: PLC0415

    metadata_path = Path(build_path) / DumpFiles.build_metadata
    metadata = json.loads(metadata_path.read_text())
    return metadata["current_library_version"]


def assert_matching_library_versions(*build_paths: str | Path) -> None:
    """Raise `BuildVersionMismatchError` if the builds pin different xorq versions."""
    from xorq.common.exceptions import BuildVersionMismatchError  # noqa: PLC0415

    versions = {str(p): _read_build_library_version(p) for p in build_paths}
    if len(set(versions.values())) > 1:
        detail = ", ".join(f"{p}={v}" for p, v in versions.items())
        raise BuildVersionMismatchError(
            f"builds recorded different xorq library versions: {detail}. "
            "Pass --ignore-venv-mismatch to proceed anyway."
        )
