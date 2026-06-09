"""Path canonicalization, stat'ing, and extractors.

The catalog-extract tempdir prefix
(``…/xorq-catalog-<random>/``) is stripped so two ``load_expr_from_zip``
calls on the same zip produce equal tokens (ADR-0007).  Local paths are
stat'd to preserve ``ModificationTimeStrategy`` invalidation semantics;
HTTP and cloud paths use ``HEAD`` / object-metadata calls.
"""

from __future__ import annotations

import itertools
import pathlib
import re
import urllib.error
import urllib.request

import yaml12


# Catalog-extract tempdir prefix. ``xorq.catalog.expr_utils.load_expr_from_zip``
# extracts each load into a fresh ``tempfile.mkdtemp(prefix="xorq-catalog-")``,
# so the absolute paths embedded in DataFusion/DuckDB plan strings differ per
# load and would otherwise leak per-load randomness into the DT token. The
# greedy regex strips everything up to and including the last
# ``xorq-catalog-<…>/`` segment, leaving the build-hashed suffix (which IS
# content-addressed and stable across reloads). Owned by ADR-0007.
_CATALOG_EXTRACT_DIR_RE = re.compile(r".*/xorq-catalog-[^/]+/")

_REMOTE_SCHEMES = ("http://", "https://", "s3://", "gs://", "gcs://")


def _canonicalize_catalog_path(s: str) -> tuple[str, bool]:
    """Strip the ``xorq-catalog-<random>/`` tempdir prefix if present.

    Returns the canonicalized path AND whether canonicalization actually
    fired — callers should skip the ``stat`` step on canonicalized paths
    because the canonicalized form is now relative and the per-load tempfile
    has a fresh inode/mtime that would defeat catalog-reload stability.
    """
    canonical = _CATALOG_EXTRACT_DIR_RE.sub("", s)
    return canonical, canonical != s


def _normalize_path_stat(path: str, **kwargs) -> tuple:
    """Stable metadata for a path: HTTP HEAD, cloud metadata, or local stat."""

    if isinstance(path, str) and path.startswith(("http://", "https://")):
        req = urllib.request.Request(
            path, method="HEAD", headers={"User-Agent": "xorq-cache"}
        )
        try:
            resp = urllib.request.urlopen(req, timeout=10)
        except (urllib.error.URLError, OSError) as exc:
            raise OSError(
                f"failed to HEAD {path!r} for cache-key metadata: {exc}"
            ) from exc
        headers = resp.info()
        return (
            ("url", path),
            *(
                (k, headers.get(k))
                for k in ("Last-Modified", "Content-Length", "Content-Type")
            ),
        )
    if isinstance(path, str) and path.startswith(("s3://", "gs://", "gcs://")):
        from xorq.expr import api  # noqa: PLC0415

        meta = api.get_object_metadata(path, **kwargs)
        return tuple(
            (k, meta.get(k))
            for k in ("location", "last_modified", "size", "e_tag", "version")
        )
    p = pathlib.Path(path)
    if p.exists():
        from xorq.common.utils import defer_utils  # noqa: PLC0415

        return defer_utils.normalize_read_path_stat(p)
    raise FileNotFoundError(f"local path does not exist: {path!r}")


def _stat_or_canonical(path: str) -> tuple:
    """Token for an extracted file path.

    Paths that were canonicalized (relative after stripping the
    ``xorq-catalog-…/`` prefix) live in a tempdir whose inode/mtime differ
    per reload — stat'ing them would defeat catalog-reload stability. The
    canonical path already carries the build-hashed prefix, which is
    content-addressed, so the canonical string alone is a stable token.

    Non-canonical (absolute) paths point at user-managed files; stat them
    to keep ``ModificationTimeStrategy`` cache invalidation working (see
    ``test_parquet_cache_storage``).
    """

    if isinstance(path, str) and (
        path.startswith(_REMOTE_SCHEMES) or pathlib.Path(path).is_absolute()
    ):
        return _normalize_path_stat(path)
    return ("canonical-build-path", path)


def _extract_duckdb_file_paths(sql_ddl: str) -> tuple[str, ...]:
    """Extract file paths from a DuckDB DDL's read_parquet/read_csv literals.

    Paths are canonicalized via :func:`_canonicalize_catalog_path` so loads of
    a catalog zip into different tempdirs produce stable tokens.
    """

    import sqlglot as sg  # noqa: PLC0415

    tree = sg.parse_one(sql_ddl, dialect="duckdb")

    def canon(raw):
        if raw.startswith(_REMOTE_SCHEMES):
            return raw
        p = pathlib.Path(raw)
        absolute = str(p if p.is_absolute() else pathlib.Path("/") / raw)
        canonical, _ = _canonicalize_catalog_path(absolute)
        return canonical

    # ReadParquet stores the path in expressions[0]; restrict to it
    # to avoid capturing string-valued kwargs.
    parquet_paths = tuple(
        canon(lit.this)
        for func in tree.find_all(sg.exp.ReadParquet)
        if func.expressions
        for lit in func.expressions[0].find_all(sg.exp.Literal)
        if lit.is_string
    )
    # ReadCSV's func.expressions hold keyword args whose string literals are
    # not paths; restrict to func.this (the path argument).
    csv_paths = tuple(
        canon(lit.this)
        for func in tree.find_all(sg.exp.ReadCSV)
        if func.this is not None
        for lit in func.this.find_all(sg.exp.Literal)
        if lit.is_string
    )
    return parquet_paths + csv_paths


def _extract_datafusion_plan_paths(ep_str: str) -> tuple[str, ...]:
    """Extract file paths from a DataFusion execution plan's ``file_groups``.

    DataFusion's plan repr strips the leading ``/`` from absolute local paths;
    we restore it. Catalog-extract tempdir prefixes
    (``…/xorq-catalog-<random>/``) are then stripped via
    :func:`_canonicalize_catalog_path` so two ``load_expr_from_zip`` calls on
    the same zip produce equal tokens (ADR-0007).
    """

    file_groups_match = re.search(r"file_groups=(\{[^}]*\})", ep_str)
    if not file_groups_match:
        return ()
    parsed = yaml12.parse_yaml(file_groups_match.group(1))
    (groups,) = parsed.values()
    out = []
    for raw in itertools.chain.from_iterable(groups):
        if raw.startswith(_REMOTE_SCHEMES):
            out.append(raw)
            continue
        p = pathlib.Path(raw)
        absolute = str(p if p.is_absolute() else pathlib.Path("/") / raw)
        canonical, _ = _canonicalize_catalog_path(absolute)
        out.append(canonical)
    return tuple(out)


__all__ = [
    "_REMOTE_SCHEMES",
    "_canonicalize_catalog_path",
    "_extract_datafusion_plan_paths",
    "_extract_duckdb_file_paths",
    "_normalize_path_stat",
    "_stat_or_canonical",
]
