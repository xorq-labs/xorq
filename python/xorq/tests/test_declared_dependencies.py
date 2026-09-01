"""Assert xorq declares every distribution it imports at module level.

An undeclared import reaches dev environments transitively (pytest, black) and
fails only on a user's bare install.  No allowlist for "it arrives via another
dependency": if xorq imports it at module level, xorq declares it.
"""

import ast
import sys

import pytest

# tomlkit rather than tomllib: xorq supports 3.10, where tomllib does not exist.
import tomlkit
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


SCANNED_PACKAGE = "python/xorq"

# vendor/ is skipped: its module-level third-party imports are upstream ibis's,
# and re-vendoring would churn any exclusion list kept here.  It is not
# unguarded -- xorq.api imports vendor/ibis, so test_bare_install catches an
# undeclared import there that is genuinely absent from a bare install.

# Modules whose module-level third-party imports belong to an optional backend.
# Excluded by path rather than by scanning an allowlist of packages, so a new
# module anywhere in xorq is in scope by default and has to be excluded
# deliberately.  test_extras_gated_modules_are_all_still_needed keeps this from
# accumulating entries that no longer apply.
EXTRAS_GATED_MODULES = (
    "python/xorq/backends/databricks/backend.py",
    "python/xorq/backends/postgres/__init__.py",
    "python/xorq/backends/pyiceberg/__init__.py",
    "python/xorq/backends/pyiceberg/compiler.py",
    "python/xorq/common/utils/bigquery_utils.py",
    "python/xorq/common/utils/databricks_utils.py",
    "python/xorq/common/utils/gcloud_utils.py",
    "python/xorq/common/utils/ibis_utils.py",
    "python/xorq/common/utils/postgres_utils.py",
    "python/xorq/common/utils/snowflake_utils.py",
    "python/xorq/common/utils/sqlite_utils.py",
    "python/xorq/expr/ml/sklearn_utils.py",
)

# Import root -> distribution name, for the cases where they differ.
IMPORT_ROOT_TO_DISTRIBUTION = {
    "attr": "attrs",
    "git": "gitpython",
    "yaml12": "py-yaml12",
    # namespace root: several declared opentelemetry-* distributions provide it
    "opentelemetry": "opentelemetry-sdk",
}

pytestmark = pytest.mark.core


def declared_dependency_names(root_dir):
    """Canonical distribution names from [project].dependencies."""
    pyproject = tomlkit.parse(root_dir.joinpath("pyproject.toml").read_text())
    return {
        canonicalize_name(Requirement(str(dependency)).name)
        for dependency in pyproject["project"]["dependencies"]
    }


def iter_core_modules(root_dir):
    """Every xorq module that must import with only the declared dependencies."""
    excluded = {root_dir.joinpath(module) for module in EXTRAS_GATED_MODULES}
    for path in sorted(root_dir.joinpath(SCANNED_PACKAGE).rglob("*.py")):
        parts = path.relative_to(root_dir).parts
        if "tests" in parts or "vendor" in parts or path.name == "conftest.py":
            continue
        if path in excluded:
            continue
        yield path


def undeclared_imports(path, declared):
    """Module-level import roots of *path* that are not declared dependencies."""
    return {
        import_root
        for import_root in module_level_import_roots(path)
        if canonicalize_name(IMPORT_ROOT_TO_DISTRIBUTION.get(import_root, import_root))
        not in declared
    }


def module_level_import_roots(path):
    """Third-party import roots imported unconditionally at module scope.

    ``tree.body`` not ``ast.walk``: guarded and deferred imports already handle
    their own absence.
    """
    tree = ast.parse(path.read_text())
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # node.level > 0 is a relative, intra-package import
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return {
        root for root in roots if root != "xorq" and root not in sys.stdlib_module_names
    }


def test_root_dir_is_the_repo_checkout(root_dir):
    """The other tests are vacuous if we resolve the wrong pyproject.toml."""
    assert root_dir.joinpath("python", "xorq", "__init__.py").exists()
    pyproject = tomlkit.parse(root_dir.joinpath("pyproject.toml").read_text())
    assert pyproject["project"]["name"] == "xorq"


def test_extras_gated_modules_all_exist(root_dir):
    """A stale path silently excludes nothing and hides a typo."""
    missing = [m for m in EXTRAS_GATED_MODULES if not root_dir.joinpath(m).is_file()]
    assert not missing, f"EXTRAS_GATED_MODULES entries do not exist: {missing}"


def test_extras_gated_modules_are_all_still_needed(root_dir):
    """Every exclusion must still have an undeclared module-level import.

    Without this the list becomes a place to park problems: an entry whose
    import was since declared, or removed, would keep a whole module out of
    scope for no reason.
    """
    declared = declared_dependency_names(root_dir)
    unnecessary = [
        module
        for module in EXTRAS_GATED_MODULES
        if not undeclared_imports(root_dir.joinpath(module), declared)
    ]
    assert not unnecessary, (
        f"EXTRAS_GATED_MODULES entries no longer have an undeclared "
        f"module-level import and should be removed: {unnecessary}"
    )


def test_core_module_scan_is_non_empty(root_dir):
    modules = list(iter_core_modules(root_dir))
    assert len(modules) > 20, f"expected to scan many modules, found {len(modules)}"


def test_core_module_imports_are_declared(root_dir):
    declared = declared_dependency_names(root_dir)
    undeclared = {}
    for path in iter_core_modules(root_dir):
        for import_root in undeclared_imports(path, declared):
            distribution = IMPORT_ROOT_TO_DISTRIBUTION.get(import_root, import_root)
            undeclared.setdefault(distribution, set()).add(
                str(path.relative_to(root_dir))
            )
    assert not undeclared, "\n".join(
        (
            *(
                f"{name!r} is imported at module level by {sorted(paths)} "
                f"but is not in [project].dependencies"
                for name, paths in sorted(undeclared.items())
            ),
            "Declare it, or -- if the import root differs from the distribution "
            "name -- add the mapping to IMPORT_ROOT_TO_DISTRIBUTION.",
        )
    )


def test_import_root_mapping_has_no_stale_entries(root_dir):
    imported = {
        import_root
        for path in iter_core_modules(root_dir)
        for import_root in module_level_import_roots(path)
    }
    stale = sorted(set(IMPORT_ROOT_TO_DISTRIBUTION) - imported)
    assert not stale, f"IMPORT_ROOT_TO_DISTRIBUTION entries no longer imported: {stale}"


@pytest.mark.parametrize(
    "source,expected",
    (
        ("import foo", {"foo"}),
        ("from foo.bar import baz", {"foo"}),
        ("import foo.bar, qux", {"foo", "qux"}),
        # relative imports are intra-package, never a declared dependency
        ("from . import sibling", set()),
        ("from .mod import thing", set()),
        # guarded: absence is handled, so it is not a packaging bug
        ("try:\n    import foo\nexcept ImportError:\n    foo = None", set()),
        # deferred on purpose, usually for an optional backend
        ("def f():\n    import foo", set()),
        ("if True:\n    import foo", set()),
        # stdlib and xorq itself are not third-party
        ("import os", set()),
        ("import xorq.api", set()),
    ),
)
def test_module_level_import_roots_classification(tmp_path, source, expected):
    path = tmp_path.joinpath("mod.py")
    path.write_text(source)
    assert module_level_import_roots(path) == expected


def test_declared_dependency_names_strips_specifiers(root_dir):
    declared = declared_dependency_names(root_dir)
    assert {"pyarrow", "pandas", "gitpython", "py-yaml12"} <= declared
    assert all(name == canonicalize_name(name) for name in declared)


def test_regression_packaging_is_declared(root_dir):
    """Undeclared, this broke `xorq run` while leaving `xorq build` green."""
    assert "packaging" in declared_dependency_names(root_dir)
    assert "packaging" in module_level_import_roots(
        root_dir.joinpath("python", "xorq", "ibis_yaml", "packager.py")
    )


@pytest.mark.parametrize("distribution", ("packaging", "numpy", "pygments", "xxhash"))
def test_declaration_is_covered_by_the_scan(root_dir, distribution):
    """Deleting any of these from pyproject.toml must fail the scan.

    Without this, a declaration can sit outside the scan and a future
    lowest-direct failure invites deleting it with nothing going red.
    """
    imported = {
        IMPORT_ROOT_TO_DISTRIBUTION.get(import_root, import_root)
        for path in iter_core_modules(root_dir)
        for import_root in module_level_import_roots(path)
    }
    assert distribution in imported
    assert distribution in declared_dependency_names(root_dir)
