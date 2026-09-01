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


# Optional backends are excluded: their imports are extras-gated and lazy.
# `backends/pandas` and `common/utils/dasher` are in scope because pandas is a
# core dependency and dasher is always loaded; they are also the only module-level
# import sites for numpy and xxhash, so without them those declarations have no
# guard at all.
CORE_PACKAGES = (
    "python/xorq/backends/pandas",
    "python/xorq/caching",
    "python/xorq/catalog",
    "python/xorq/common/utils/dasher",
    "python/xorq/flight",
    "python/xorq/ibis_yaml",
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


def declared_dependency_names(project_root):
    """Normalized distribution names from [project].dependencies."""
    pyproject = tomlkit.parse(project_root.joinpath("pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]
    # strip extras, version specifiers and environment markers
    return {
        str(dependency)
        .split(";")[0]
        .strip()
        .split("[")[0]
        .split("<")[0]
        .split(">")[0]
        .split("=")[0]
        .split("!")[0]
        .split("~")[0]
        .strip()
        .lower()
        .replace("_", "-")
        for dependency in dependencies
    }


def iter_core_modules(project_root):
    for package in CORE_PACKAGES:
        for path in sorted(project_root.joinpath(package).rglob("*.py")):
            parts = path.relative_to(project_root).parts
            if "tests" in parts or path.name == "conftest.py":
                continue
            yield path


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


def test_project_root_is_the_repo_checkout(project_root):
    """The other tests are vacuous if we resolve the wrong pyproject.toml."""
    assert project_root.joinpath("python", "xorq", "__init__.py").exists()
    pyproject = tomlkit.parse(project_root.joinpath("pyproject.toml").read_text())
    assert pyproject["project"]["name"] == "xorq"


def test_core_packages_all_exist(project_root):
    """A typo in CORE_PACKAGES would silently scan nothing."""
    missing = [
        package
        for package in CORE_PACKAGES
        if not project_root.joinpath(package).is_dir()
    ]
    assert not missing, f"CORE_PACKAGES entries do not exist: {missing}"


def test_core_module_scan_is_non_empty(project_root):
    modules = list(iter_core_modules(project_root))
    assert len(modules) > 20, f"expected to scan many modules, found {len(modules)}"


def test_core_module_imports_are_declared(project_root):
    declared = declared_dependency_names(project_root)
    undeclared = {}
    for path in iter_core_modules(project_root):
        for import_root in module_level_import_roots(path):
            distribution = IMPORT_ROOT_TO_DISTRIBUTION.get(import_root, import_root)
            if distribution.lower().replace("_", "-") in declared:
                continue
            undeclared.setdefault(distribution, set()).add(
                str(path.relative_to(project_root))
            )
    assert not undeclared, "\n".join(
        f"{name!r} is imported at module level by {sorted(paths)} "
        f"but is not in [project].dependencies"
        for name, paths in sorted(undeclared.items())
    )


def test_import_root_mapping_has_no_stale_entries(project_root):
    imported = {
        import_root
        for path in iter_core_modules(project_root)
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


def test_declared_dependency_names_strips_specifiers(project_root):
    declared = declared_dependency_names(project_root)
    assert {"pyarrow", "pandas", "gitpython", "py-yaml12"} <= declared
    assert not any(character in name for name in declared for character in "<>=!;[ ")


def test_regression_packaging_is_declared(project_root):
    """Undeclared, this broke `xorq run` while leaving `xorq build` green."""
    assert "packaging" in declared_dependency_names(project_root)
    assert "packaging" in module_level_import_roots(
        project_root.joinpath("python", "xorq", "ibis_yaml", "packager.py")
    )


@pytest.mark.parametrize("distribution", ("packaging", "numpy", "pygments", "xxhash"))
def test_declaration_is_covered_by_the_scan(project_root, distribution):
    """Deleting any of these from pyproject.toml must fail the scan.

    Without this, a declaration can sit outside CORE_PACKAGES and a future
    lowest-direct failure invites deleting it with nothing going red.
    """
    imported = {
        IMPORT_ROOT_TO_DISTRIBUTION.get(import_root, import_root)
        for path in iter_core_modules(project_root)
        for import_root in module_level_import_roots(path)
    }
    assert distribution in imported
    assert distribution in declared_dependency_names(project_root)
