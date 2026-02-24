import importlib


__all__ = [  # noqa: PLE0604
    "Catalog",
    "CatalogAddition",
    "CatalogEntry",
    "CatalogRemoval",
    "build_expr_context",
    "build_expr_context_tgz",
    "load_expr_from_tgz",
]

_CATALOG_NAMES = frozenset(
    ("Catalog", "CatalogAddition", "CatalogEntry", "CatalogRemoval")
)
_EXPR_UTILS_NAMES = frozenset(
    ("build_expr_context", "build_expr_context_tgz", "load_expr_from_tgz")
)


def __getattr__(name):
    if name in _CATALOG_NAMES:
        mod = importlib.import_module("xorq.catalog.catalog")
        return getattr(mod, name)
    if name in _EXPR_UTILS_NAMES:
        mod = importlib.import_module("xorq.catalog.expr_utils")
        return getattr(mod, name)
    raise AttributeError(f"module 'xorq.catalog' has no attribute {name!r}")
