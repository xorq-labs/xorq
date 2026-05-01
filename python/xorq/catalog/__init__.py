__all__ = [
    "CatalogConfigurationError",
    "CatalogPushError",
]


def __getattr__(name):
    if name in __all__:
        from xorq.catalog import catalog as _catalog  # noqa: PLC0415

        return getattr(_catalog, name)
    raise AttributeError(f"module 'xorq.catalog' has no attribute {name!r}")
