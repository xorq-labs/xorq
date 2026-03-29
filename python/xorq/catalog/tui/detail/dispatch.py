from functools import cache


_STRATEGY_REGISTRY: dict[str, type] = {}


def register_detail(kind: str):
    """Decorator to register a DetailStrategy implementation for an ExprKind value."""

    def decorator(cls):
        _STRATEGY_REGISTRY[kind] = cls
        return cls

    return decorator


@cache
def get_detail_strategy(kind: str):
    """Look up the DetailStrategy for a given kind string.

    Falls back to StandardDetail for unknown kinds, ensuring forward
    compatibility as new ExprKind values are added.
    """
    from xorq.catalog.tui.detail.standard import StandardDetail  # noqa: PLC0415

    cls = _STRATEGY_REGISTRY.get(kind, StandardDetail)
    return cls()
