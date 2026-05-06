from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xorq.config import default_backend
from xorq.examples.core import (
    get_name_to_suffix,
    get_table_from_name,
    whitelist,
)


if TYPE_CHECKING:
    import xorq.vendor.ibis.expr.types as ir
    from xorq.vendor.ibis.backends import BaseBackend


class Example:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def fetch(
        self,
        backend: BaseBackend | None = None,
        table_name: str | None = None,
        deferred: bool = True,
        **kwargs: Any,
    ) -> ir.Table:
        if backend is None:
            backend = default_backend()

        return get_table_from_name(
            self.name,
            backend,
            table_name or self.name,
            deferred=deferred,
            **kwargs,
        )


def __dir__() -> tuple[str, ...]:
    return (
        "get_table_from_name",
        *whitelist,
    )


def __getattr__(name: str) -> Any:
    from xorq.vendor.ibis import examples as ibex  # noqa: PLC0415

    lookup = get_name_to_suffix()

    if name not in lookup:
        return getattr(ibex, name)

    return Example(name)


__all__ = (
    "get_table_from_name",
    *whitelist,
)
