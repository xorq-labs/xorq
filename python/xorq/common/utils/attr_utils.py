from __future__ import annotations

from typing import Any

from toolz import compose


convert_sorted_kwargs_tuple = compose(tuple, sorted, dict.items, dict)


def validate_kwargs_tuple(instance: Any, attribute: Any, value: Any) -> None:
    assert isinstance(value, tuple) and all(
        isinstance(el, tuple) and len(el) == 2 for el in value
    )
