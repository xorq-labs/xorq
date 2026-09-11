from __future__ import annotations

from typing import Any

from attr import (
    field,
    fields,
)
from toolz import compose


convert_sorted_kwargs_tuple = compose(tuple, sorted, dict.items, dict)


SECRET_METADATA_KEY = "secret"


def secret_field(**kwargs: Any) -> Any:
    """An attrs field holding a credential: suppressed in ``repr``, marked secret.

    Secrecy stated once, at the field, rather than remembered at every print
    site. attrs defaults to ``repr=True``, so a plain `field()` holding a
    resolved credential prints the plaintext into any log line, traceback,
    debugger frame, or attrs validator error -- one forgotten f-string away
    from a leak. That suppression is what this delivers directly.

    The ``metadata`` marker is what ``secret_field_names`` reads, and it exists
    so the tree's two secrecy mechanisms can check each other: this one (repr
    suppression on an object holding a RESOLVED credential) and the profile
    machinery's (which enforces env-var *references* in a saved profile). They
    are disjoint paths over overlapping fields, so a credential added to one
    and forgotten in the other is exactly the drift a cross-check catches.
    """
    # the marker last, so caller metadata can't unmark the field: a
    # secret_field that secret_field_names doesn't report would silently
    # break the cross-check
    metadata = {**kwargs.pop("metadata", {}), SECRET_METADATA_KEY: True}
    return field(repr=False, metadata=metadata, **kwargs)


def secret_field_names(cls: type) -> tuple[str, ...]:
    """The ``secret_field``-marked field names of an attrs class."""
    return tuple(f.name for f in fields(cls) if f.metadata.get(SECRET_METADATA_KEY))


def validate_kwargs_tuple(instance: Any, attribute: Any, value: Any) -> None:
    assert isinstance(value, tuple) and all(
        isinstance(el, tuple) and len(el) == 2 for el in value
    )
