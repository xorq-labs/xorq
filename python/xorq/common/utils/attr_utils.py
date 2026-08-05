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


IDENTITY_METADATA_KEY = "identity"


def non_identity_field(**kwargs: Any) -> Any:
    """An attrs field deliberately kept OUT of derived identity.

    Where identity membership is derived from the attrs declaration (see
    ``xorq.backends.rest.config.identity_field_names``), a field is
    identity-bearing by default and opting out is an explicit annotation. This
    is that annotation, spelled as what it means: ``non_identity_field(...)``
    reads as "not part of identity" at the use site, where the underlying
    ``metadata={"identity": False}`` reads as its own inverse -- the key is
    named for the *concept*, so the value that excludes a field is ``False``,
    and an author copying the annotation onto a field they wanted hashed would
    silently exclude it. That is the one polarity mistake with cache-poisoning
    consequences: an identity-bearing field left unhashed makes two different
    configs share a hash, so cached data from one is served as current data for
    the other. Deriving identity exists to make the failure direction a
    spurious cache miss instead; the helper keeps the annotation on that side.

    Opting out therefore stays a conscious act carrying its own justification
    (the ``# why`` comment beside it), not an omission from a list nobody
    re-reads.
    """
    metadata = {IDENTITY_METADATA_KEY: False, **kwargs.pop("metadata", {})}
    return field(metadata=metadata, **kwargs)


def validate_kwargs_tuple(instance: Any, attribute: Any, value: Any) -> None:
    assert isinstance(value, tuple) and all(
        isinstance(el, tuple) and len(el) == 2 for el in value
    )
