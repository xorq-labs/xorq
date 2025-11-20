import functools
import importlib.metadata


@functools.cache
def _get_backend_names(*, exclude: tuple[str] = ()) -> frozenset[str]:
    """Return the set of known backend names.

    Parameters
    ----------
    exclude
        These backend names should be excluded from the result

    Notes
    -----
    This function returns a frozenset to prevent cache pollution.

    If a `set` is used, then any in-place modifications to the set
    are visible to every caller of this function.

    """

    entrypoints = importlib.metadata.entry_points(group="xorq.backends")
    return frozenset(ep.name for ep in entrypoints).difference(exclude)
