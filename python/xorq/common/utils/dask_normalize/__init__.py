import dask

import xorq.common.utils.dask_normalize.dask_normalize_expr  # noqa: F401
import xorq.common.utils.dask_normalize.dask_normalize_function  # noqa: F401
import xorq.common.utils.dask_normalize.dask_normalize_other  # noqa: F401


dask.config.set({"tokenize.ensure-deterministic": True})


@dask.base.normalize_token.register(object)
def raise_generic_object(o):
    method = getattr(o, "__dask_tokenize__", None)
    if method is not None and not isinstance(o, type):
        return method()
    raise ValueError(f"Object {o!r} cannot be deterministically hashed")


def get_normalize_token_subset(value=dask.tokenize.normalize_object):
    return {k: v for k, v in dask.base.normalize_token._lookup.items() if v == value}


# we have to clear out any classes that might have already been registered to dask's default normalize_object
bad_keys = tuple(get_normalize_token_subset(dask.tokenize.normalize_object))
for bad_key in bad_keys:
    dask.base.normalize_token._lookup.pop(bad_key)
