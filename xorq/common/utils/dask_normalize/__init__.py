import dask

import xorq.common.utils.dask_normalize.dask_normalize_expr  # noqa: F401
import xorq.common.utils.dask_normalize.dask_normalize_function  # noqa: F401
import xorq.common.utils.dask_normalize.dask_normalize_other  # noqa: F401


dask.config.set({"tokenize.ensure-deterministic": True})


@dask.base.normalize_token.register(object)
def raise_generic_object(o):
    raise ValueError(f"Object {o!r} cannot be deterministically hashed")
