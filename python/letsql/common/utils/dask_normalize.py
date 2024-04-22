import dask

import letsql.common.utils.dask_normalize_expr  # noqa: F401
import letsql.common.utils.dask_normalize_function  # noqa: F401
import letsql.common.utils.dask_normalize_other  # noqa: F401


dask.config.set({"tokenize.ensure-deterministic": True})
