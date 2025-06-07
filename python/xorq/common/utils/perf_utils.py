from datetime import datetime
from time import sleep

import pandas as pd

from xorq.common.utils.trace_utils import (
    Trace,
)


def clear_caches(expr):
    def clear_cache(node):
        from xorq.caching import ParquetStorage
        from xorq.expr.relations import CachedNode

        assert isinstance(node, CachedNode)
        expr = node.to_expr()
        if expr.ls.exists():
            storage = expr.ls.storage
            key = expr.ls.get_key()
            if isinstance(storage, ParquetStorage):
                key = node.storage.cache.storage.get_loc(key)
                key.unlink()
            else:
                storage.cache.drop(node.parent)
            return key
        else:
            return None

    return tuple(clear_cache(node) for node in expr.ls.cached_nodes)


def compare_runs(expr, sleep_duration=5):
    cleared = clear_caches(expr)
    first_cutoff = datetime.now()
    expr.execute()
    second_cutoff = datetime.now()
    expr.execute()
    sleep(sleep_duration)
    (traces, partials) = Trace.process_path()
    assert not partials
    df = pd.DataFrame(
        {
            "trace_id": trace.trace_id,
            "start_datetime": trace.start_datetime,
            "end_datetime": trace.end_datetime,
            "duration": trace.duration,
            "trace": trace,
        }
        for trace in traces
    )
    (first, second) = (
        df[lambda t: t.start_datetime.between(first_cutoff, second_cutoff)],
        df[lambda t: t.start_datetime.ge(second_cutoff)],
    )
    return cleared, first, second
