import pandas as pd

from xorq.common.utils.import_utils import (
    import_from_github,
)
from xorq.common.utils.perf_utils import (
    compare_runs,
)


tag = "v0.2.2"
lib = import_from_github(
    "xorq-labs", "xorq", "examples/complex_cached_expr.py", tag=tag
)


if __name__ == "__main__":
    (train_predicted, *_) = lib.make_exprs()
    (cleared, uncached_df, cached_df) = compare_runs(train_predicted)
    uncached_duration, cached_duration = (
        (df.end_datetime.max() - df.start_datetime.min()).total_seconds()
        for df in (uncached_df, cached_df)
    )
    delta_series = pd.Series(
        {
            "uncached_duration": uncached_duration,
            "cached_duration": cached_duration,
            "delta_duration": cached_duration - uncached_duration,
        }
    )
    (cache_miss_events, cache_hit_events) = (
        pd.concat(
            (
                pd.DataFrame(
                    dct for dct in trace.cache_event_dcts if dct["name"] == name
                )
                for trace in df.trace
            ),
            ignore_index=True,
        )
        for (df, name) in (
            (uncached_df, "cache.miss"),
            (cached_df, "cache.hit"),
        )
    )
    print(delta_series.round(2))
