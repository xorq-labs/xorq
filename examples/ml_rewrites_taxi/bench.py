"""Benchmark native ML rewrites on NYC taxi trips.

Run: python examples/ml_rewrites_taxi/bench.py [--months 1 3 6] [--repeats 3]

Every (query, model, months, config) runs in its own process so peak RSS
is its own. Each repeat is end to end, fit included: memos are cleared
between repeats because the opt-out UDF path refits on every execution.
Writes results/bench.csv and results/bench.png next to this file.
"""

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

import xorq.api as xo
from xorq.expr.ml.native import inline, rewrite


sys.path.insert(0, str(Path(__file__).parent))
import demo  # noqa: E402


RESULTS = Path(__file__).parent / "results"
MODELS = {
    "ridge": Ridge,
    "tree": lambda: DecisionTreeRegressor(max_depth=8, random_state=0),
}
# (backend, options.ml.rewrite, options.ml.rules); None = plain sklearn
CONFIGS = {
    "sklearn": None,
    "xorq-udf": ("datafusion", False, frozenset()),
    "xorq-inline": ("datafusion", True, frozenset({"inline"})),
    "xorq-rewrite": ("datafusion", True, frozenset({"inline", "agg"})),
    "xorq-rewrite-duckdb": ("duckdb", True, frozenset({"inline", "agg"})),
}
CASES = (
    ("score", "ridge", tuple(CONFIGS)),
    ("agg", "ridge", tuple(CONFIGS)),
    ("agg", "tree", tuple(CONFIGS)),
    ("fit-segment", "ridge", ("sklearn", "xorq-udf", "xorq-rewrite-duckdb")),
)
CONNECT = {"datafusion": xo.connect, "duckdb": xo.duckdb.connect}


def sklearn_query(query, model, months):
    """What a user writes without xorq: pull the trips, then pandas."""
    df = demo.trips(xo.duckdb.connect(), months).execute()
    train = df[df.tpep_pickup_datetime < "2024-01-08"]
    features, target = list(demo.FEATURES), demo.TARGET
    if query == "fit-segment":
        return pd.DataFrame(
            {
                "payment_type": key,
                "model": demo.make_pipeline(model).fit(group[features], group[target]),
            }
            for key, group in train.groupby("payment_type")
        )[["payment_type"]]
    pipeline = demo.make_pipeline(model).fit(train[features], train[target])
    df["predicted"] = pipeline.predict(df[features])
    if query == "score":
        week = df.tpep_pickup_datetime.between("2024-01-15", "2024-01-22")
        return df[(df.PULocationID == demo.JFK) & week][
            ["tpep_pickup_datetime", "trip_distance", demo.TARGET, "predicted"]
        ]
    return (
        df.groupby(["PULocationID", "hour"])
        .agg(trips=("predicted", "size"), mean_predicted=("predicted", "mean"))
        .reset_index()
    )


def xorq_query(query, model, months, config):
    backend, rewrite_on, rules = CONFIGS[config]
    xo.options.ml.rewrite, xo.options.ml.rules = rewrite_on, rules
    taxi = demo.trips(CONNECT[backend](), months)
    result = demo.QUERIES[query](taxi, model).execute()
    if query == "fit-segment":
        return result[["payment_type"]]
    return result


def clear_memos():
    rewrite.materialize.cache_clear()
    rewrite.materialize_by.cache_clear()
    inline.translated_output.cache_clear()


def worker(query, model_name, months, config, repeats):
    python_rows = []
    for cls in (Ridge, DecisionTreeRegressor):

        def counting(self, X, *args, _predict=cls.predict, **kwargs):
            python_rows.append(len(X))
            return _predict(self, X, *args, **kwargs)

        cls.predict = counting
    run = (
        (lambda: sklearn_query(query, MODELS[model_name](), months))
        if CONFIGS[config] is None
        else (lambda: xorq_query(query, MODELS[model_name](), months, config))
    )
    seconds = []
    for _ in range(repeats):
        clear_memos()
        python_rows.clear()
        start = time.perf_counter()
        result = run()
        seconds.append(time.perf_counter() - start)
    out = RESULTS / f"{query}-{model_name}-{len(months)}-{config}.parquet"
    result.to_parquet(out)
    return {
        "seconds": statistics.median(seconds),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "python_rows": sum(python_rows),
        "result": str(out),
    }


def max_rel_error(query, actual, expected):
    """Largest gap to the sklearn baseline; inf on a row mismatch."""
    keys = {
        "score": ["tpep_pickup_datetime", "trip_distance"],
        "agg": ["PULocationID", "hour"],
        "fit-segment": ["payment_type"],
    }[query]
    actual, expected = (
        frame.sort_values(keys).reset_index(drop=True) for frame in (actual, expected)
    )
    if not actual[keys].equals(expected[keys]):
        return float("inf")
    if query == "fit-segment":
        return 0.0
    value = {"score": "predicted", "agg": "mean_predicted"}[query]
    # relative, but absolute below 1: a tree can predict a zero fare
    scale = expected[value].abs().clip(lower=1)
    return float(((actual[value] - expected[value]).abs() / scale).max())


def plot(frame, path):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    colors = dict(zip(CONFIGS, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    cases = frame[["query", "model"]].drop_duplicates().to_records(index=False)
    fig, axes = plt.subplots(1, len(cases), figsize=(4 * len(cases), 3.8))
    for ax, (query, model) in zip(axes, cases):
        case = frame[(frame["query"] == query) & (frame.model == model)]
        for config, rows in case.groupby("config", sort=False):
            ax.plot(rows.rows_m, rows.seconds, marker="o", color=colors[config])
        ax.set(title=f"{query} ({model})", xlabel="trips (millions)", yscale="log")
    axes[0].set_ylabel("seconds, fit included (log)")
    handles = [plt.Line2D([], [], color=color, marker="o") for color in colors.values()]
    fig.legend(handles, colors, loc="lower center", ncol=len(colors), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--worker", nargs=4, metavar=("QUERY", "MODEL", "MONTHS", "CONFIG")
    )
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.worker:
        query, model, months, config = args.worker
        print(
            json.dumps(
                worker(query, model, range(1, int(months) + 1), config, args.repeats)
            )
        )
        return
    rows = []
    for months in args.months:
        n_rows = demo.trips(xo.duckdb.connect(), range(1, months + 1)).count().execute()
        for query, model, configs in CASES:
            for config in configs:
                command = [sys.executable, __file__, "--repeats", str(args.repeats)]
                command += ["--worker", query, model, str(months), config]
                record = json.loads(subprocess.check_output(command).splitlines()[-1])
                rows.append(
                    {
                        "query": query,
                        "model": model,
                        "months": months,
                        "rows_m": n_rows / 1e6,
                        "config": config,
                        **record,
                    }
                )
                print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    baseline = frame[frame.config == "sklearn"].set_index(["query", "model", "months"])
    frame["max_rel_error"] = [
        max_rel_error(
            row.query,
            pd.read_parquet(row.result),
            pd.read_parquet(baseline.loc[(row.query, row.model, row.months), "result"]),
        )
        for row in frame.itertuples()
    ]
    frame.drop(columns="result").to_csv(RESULTS / "bench.csv", index=False)
    plot(frame, RESULTS / "bench.png")
    print(frame.drop(columns="result").to_string(index=False))


if __name__ == "__main__":
    main()
