import functools
import json
import pathlib

import pandas as pd
import requests
import toolz

import xorq.api as xo
from xorq.common.utils.toolz_utils import curry


base_api_url = "https://hacker-news.firebaseio.com/v0"


@curry
def simple_disk_cache(f, cache_dir, serde):
    cache_dir.mkdir(parents=True, exist_ok=True)

    def wrapped(**kwargs):
        name = ",".join(f"{key}={value}" for key, value in kwargs.items())
        path = cache_dir.joinpath(name)
        if path.exists():
            value = serde.loads(path.read_text())
        else:
            value = f(**kwargs)
            path.write_text(serde.dumps(value))
        return value

    return wrapped


def get_json(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


@simple_disk_cache(cache_dir=pathlib.Path("./hackernews-items"), serde=json)
def get_hackernews_item(*, item_id):
    return get_json(f"{base_api_url}/item/{item_id}.json")


@functools.cache
def get_hackernews_maxitem():
    return get_json(f"{base_api_url}/maxitem.json")


def get_hackernews_stories(maxitem, n):
    gen = (
        toolz.excepts(requests.exceptions.SSLError, get_hackernews_item)(
            item_id=item_id
        )
        for item_id in range(maxitem - n, maxitem)
    )
    gen = filter(None, gen)
    df = pd.DataFrame(gen).reindex(columns=schema_out)
    return df


@curry
def get_hackernews_stories_batch(df, filter=slice(None)):
    series = df.apply(lambda row: get_hackernews_stories(**row), axis=1)
    return (
        pd.concat(series.values, ignore_index=True).loc[filter].reset_index(drop=True)
    )


schema_in = xo.schema({"maxitem": int, "n": int})
schema_out = xo.schema(
    {
        "by": "string",
        "id": "int64",
        "parent": "float64",
        "text": "string",
        "time": "int64",
        "type": "string",
        "kids": "array<int64>",
        # "deleted": "bool",
        "descendants": "float64",
        "score": "float64",
        "title": "string",
        "url": "string",
        # "dead": "bool",
    }
)


do_hackernews_fetcher_udxf = xo.expr.relations.flight_udxf(
    process_df=get_hackernews_stories_batch(
        filter=lambda t: t.type.eq("story") & t.title.notnull()
    ),
    maybe_schema_in=schema_in,
    maybe_schema_out=schema_out,
    name="HackerNewsFetcher",
)
t = xo.memtable(
    data=({"maxitem": 43182839, "n": 1000},),
    name="t",
)
expr = do_hackernews_fetcher_udxf(t)


if __name__ == "__pytest_main__":
    df = expr.execute()
    print(df)
    pytest_examples_passed = True
