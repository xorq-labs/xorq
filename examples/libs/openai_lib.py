import functools
import operator
import os
from pathlib import Path
from urllib.parse import unquote_plus

import dask
import pandas as pd
import toolz
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import xorq as xo
from xorq.flight.utils import (
    schema_concat,
    schema_contains,
)


@toolz.curry
def simple_disk_cache(f, cache_dir):
    cache_dir = Path(cache_dir).absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)

    def wrapped(*args, **kwargs):
        name = dask.base.tokenize(*args, **kwargs)
        path = cache_dir.joinpath(name)
        if path.exists():
            value = path.read_text()
        else:
            value = f(*args, **kwargs)
            path.write_text(value)
        return value

    return wrapped


@functools.cache
def get_client():
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    return client


request_timeout = 3


@simple_disk_cache(cache_dir=Path("./openai-sentiment"))
def extract_sentiment(text):
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def completion_with_backoff(**kwargs):
        return get_client().chat.completions.create(**kwargs)

    if text == "":
        return "NEUTRAL"
    messages = [
        {
            "role": "system",
            "content": "You are an AI language model trained to analyze and detect the sentiment of hackernews forum comments.",
        },
        {
            "role": "user",
            "content": f"Analyze the following hackernews comment and determine if the sentiment is: positive, negative or neutral. "
            f"Return only a single word, either POSITIVE, NEGATIVE or NEUTRAL: {text}",
        },
    ]
    try:
        response = completion_with_backoff(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=30,
            temperature=0,
            timeout=request_timeout,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"


@toolz.curry
def get_hackernews_sentiment_batch(df: pd.DataFrame, input_col, append_col):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        values = tuple(
            executor.map(toolz.compose(extract_sentiment, unquote_plus), df[input_col])
        )
    return df.assign(**{append_col: values})


input_col = "text"
append_col = "sentiment"
schema_requirement = xo.schema({input_col: "str"})
schema_append = xo.schema({append_col: "str"})
maybe_schema_in = toolz.compose(schema_contains(schema_requirement), xo.schema)
maybe_schema_out = toolz.compose(
    operator.methodcaller("to_pyarrow"),
    schema_concat(to_concat=schema_append),
    xo.Schema.from_pyarrow,
)


do_hackernews_sentiment_udxf = xo.expr.relations.flight_udxf(
    process_df=get_hackernews_sentiment_batch(
        input_col=input_col, append_col=append_col
    ),
    maybe_schema_in=maybe_schema_in,
    maybe_schema_out=maybe_schema_out,
    name="HackerNewsSentimentAnalyzer",
)
