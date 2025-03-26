import functools
import json
import operator
import os
from pathlib import Path
from urllib.parse import unquote_plus

import dask
import pandas as pd
import toolz
from openai import OpenAI

import xorq as xo
from xorq.flight.utils import (
    schema_concat,
    schema_contains,
)


_completions_kwargs = (
    ("model", "gpt-3.5-turbo"),
    ("max_tokens", 30),
    ("temperature", 0),
    ("timeout", 3),
)


@toolz.curry
def simple_disk_cache(f, cache_dir, serde):
    cache_dir = Path(cache_dir).absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)

    def wrapped(*args, **kwargs):
        name = dask.base.tokenize(*args, **kwargs)
        path = cache_dir.joinpath(name)
        if path.exists():
            value = serde.loads(path.read_text())
        else:
            value = f(*args, **kwargs)
            path.write_text(serde.dumps(value))
        return value

    return wrapped


@functools.cache
def get_client():
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    return client


@simple_disk_cache(cache_dir=Path("./openai-sentiment"), serde=json)
def request_chat_completions_dict(**kwargs):
    return get_client().chat.completions.create(**kwargs).model_dump()


def make_hn_sentiment_messages(text):
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
    return messages


def request_one_chat_completion(messages, **kwargs):
    if kwargs.get("n", 1) != 1:
        raise ValueError
    if "messages" in kwargs:
        raise ValueError
    _kwargs = dict(_completions_kwargs) | kwargs
    response_dict = request_chat_completions_dict(
        messages=messages,
        **_kwargs,
    )
    content = response_dict["choices"][0]["message"]["content"]
    return content


def extract_hn_sentiment(text, **kwargs):
    if text == "":
        return "NEUTRAL"
    messages = make_hn_sentiment_messages(text)
    try:
        content = request_one_chat_completion(messages, **kwargs)
    except Exception as e:
        content = f"ERROR: {e}"
    return content


@toolz.curry
def get_hackernews_sentiment_batch(df: pd.DataFrame, input_col, append_col):
    values = df[input_col].map(toolz.compose(extract_hn_sentiment, unquote_plus))
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
