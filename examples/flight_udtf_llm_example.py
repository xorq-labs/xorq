import functools
import operator
import os
from urllib.parse import unquote_plus

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


@functools.cache
def get_client():
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    return client


request_timeout = 3


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
    return df.assign(
        **{
            append_col: df[input_col].map(
                toolz.compose(extract_sentiment, unquote_plus)
            )
        }
    )


input_col = "text"
append_col = "sentiment"
schema_requirement = xo.schema({input_col: "!str"})
schema_append = xo.schema({append_col: "!str"})
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

hn = xo.examples.hn_posts_nano.fetch(table_name="hackernews")

expr = (
    hn.order_by(hn.time.desc())
    .filter(
        xo.or_(
            hn.text.cast(str).like("%ClickHouse%"),
            hn.title.cast(str).like("%ClickHouse%"),
        )
    )
    .select(hn.text)
    .limit(2)
    .pipe(do_hackernews_sentiment_udxf)
)

df = expr.execute()
print(df)
