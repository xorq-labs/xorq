import functools
import os
from urllib.parse import unquote_plus

import pandas as pd
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import xorq as xo


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


def get_hackernews_sentiment_batch(df: pd.DataFrame):
    df["text"].apply(lambda x: unquote_plus(x.decode("utf-8")))
    return df.assign(sentiment=df.text.apply(extract_sentiment))


schema_in = xo.schema(
    {
        "text": "!binary",
    }
)
schema_out = xo.schema({"text": "!str", "sentiment": "!str"})

do_hackernews_sentiment_udxf = xo.expr.relations.flight_udxf(
    process_df=get_hackernews_sentiment_batch,
    maybe_schema_in=schema_in.to_pyarrow(),
    maybe_schema_out=schema_out.to_pyarrow(),
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
)

expr = do_hackernews_sentiment_udxf(expr)
df = expr.execute()
print(df)
