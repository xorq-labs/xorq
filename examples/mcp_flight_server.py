import pathlib
import pickle
import sys

import pandas as pd
import xgboost as xgb

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.common.utils.import_utils import import_python
from xorq.flight.exchanger import make_udxf


FlightMCPServer = import_python("examples/libs/mcp_lib.py").FlightMCPServer


TFIDF_MODEL_PATH = pathlib.Path(xo.options.pins.get_path("hn_tfidf_fitted_model"))
XGB_MODEL_PATH = pathlib.Path(xo.options.pins.get_path("hn_sentiment_reg"))


def load_models():
    transformer = pickle.loads(pathlib.Path(TFIDF_MODEL_PATH).read_bytes())
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)
    return transformer, xgb_model


transformer, xgb_model = load_models()


schema_in = xo.schema({"title": str})
schema_out = xo.schema({"sentiment_score": dt.float})


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    features = transformer.transform(df["title"])
    predictions = xgb_model.predict(features)
    result = pd.DataFrame({"sentiment_score": predictions.astype(float)})
    return result


sentiment_udxf = make_udxf(
    score_sentiment,
    schema_in.to_pyarrow(),
    schema_out.to_pyarrow(),
    name="sentiment_scorer",
)

mcp_server = FlightMCPServer("sentiment-analysis")


def sentiment_input_mapper(**kwargs):
    text = kwargs.get("kwargs", "")
    return xo.memtable({"title": [text]}, schema=schema_in)


def sentiment_output_mapper(result_df):
    if len(result_df) == 0:
        return {"sentiment_score": 0, "interpretation": "No result"}

    score = float(result_df["sentiment_score"].iloc[0])

    return {"sentiment_score": score}


mcp_server.create_mcp_tool(
    sentiment_udxf,
    input_mapper=sentiment_input_mapper,
    tool_name="analyze_sentiment",
    description="Analyze the sentiment of a text",
    output_mapper=sentiment_output_mapper,
)

if __name__ == "__main__":
    try:
        mcp_server.run(transport="stdio")
    except Exception:
        sys.exit(1)
