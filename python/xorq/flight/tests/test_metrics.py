import time

import pandas as pd
import pyarrow as pa

from xorq.flight.metrics import instrument_rpc, setup_console_metrics


@instrument_rpc("dummy_do_get")
def dummy_do_get(self, context, *args, **kwargs):
    # Create a dictionary with column names as keys and lists as values
    data = {
        "Name": ["John", "Alice", "Bob"],
        "Age": [25, 30, 35],
        "City": ["New York", "London", "Paris"],
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    return pa.Table.from_pandas(df)


def test_setup_console_metrics_simple(capsys):
    setup_console_metrics()
    dummy_do_get(None, None)

    for i in range(20):
        time.sleep(1)
        if (
            "flight_server.requests_total{method=dummy_do_get} 1"
            in capsys.readouterr().out
        ):
            return

    assert False
