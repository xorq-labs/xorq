import toolz

import xorq as xo
from xorq.common.utils.import_utils import import_python
from xorq.flight.client import FlightClient


m = import_python(xo.options.pins.get_path("hackernews_lib"))


(transform_port, transform_command) = (
    8765,
    "execute-unbound-expr-4de287288267a2b06a16c7d8bcc2011b",
)
(predict_port, predict_command) = (
    8766,
    "execute-unbound-expr-d9ab9b66dc4a3beafe4de74f66b4edb9",
)


z = (
    xo.memtable([{"maxitem": 43346282, "n": 1000}])
    .pipe(m.do_hackernews_fetcher_udxf)
    .filter(xo._.text.notnull())
    .mutate(
        **{
            "sentiment": xo.literal(None).cast(str),
            "sentiment_int": xo.literal(None).cast(int),
        }
    )
)


transform_client = FlightClient(port=transform_port)
transform_do_exchange = toolz.curry(transform_client.do_exchange, transform_command)
predict_client = FlightClient(port=predict_port)
predict_do_exchange = toolz.curry(predict_client.do_exchange, predict_command)


assert transform_command in transform_client.do_action_one("list-exchanges")
assert predict_command in predict_client.do_action_one("list-exchanges")

# in server script, do_exchange receives an expr, here they receive record_batch_reader
(fut0, rbr_out0) = transform_do_exchange(z.to_pyarrow_batches())
(fut1, rbr_out1) = predict_do_exchange(rbr_out0)
out = rbr_out1.read_pandas()
print(out)
