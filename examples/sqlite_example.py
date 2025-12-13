import numpy as np
import pandas as pd
import pyarrow as pa

import xorq.api as xo
from xorq.caching import SourceCache


np.random.seed(42)

# Create a DataFrame with customer purchase data
df = pd.DataFrame(
    {
        "customer_age": np.random.randint(18, 75, 10000),
        "annual_income": np.random.normal(55000, 20000, 10000).round(2),
        "purchase_amount": np.random.gamma(2, 50, 10000).round(2),
        "product_category": np.random.choice(
            [
                "Electronics",
                "Clothing",
                "Home & Garden",
                "Sports",
                "Books",
                "Beauty",
                "Automotive",
                "Food",
            ],
            10000,
        ),
    }
)

reader = pa.Table.from_pandas(df).to_reader(max_chunksize=1000)

sqlite_con = xo.sqlite.connect()

t = sqlite_con.read_record_batches(reader, table_name="customer_purchase_data")


expr = (
    t.into_backend(con=xo.connect())
    .mutate(
        annual_income=xo._.annual_income.clip(25000, 150000),
        purchase_amount=xo._.annual_income.clip(10, 2000),
    )
    .select(
        "customer_age",
        "annual_income",
        "purchase_amount",
    )
    .cache(SourceCache.from_kwargs(source=sqlite_con))
)


if __name__ == "__pytest_main__":
    res = expr.execute()
    assert not res.empty
    assert sqlite_con.list_tables()
    pytest_examples_passed = True
