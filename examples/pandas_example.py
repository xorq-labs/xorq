import pandas as pd

import xorq.api as xo


con = xo.connect()

df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]})
t = con.create_table("frame", df)
expr = t.head(3)


if __name__ == "__pytest_main__":
    res = expr.execute()
    print(res)
    pytest_examples_passed = True
