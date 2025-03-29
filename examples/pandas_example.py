import pandas as pd

import xorq as xo


con = xo.connect()

df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]})
t = con.create_table("frame", df)

res = t.head(3).execute()
print(res)
pytest_examples_passed = True
