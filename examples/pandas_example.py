import pandas as pd

import letsql as ls

con = ls.connect()

df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]})
t = con.register(df, "frame")

res = t.head(3).execute()
print(res)
