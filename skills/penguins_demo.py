"""Skill scaffold: penguins_demo
Minimal multi-engine example (penguins template) for demonstrations.

Referenced prompts:
# - context_blocks/xorq_core
# - context_blocks/expression_deliverables

After editing, build with `xorq build penguins_demo.py -e expr`
and catalog via `xorq catalog add builds/<hash> --alias penguins-agg`.
"""

import xorq.api as xo


con = xo.connect()
table = con.table("PENGUINS")
print(table.schema())  # align with context_blocks/must_check_schema

expr = (
    table
    # TODO: customize filters/aggregations
)
