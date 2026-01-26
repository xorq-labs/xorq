"""
Test the diamonds dataset filtering
"""

import xorq.api as xo
from xorq.vendor import ibis

# Connect and get the diamonds source
con = xo.connect()
diamonds_source = xo.catalog.get("diamonds-source")

# Define constraints
BUDGET = 10000
MIN_CLARITY = ['SI1', 'SI2', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
MIN_COLOR = ['D', 'E', 'F', 'G']

# Filter diamonds by quality constraints
filtered_diamonds = (
    diamonds_source
    .filter([
        ibis._.clarity.isin(MIN_CLARITY),
        ibis._.color.isin(MIN_COLOR),
        ibis._.price <= BUDGET,
        ibis._.price > 0
    ])
)

# Count how many diamonds match our criteria
diamond_count = (
    filtered_diamonds
    .aggregate([
        ibis._.count().name("total_diamonds"),
        ibis._.price.sum().name("total_value"),
        ibis._.carat.sum().name("total_carat"),
        ibis._.price.mean().name("avg_price"),
        ibis._.carat.mean().name("avg_carat")
    ])
)

# Check distribution by cut
cut_distribution = (
    filtered_diamonds
    .group_by("cut")
    .aggregate([
        ibis._.count().name("count"),
        ibis._.carat.sum().name("total_carat"),
        ibis._.price.mean().name("avg_price")
    ])
    .order_by(ibis.desc("count"))
)

expr = diamond_count
cut_expr = cut_distribution