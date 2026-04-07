"""Demonstrates registering a custom third-party ExprBuilder handler.

Any domain object that produces tagged expressions can integrate with the
ExprBuilder system. Register a TagHandler with extract_metadata and from_tagged
callbacks, tag your expressions, and the catalog + ls.builder will handle the
rest.

This example builds a simple "FeatureStore" that tags expressions with feature
metadata. The handler recovers the FeatureStore from the tag so you can build
new feature sets from a catalog entry.
"""

import tempfile
from pathlib import Path

import xorq.api as xo
from xorq.catalog.catalog import Catalog
from xorq.expr.builders import (
    TagHandler,
    register_tag_handler,
)


# ---------------------------------------------------------------------------
# 1. Define a simple domain object — a "FeatureStore"
# ---------------------------------------------------------------------------


class FeatureStore:
    """A toy feature store that selects named feature sets from a base table."""

    def __init__(self, table, features):
        self.table = table
        self.features = features  # dict of name -> list of columns

    def get_feature_set(self, name):
        """Select a named feature set and tag the result."""
        cols = self.features[name]
        return self.table.select(*cols).tag(
            "feature_store",
            feature_set=name,
            all_feature_sets=tuple(self.features.keys()),
            columns=tuple(cols),
        )


# ---------------------------------------------------------------------------
# 2. Register a TagHandler for "feature_store" tags
# ---------------------------------------------------------------------------


def _extract_metadata(tag_node):
    meta = tag_node.metadata
    return {
        "type": "feature_store",
        "description": f"feature set: {meta.get('feature_set')}",
        "feature_set": meta.get("feature_set"),
        "all_feature_sets": meta.get("all_feature_sets"),
        "columns": meta.get("columns"),
    }


def _from_tagged(tag_node):
    # Recover the base table from the tag's parent
    base_table = tag_node.parent.to_expr()
    meta = tag_node.metadata
    # Reconstruct the FeatureStore with the known feature sets
    # In a real system you'd recover the full feature definitions from
    # metadata or a registry — here we just know the selected columns.
    features = {meta["feature_set"]: list(meta["columns"])}
    return FeatureStore(base_table, features)


register_tag_handler(
    "feature_store",
    TagHandler(extract_metadata=_extract_metadata, from_tagged=_from_tagged),
)


# ---------------------------------------------------------------------------
# 3. Create data and a FeatureStore
# ---------------------------------------------------------------------------

con = xo.connect()
table = con.create_table(
    "sensor_data",
    {
        "timestamp": [1, 2, 3, 4, 5],
        "temperature": [20.1, 21.3, 19.8, 22.0, 20.5],
        "humidity": [45.0, 47.2, 44.1, 48.3, 46.0],
        "pressure": [1013.0, 1012.5, 1014.0, 1011.8, 1013.2],
        "wind_speed": [5.2, 3.1, 7.8, 2.4, 6.1],
    },
)

store = FeatureStore(
    table,
    features={
        "weather": ["temperature", "humidity", "pressure"],
        "wind": ["wind_speed"],
        "all": ["temperature", "humidity", "pressure", "wind_speed"],
    },
)

# Build a tagged expression for the "weather" feature set
weather_expr = store.get_feature_set("weather")
print("Weather features:")
print(weather_expr.execute())

# ---------------------------------------------------------------------------
# 4. Catalog the tagged expression
# ---------------------------------------------------------------------------

catalog_dir = Path(tempfile.mkdtemp()) / "feature-catalog"
catalog = Catalog.from_repo_path(catalog_dir, init=True)

catalog.add(weather_expr, aliases=("sensor-weather",), sync=False)

entry = catalog.get_catalog_entry("sensor-weather", maybe_alias=True)
print(f"\nEntry kind: {entry.kind}")
print(f"Builder metadata: {entry.metadata.builders}")

# ---------------------------------------------------------------------------
# 5. Recover the FeatureStore from the catalog entry via ls.builder
# ---------------------------------------------------------------------------

recovered_store = entry.expr.ls.builder
print(f"\nRecovered type: {type(recovered_store).__name__}")
print(f"Available feature sets: {list(recovered_store.features.keys())}")

# Use the recovered store to get the same feature set
recovered_expr = recovered_store.get_feature_set("weather")
print("\nRecovered weather features:")
print(recovered_expr.execute())


if __name__ == "__pytest_main__":
    pytest_examples_passed = True
