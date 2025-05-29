import pandas as pd
import pyarrow as pa
import xorq as xo
from xorq.flight.client import FlightClient
from typing import List, Mapping, Any


class Entity:
    """
    Represents an entity for which features are computed (e.g., city, user_id).
    """
    def __init__(self, name: str, key_column: str, timestamp_column: str, description: str = ""):
        self.name = name
        self.key_column = key_column
        self.timestamp_column = timestamp_column
        self.description = description


class Feature:
    """
    Wraps an expression with metadata.
    expr: an ibis/xorq ColumnExpr returning one column.
    """
    def __init__(
        self,
        name: str,
        entity: Entity,
        expr: Any,
        dtype: str,
        description: str = "",
    ):
        self.name = name
        self.entity = entity
        self.expr = expr
        self.dtype = dtype
        self.description = description


class FeatureRegistry:
    """
    Registry of entities and features.
    """
    def __init__(self):
        self.entities: Mapping[str, Entity] = {}
        self.features: Mapping[str, Feature] = {}

    def register_entity(self, entity: Entity):
        self.entities[entity.name] = entity

    def register_feature(self, feature: Feature):
        if feature.entity.name not in self.entities:
            raise ValueError(f"Entity {feature.entity.name} not registered")
        self.features[feature.name] = feature

    def list_features(self, entity_name: str) -> List[Feature]:
        return [f for f in self.features.values() if f.entity.name == entity_name]


class FeatureView:
    """
    Bundles entity and features with offline/online expressions.
    Works directly with xorq expressions - no DataSource needed.
    """
    def __init__(
        self,
        name: str,
        entity: Entity,
        features: List[Feature],
        offline_expr: Any = None,  # Complete offline expression/table
        online_expr: Any = None,   # Complete online expression/table
    ):
        self.name = name
        self.entity = entity
        self.features = features
        self._offline_expr = offline_expr
        self._online_expr = online_expr

    def offline_expr(self):
        """Return the complete offline expression - already contains all feature computations"""
        if self._offline_expr is None:
            raise ValueError(f"No offline expression provided for view {self.name}")

        return self._offline_expr

    def online_expr(self):
        """Build the complete online expression with feature computations"""
        if self._online_expr is None:
            raise ValueError(f"No online expression provided for view {self.name}")

        base = self._online_expr
        mapping = {
            f.name: f.expr.unbind()  # "unbind" → "bind to this table"
            for f in self.features
        }
        cols = [self.entity.key_column, self.entity.timestamp_column] + list(mapping.keys())
        return base.select(cols)


class FeatureStore:
    """
    Main entry: register views, materialize batch, serve & feed online.
    Works directly with expressions - no DataSource management.
    """
    def __init__(self, online_client: FlightClient = None):
        self.registry = FeatureRegistry()
        self.views: Mapping[str, FeatureView] = {}
        self.online_client = online_client  # Direct FlightClient for online operations

    def register_view(self, view: FeatureView):
        if view.entity.name not in self.registry.entities:
            raise ValueError("Entity not registered before view")
        for f in view.features:
            self.registry.register_feature(f)
        self.views[view.name] = view

    def materialize_online(self, view_name: str):
        """
        Materialize features from offline expression to online storage.

        Args:
            view_name: Name of the feature view
            online_table_name: Name of the table in online storage
        """
        view = self.views[view_name]

        # Execute the complete offline expression
        batch_df = view.offline_expr().execute()

        # Get latest values per entity key
        latest = (
            batch_df
              .sort_values(view.entity.timestamp_column)
              .groupby(view.entity.key_column)
              .tail(1)
        )

        # Upload to online storage
        if self.online_client is None:
            raise ValueError("No online client configured")

        tbl = pa.Table.from_pandas(latest)
        self.online_client.upload_data(
            view_name,
            tbl,
            overwrite=True
        )

    def get_online_features(
        self,
        view_name: str,
        rows: List[dict],
    ) -> pd.DataFrame:
        """
        Get online features for the given entity keys.
        Uses the view's online expression.
        """
        view = self.views[view_name]
        key_col = view.entity.key_column

        keys_df = pd.DataFrame(rows)

        # Use the complete online expression and filter by keys
        expr = (
            view
              .online_expr()
              .filter(xo._[key_col].isin(keys_df[key_col].tolist()))
        )

        # Execute through the online client
        if self.online_client is None:
            raise ValueError("No online client configured")

        return self.online_client.execute(expr)
