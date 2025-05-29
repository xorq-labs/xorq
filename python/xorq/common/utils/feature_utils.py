import pandas as pd
import pyarrow as pa
import xorq as xo
from xorq.flight.client import FlightClient
from typing import List, Mapping, Any


class Entity:
    """
    Represents an entity for which features are computed (e.g., city, user_id).
    Acts like a primary key for joins and feature grouping.
    """
    def __init__(self, name: str, key_column: str, description: str = ""):
        self.name = name
        self.key_column = key_column
        self.description = description


class Feature:
    """
    Represents a feature with its offline expression and metadata.
    Online expressions are auto-generated from the offline schema.
    """
    def __init__(
        self,
        name: str,
        entity: Entity,
        timestamp_column: str,
        offline_expr: Any,
        dtype: str = "float",
        description: str = "",
    ):
        self.name = name
        self.entity = entity
        self.timestamp_column = timestamp_column
        self.dtype = dtype
        self.description = description
        self._offline_expr = offline_expr

    def offline_expr(self):
        return self._offline_expr

    def get_schema(self):
        """Get the schema from the offline expression."""
        return self._offline_expr.schema()


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
    Groups multiple features for the same entity.
    Builds combined expressions by joining individual feature expressions.
    Online expressions are auto-generated from offline schema.
    """
    def __init__(
        self,
        name: str,
        entity: Entity,
        features: List[Feature],
    ):
        self.name = name
        self.entity = entity
        self.features = features

        # Validate all features belong to the same entity
        for feature in features:
            if feature.entity.name != entity.name:
                raise ValueError(f"Feature {feature.name} belongs to entity {feature.entity.name}, "
                               f"but view expects entity {entity.name}")

    def offline_expr(self):
        """
        Build a combined offline expression by joining all feature expressions.
        """
        if not self.features:
            raise ValueError(f"No features in view {self.name}")

        # Start with the first feature's expression
        base_expr = self.features[0].offline_expr()

        # Join with remaining features on entity key
        for feature in self.features[1:]:
            feature_expr = feature.offline_expr()
            base_expr = base_expr.join(
                feature_expr,
                self.entity.key_column,
                how="full_outer"
            )

        return base_expr

    def get_combined_schema(self):
        """Get the combined schema from all features in this view."""
        offline_expr = self.offline_expr()
        return offline_expr.schema()


class FeatureStore:
    """
    Main entry: register views, materialize batch, serve & feed online.
    Auto-generates online expressions from offline schemas.
    """
    def __init__(self, online_client: FlightClient = None):
        self.registry = FeatureRegistry()
        self.views: Mapping[str, FeatureView] = {}
        self.online_client = online_client

    def register_view(self, view: FeatureView):
        if view.entity.name not in self.registry.entities:
            raise ValueError("Entity not registered before view")
        for f in view.features:
            self.registry.register_feature(f)
        self.views[view.name] = view

    def _build_online_expr(self, view_name: str):
        """
        Auto-generate online expression from the offline schema.
        This creates a simple SELECT statement on the online store table.
        """
        view = self.views[view_name]

        if self.online_client is None:
            raise ValueError("No online client configured")

        # Get schema from offline expression
        schema = view.get_combined_schema()

        # Extract column names from schema
        column_names = [field for field in schema]

        online_expr = xo.table(name=view_name, schema=schema).select(column_names)

        return online_expr

    def materialize_online(self, view_name: str):
        """
        Materialize features from offline expression to online storage.
        """
        view = self.views[view_name]

        # Execute the combined offline expression
        batch_df = view.offline_expr().execute()

        # Get the timestamp column from the first feature
        timestamp_column = view.features[0].timestamp_column

        # Get latest values per entity key
        latest = (
            batch_df
              .sort_values(timestamp_column)
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
        Uses auto-generated online expression.
        """
        view = self.views[view_name]
        key_col = view.entity.key_column

        keys_df = pd.DataFrame(rows)

        # Use the auto-generated online expression and filter by keys
        online_expr = self._build_online_expr(view_name)
        filtered_expr = online_expr.filter(
            xo._[key_col].isin(keys_df[key_col].tolist())
        )

        # Execute through the online client
        if self.online_client is None:
            raise ValueError("No online client configured")

        return self.online_client.execute(filtered_expr)
