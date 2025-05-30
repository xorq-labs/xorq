import pandas as pd
import pyarrow as pa
import xorq as xo
from xorq.flight.client import FlightClient
from typing import List, Mapping, Any, Union, Optional
from datetime import datetime, timedelta
import attrs


@attrs.define
class Entity:
    """
    Acts like a primary key for joins and feature grouping.
    """
    name: str
    key_column: str
    description: str = ""


@attrs.define
class Feature:
    """
    Represents a feature with its offline expression and metadata.
    Online expressions are auto-generated from the offline schema.
    """
    name: str
    entity: Entity
    timestamp_column: str
    offline_expr: Any = attrs.field(alias="_offline_expr")
    description: str = ""
    ttl: Optional[timedelta] = None

    def offline_expr(self):
        return self._offline_expr

    def get_schema(self):
        """Get the schema from the offline expression."""
        return self._offline_expr.schema()

    def is_expired(self, feature_timestamp: datetime, current_time: datetime = None) -> bool:
        """Check if a feature is expired based on its TTL."""
        if self.ttl is None:
            return False

        if current_time is None:
            current_time = datetime.now()

        return (current_time - feature_timestamp) > self.ttl


@attrs.define
class FeatureRegistry:
    """
    Registry of entities and features.
    """
    entities: Mapping[str, Entity] = attrs.field(factory=dict)
    features: Mapping[str, Feature] = attrs.field(factory=dict)

    def register_entity(self, entity: Entity):
        self.entities[entity.name] = entity

    def register_feature(self, feature: Feature):
        if feature.entity.name not in self.entities:
            raise ValueError(f"Entity {feature.entity.name} not registered")
        self.features[feature.name] = feature

    def list_features(self, entity_name: str) -> List[Feature]:
        return [f for f in self.features.values() if f.entity.name == entity_name]


@attrs.define
class FeatureView:
    """
    Groups multiple features for the same entity.
    Builds combined expressions by joining individual feature expressions.
    """
    name: str
    entity: Entity
    features: List[Feature]
    ttl: Optional[timedelta] = None  # Default TTL for all features in the view

    def __attrs_post_init__(self):
        for feature in self.features:
            if feature.entity.name != self.entity.name:
                raise ValueError(f"Feature {feature.name} belongs to entity {feature.entity.name}, "
                               f"but view expects entity {self.entity.name}")

            # Set view's TTL as default if feature doesn't have its own TTL
            if feature.ttl is None and self.ttl is not None:
                feature.ttl = self.ttl

    def offline_expr(self):
        if not self.features:
            raise ValueError(f"No features in view {self.name}")

        base_expr = self.features[0].offline_expr

        for feature in self.features[1:]:
            feature_expr = feature.offline_expr()
            base_expr = base_expr.join(
                feature_expr,
                self.entity.key_column,
                how="full_outer"
            )

        return base_expr

    def get_schema(self):
        offline_expr = self.offline_expr()
        return offline_expr.schema()

    def get_effective_ttl(self) -> Optional[timedelta]:
        """Get the minimum TTL among all features in the view."""
        ttls = [f.ttl for f in self.features if f.ttl is not None]
        return min(ttls) if ttls else None


@attrs.define
class FeatureStore:
    """
    Main entry: register views, materialize batch, serve & feed online.
    Auto-generates online expressions from offline schemas.
    """
    online_client: FlightClient = None
    registry: FeatureRegistry = attrs.field(factory=FeatureRegistry)
    views: Mapping[str, FeatureView] = attrs.field(factory=dict)

    def register_view(self, view: FeatureView):
        if view.entity.name not in self.registry.entities:
            raise ValueError("Entity not registered before view")
        for f in view.features:
            self.registry.register_feature(f)
        self.views[view.name] = view

    def _build_online_expr(self, view_name: str):
        view = self.views[view_name]

        if self.online_client is None:
            raise ValueError("No online client configured")

        # Get schema from offline expression
        schema = view.get_schema()

        # Extract column names from schema
        column_names = [field for field in schema]

        online_expr = xo.table(name=view_name, schema=schema).select(column_names)

        return online_expr

    def _parse_feature_references(self, features: List[str]) -> List[tuple]:
        """
        Parse feature references in the format "view_name:feature_name"
        Returns list of (view_name, feature_name) tuples
        """
        parsed_features = []
        for feature_ref in features:
            if ":" not in feature_ref:
                raise ValueError(f"Feature reference must be in format 'view_name:feature_name', got: {feature_ref}")

            view_name, feature_name = feature_ref.split(":", 1)
            if view_name not in self.views:
                raise ValueError(f"View '{view_name}' not found")

            view = self.views[view_name]
            feature_exists = any(f.name == feature_name for f in view.features)
            if not feature_exists:
                raise ValueError(f"Feature '{feature_name}' not found in view '{view_name}'")

            parsed_features.append((view_name, feature_name))

        return parsed_features

    def _apply_ttl_filter(self, df: pd.DataFrame, view: FeatureView, current_time: datetime = None) -> pd.DataFrame:
        """Apply TTL filtering to remove expired features."""
        if current_time is None:
            current_time = datetime.now()

        # Get the timestamp column from the first feature (assuming all features in view use same timestamp)
        timestamp_column = view.features[0].timestamp_column

        if timestamp_column not in df.columns:
            return df

        # Convert timestamp column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        # Apply per-feature TTL filtering
        mask = pd.Series([True] * len(df), index=df.index)

        for feature in view.features:
            if feature.ttl is not None:
                feature_mask = (current_time - df[timestamp_column]) <= feature.ttl

                feature_columns = [col for col in df.columns if col.startswith(feature.name)]
                if feature_columns:
                    mask &= feature_mask

        return df[mask]

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
    ) -> pd.DataFrame:
        # Could this be done with `xorq run`?
        # Validate entity_df has required columns
        if "event_timestamp" not in entity_df.columns:
            raise ValueError("entity_df must contain 'event_timestamp' column")

        # Parse feature references
        parsed_features = self._parse_feature_references(features)

        # Group features by view to minimize joins
        features_by_view = {}
        for view_name, feature_name in parsed_features:
            if view_name not in features_by_view:
                features_by_view[view_name] = []
            features_by_view[view_name].append(feature_name)

        # Start with the entity_df as base
        result_df = entity_df.copy()

        # For each view, get the historical features and join
        for view_name, feature_names in features_by_view.items():
            view = self.views[view_name]
            entity_key = view.entity.key_column

            # Validate that entity_df contains the required entity key
            if entity_key not in entity_df.columns:
                raise ValueError(f"entity_df must contain '{entity_key}' column for view '{view_name}'")

            view_expr = view.offline_expr()
            con = xo.duckdb.connect()

            entity_expr = xo.memtable(entity_df).into_backend(con=con)

            timestamp_col = view.features[0].timestamp_column

            if timestamp_col != "event_timestamp":
                view_expr_renamed = view_expr.rename(**{"event_timestamp": timestamp_col}).into_backend(con=con)
            else:
                view_expr_renamed = view_expr

            # Perform point-in-time join using asof_join
            # This finds the most recent record in view_expr where:
            # 1. feature_timestamp <= event_timestamp (temporal condition)
            # 2. entity keys match (predicates)
            historical_expr = entity_expr.asof_join(
                view_expr_renamed.mutate(event_timestamp=xo._.event_timestamp.cast('timestamp')),
                on="event_timestamp",  # Join on the timestamp column (now both tables have this name)
                predicates=entity_key,  # Entity key matching
            )

            feature_columns = [entity_key, "event_timestamp"] + feature_names

            # Handle case where not all requested features exist in the expression
            available_columns = historical_expr.schema()
            selected_columns = [col for col in feature_columns if col in available_columns]

            if not selected_columns:
                raise ValueError(f"No requested features found in view '{view_name}'")

            historical_df = historical_expr.select(selected_columns).execute()

            if len(features_by_view) == 1:
                result_df = historical_df
            else:
                join_keys = [entity_key, "event_timestamp"]
                result_df = result_df.merge(
                    historical_df,
                    on=join_keys,
                    how="left",
                    suffixes=("", f"_{view_name}")
                )

        return result_df

    def materialize_online(self, view_name: str, current_time: datetime = None):
        """
        Materialize features to online store with TTL filtering.

        Args:
            view_name: Name of the view to materialize
            current_time: Current time for TTL calculations (defaults to now)
        """
        view = self.views[view_name]

        # Execute the offline expression to get batch data
        batch_df = view.offline_expr().execute()

        # Get the timestamp column from the first feature
        timestamp_column = view.features[0].timestamp_column

        # Apply TTL filtering before getting latest values
        if current_time is None:
            current_time = datetime.now()

        filtered_df = self._apply_ttl_filter(batch_df, view, current_time)

        if filtered_df.empty:
            print(f"Warning: All features in view '{view_name}' are expired based on TTL")
            return

        # Get latest values per entity key from non-expired features
        latest = (
            filtered_df
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

        print(f"Materialized {len(latest)} non-expired feature records for view '{view_name}'")

    def get_online_features(
        self,
        view_name: str,
        rows: List[dict] = None,
        entity_df: pd.DataFrame = None,
        apply_ttl: bool = True,
        current_time: datetime = None,
    ) -> pd.DataFrame:
        """
        Get online features with optional TTL filtering.

        Args:
            view_name: Name of the view
            rows: List of entity key dictionaries
            entity_df: DataFrame with entity keys
            apply_ttl: Whether to filter out expired features
            current_time: Current time for TTL calculations
        """
        view = self.views[view_name]
        key_col = view.entity.key_column

        if entity_df is not None:
            keys_df = entity_df
        elif rows is not None:
            keys_df = pd.DataFrame(rows)
        else:
            raise ValueError("Either 'rows' or 'entity_df' must be provided")

        if key_col not in keys_df.columns:
            raise ValueError(f"Entity key '{key_col}' not found in input data")

        online_expr = self._build_online_expr(view_name)
        filtered_expr = online_expr.filter(
            xo._[key_col].isin(keys_df[key_col].tolist())
        )

        if self.online_client is None:
            raise ValueError("No online client configured")

        result_df = self.online_client.execute(filtered_expr).to_pandas()

        # Apply TTL filtering if requested
        if apply_ttl:
            result_df = self._apply_ttl_filter(result_df, view, current_time)

        result_df = keys_df.merge(result_df, on=key_col, how="left")

        return result_df

    def cleanup_expired_features(self, view_name: str, current_time: datetime = None):
        """
        Remove expired features from online storage based on TTL.
        This is useful for periodic cleanup jobs.
        """
        if current_time is None:
            current_time = datetime.now()

        view = self.views[view_name]

        # Get all online features
        online_expr = self._build_online_expr(view_name)
        all_features_df = self.online_client.execute(online_expr).to_pandas()

        # Apply TTL filtering to get non-expired features
        valid_features_df = self._apply_ttl_filter(all_features_df, view, current_time)

        expired_count = len(all_features_df) - len(valid_features_df)

        if expired_count > 0:
            # Re-upload only the valid features
            tbl = pa.Table.from_pandas(valid_features_df)
            self.online_client.upload_data(
                view_name,
                tbl,
                overwrite=True
            )
            print(f"Cleaned up {expired_count} expired features from view '{view_name}'")
        else:
            print(f"No expired features found in view '{view_name}'")
