from datetime import datetime, timedelta
from typing import Any, List, Mapping, Optional

import pandas as pd
import pyarrow as pa
from attrs import (
    define,
    field,
)
from attrs.validators import (
    instance_of,
)

import xorq as xo
from xorq.flight import Backend as FlightBackend
from xorq.flight.client import FlightClient
from xorq.vendor.ibis.expr.types.core import (
    Expr,
)


@define
class Entity:
    """
    Acts like a primary key for joins and feature grouping.
    """

    name: str = field(validator=instance_of(str))
    key_column: str = field(validator=instance_of(str))
    description: str = field(validator=instance_of(str))


@define
class Feature:
    """
    Represents a feature with its offline expression and metadata.
    Online expressions are auto-generated from the offline schema.
    """

    name: str = field(validator=instance_of(str))
    entity: Entity = field(validator=instance_of(Entity))
    timestamp_column: str = field(validator=instance_of(str))
    offline_expr: Any = field(validator=instance_of(Expr))
    description: str = field(validator=instance_of(str))
    ttl: Optional[timedelta] = field(default=None)

    def get_schema(self):
        """Get the schema from the offline expression."""
        return self.offline_expr.schema()

    def is_expired_expr(self, feature_timestamp_col, current_time: datetime = None):
        """Return an expression that checks if a feature is expired based on its TTL."""
        if self.ttl is None:
            return xo.literal(False)

        if current_time is None:
            current_time = datetime.now()

        current_time_lit = xo.literal(current_time)
        time_diff = current_time_lit - feature_timestamp_col
        ttl_lit = xo.literal(self.ttl.total_seconds()).cast("interval")

        return time_diff > ttl_lit


@define
class FeatureRegistry:
    """
    Registry of entities and features.
    """

    entities: Mapping[str, Entity] = field(factory=dict)
    features: Mapping[str, Feature] = field(factory=dict)

    def register_entity(self, entity: Entity):
        self.entities[entity.name] = entity

    def register_feature(self, feature: Feature):
        if feature.entity.name not in self.entities:
            raise ValueError(f"Entity {feature.entity.name} not registered")
        self.features[feature.name] = feature

    def list_features(self, entity_name: str) -> List[Feature]:
        return [f for f in self.features.values() if f.entity.name == entity_name]


@define
class FeatureView:
    """
    Groups multiple features for the same entity.
    Builds combined expressions by joining individual feature expressions.
    """

    name: str = field()
    entity: Entity = field()
    features: List[Feature] = field()
    ttl: Optional[timedelta] = field(default=None)

    def __attrs_post_init__(self):
        for feature in self.features:
            if feature.entity.name != self.entity.name:
                raise ValueError(
                    f"Feature {feature.name} belongs to entity {feature.entity.name}, "
                    f"but view expects entity {self.entity.name}"
                )

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
                feature_expr, self.entity.key_column, how="full_outer"
            )

        return base_expr

    def get_schema(self):
        offline_expr = self.offline_expr()
        return offline_expr.schema()

    def get_effective_ttl(self) -> Optional[timedelta]:
        """Get the minimum TTL among all features in the view."""
        ttls = [f.ttl for f in self.features if f.ttl is not None]
        return min(ttls) if ttls else None


@define
class FeatureStore:
    """
    Main entry: register views, materialize batch, serve & feed online.
    Auto-generates online expressions from offline schemas.
    """

    online_client: FlightClient = field(default=None)
    registry: FeatureRegistry = field(factory=FeatureRegistry)
    views: Mapping[str, FeatureView] = field(factory=dict)

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

        # Hack: not sure how best to build bound expr without Backend
        # we probably need from_connection() implemented in Flight Backend
        fb = FlightBackend()
        fb.con = self.online_client

        online_expr = fb.tables[view_name].select(column_names)

        return online_expr

    def _parse_feature_references(self, features: List[str]) -> List[tuple]:
        """
        Parse feature references in the format "view_name:feature_name"
        Returns list of (view_name, feature_name) tuples
        """
        parsed_features = []
        for feature_ref in features:
            if ":" not in feature_ref:
                raise ValueError(
                    f"Feature reference must be in format 'view_name:feature_name', got: {feature_ref}"
                )

            view_name, feature_name = feature_ref.split(":", 1)
            if view_name not in self.views:
                raise ValueError(f"View '{view_name}' not found")

            view = self.views[view_name]
            feature_exists = any(f.name == feature_name for f in view.features)
            if not feature_exists:
                raise ValueError(
                    f"Feature '{feature_name}' not found in view '{view_name}'"
                )

            parsed_features.append((view_name, feature_name))

        return parsed_features

    def _apply_ttl_filter_expr(
        self, expr, view: FeatureView, current_time: datetime = None
    ):
        """Apply TTL filtering to an expression to remove expired features."""
        if current_time is None:
            current_time = datetime.now()

        # Get the timestamp column from the first feature (assuming all features in view use same timestamp)
        timestamp_column = view.features[0].timestamp_column

        schema_fields = expr.schema()
        if timestamp_column not in schema_fields:
            return expr

        # Create TTL filter expressions for each feature
        ttl_filters = []

        for feature in view.features:
            if feature.ttl is not None:
                # Create expression for TTL check
                current_time_lit = xo.literal(current_time)
                time_diff = current_time_lit - xo._[timestamp_column]
                ttl_seconds = xo.literal(feature.ttl.total_seconds())

                # Feature is valid if time_diff <= ttl
                feature_valid = time_diff <= ttl_seconds
                ttl_filters.append(feature_valid)

        # If no TTL filters, return original expression
        if not ttl_filters:
            return expr

        # Combine all TTL filters with AND
        combined_filter = ttl_filters[0]
        for filter_expr in ttl_filters[1:]:
            combined_filter = combined_filter & filter_expr

        return expr.filter(combined_filter)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
    ):
        """
        Get historical features and return as an expression.

        Returns:
            xorq expression that can be executed to get the historical features
        """
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

        # Create expression from entity_df
        con = xo.duckdb.connect()
        result_expr = xo.memtable(entity_df).into_backend(con=con)

        # For each view, get the historical features and join
        for view_name, feature_names in features_by_view.items():
            view = self.views[view_name]
            entity_key = view.entity.key_column

            # Validate that entity_df contains the required entity key
            if entity_key not in entity_df.columns:
                raise ValueError(
                    f"entity_df must contain '{entity_key}' column for view '{view_name}'"
                )

            view_expr = view.offline_expr().into_backend(con=con)
            timestamp_col = view.features[0].timestamp_column

            # Rename timestamp column if needed
            if timestamp_col != "event_timestamp":
                view_expr = view_expr.rename(**{"event_timestamp": timestamp_col})

            # Perform point-in-time join using asof_join
            historical_expr = result_expr.asof_join(
                view_expr.mutate(
                    event_timestamp=xo._.event_timestamp.cast("timestamp")
                ),
                on="event_timestamp",
                predicates=entity_key,
            )

            # Select only the requested features plus keys
            feature_columns = [entity_key, "event_timestamp"] + feature_names
            available_columns = historical_expr.schema()
            selected_columns = [
                col for col in feature_columns if col in available_columns
            ]

            if not selected_columns:
                raise ValueError(f"No requested features found in view '{view_name}'")

            result_expr = historical_expr.select(selected_columns)

        return result_expr

    def materialize_online(self, view_name: str, current_time: datetime = None):
        """
        Materialize features to online store with TTL filtering using expressions.

        Args:
            view_name: Name of the view to materialize
            current_time: Current time for TTL calculations (defaults to now)
        """
        view = self.views[view_name]

        # Get the offline expression
        batch_expr = view.offline_expr()

        # Get the timestamp column from the first feature
        timestamp_column = view.features[0].timestamp_column

        # Apply TTL filtering using expressions
        if current_time is None:
            current_time = datetime.now()

        filtered_expr = self._apply_ttl_filter_expr(batch_expr, view, current_time)

        # Get latest values per entity key using expressions
        # Sort by timestamp and get the last record for each entity
        latest_expr = (
            filtered_expr.order_by([view.entity.key_column, timestamp_column])
            .mutate(
                row_number=xo.row_number().over(
                    group_by=view.entity.key_column, order_by=xo.desc(timestamp_column)
                )
            )
            .filter(xo._.row_number == 0)
            .drop("row_number")
        )

        # Execute to get the materialized data
        # TODO: Do not execute here
        latest_df = latest_expr.execute()

        if latest_df.empty:
            print(
                f"Warning: All features in view '{view_name}' are expired based on TTL"
            )
            return

        # Upload to online storage
        if self.online_client is None:
            raise ValueError("No online client configured")

        tbl = pa.Table.from_pandas(latest_df)
        self.online_client.upload_data(view_name, tbl, overwrite=True)

        print(
            f"Materialized {len(latest_df)} non-expired feature records for view '{view_name}'"
        )

    def get_online_features(
        self,
        view_name: str,
        rows: List[dict] = None,
        entity_df: pd.DataFrame = None,
        apply_ttl: bool = True,
        current_time: datetime = None,
    ):
        """
        Get online features with optional TTL filtering and return as an expression.

        Args:
            view_name: Name of the view
            rows: List of entity key dictionaries
            entity_df: DataFrame with entity keys
            apply_ttl: Whether to filter out expired features
            current_time: Current time for TTL calculations

        Returns:
            xorq expression that can be executed to get the online features
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

        # Build online expression and filter by entity keys
        online_expr = self._build_online_expr(view_name)
        filtered_expr = online_expr.filter(
            xo._[key_col].isin(keys_df[key_col].tolist())
        )

        # Apply TTL filtering if requested
        if apply_ttl:
            filtered_expr = self._apply_ttl_filter_expr(
                filtered_expr, view, current_time
            )

        # Join with keys to ensure all requested keys are present
        con = xo.duckdb.connect()
        keys_expr = xo.memtable(keys_df).into_backend(con=con)

        result_expr = keys_expr.join(
            filtered_expr.into_backend(con=con), key_col, how="left"
        )

        return result_expr

    def cleanup_expired_features(self, view_name: str, current_time: datetime = None):
        """
        Remove expired features from online storage based on TTL using expressions.
        This is useful for periodic cleanup jobs.

        Returns:
            xorq expression representing the cleaned up features
        """
        if current_time is None:
            current_time = datetime.now()

        view = self.views[view_name]

        # Get all online features as expression
        online_expr = self._build_online_expr(view_name)

        # Apply TTL filtering to get non-expired features
        valid_features_expr = self._apply_ttl_filter_expr(
            online_expr, view, current_time
        )

        # Execute both to get counts for logging
        all_features_df = online_expr.execute()
        valid_features_df = valid_features_expr.execute()

        expired_count = len(all_features_df) - len(valid_features_df)

        if expired_count > 0:
            # Re-upload only the valid features
            tbl = pa.Table.from_pandas(valid_features_df)
            self.online_client.upload_data(view_name, tbl, overwrite=True)
            print(
                f"Cleaned up {expired_count} expired features from view '{view_name}'"
            )
        else:
            print(f"No expired features found in view '{view_name}'")

        return valid_features_expr

    def get_historical_features_expr(
        self,
        entity_expr,
        features: List[str],
    ):
        """
        Get historical features from an entity expression and return as an expression.

        Args:
            entity_expr: xorq expression containing entity keys and event_timestamp
            features: List of feature references in format "view_name:feature_name"

        Returns:
            xorq expression that can be executed to get the historical features
        """
        # Parse feature references
        parsed_features = self._parse_feature_references(features)

        # Group features by view to minimize joins
        features_by_view = {}
        for view_name, feature_name in parsed_features:
            if view_name not in features_by_view:
                features_by_view[view_name] = []
            features_by_view[view_name].append(feature_name)

        # Start with the entity expression
        result_expr = entity_expr

        # For each view, get the historical features and join
        for view_name, feature_names in features_by_view.items():
            view = self.views[view_name]
            entity_key = view.entity.key_column

            view_expr = view.offline_expr()
            timestamp_col = view.features[0].timestamp_column

            # Rename timestamp column if needed
            if timestamp_col != "event_timestamp":
                view_expr = view_expr.rename(**{timestamp_col: "event_timestamp"})

            # Perform point-in-time join using asof_join
            historical_expr = result_expr.asof_join(
                view_expr.mutate(
                    event_timestamp=xo._.event_timestamp.cast("timestamp")
                ),
                on="event_timestamp",
                predicates=entity_key,
            )

            # Select only the requested features plus keys
            feature_columns = [entity_key, "event_timestamp"] + feature_names
            available_columns = historical_expr.schema()
            selected_columns = [
                col for col in feature_columns if col in available_columns
            ]

            if not selected_columns:
                raise ValueError(f"No requested features found in view '{view_name}'")

            result_expr = historical_expr.select(selected_columns)

        return result_expr
