from datetime import datetime, timedelta
from typing import List, Mapping, Optional, Tuple

import pandas as pd
import pyarrow as pa
from attrs import (
    define,
    field,
    frozen,
)
from attrs.validators import (
    deep_iterable,
    instance_of,
    optional,
)

import xorq as xo
from xorq.flight import Backend as FlightBackend
from xorq.flight.client import FlightClient
from xorq.vendor.ibis.expr.types.core import (
    Expr,
)


@frozen
class Entity:
    """
    Acts like a primary key for joins and feature grouping.
    """

    name: str = field(validator=instance_of(str))
    key_column: str = field(validator=instance_of(str))
    description: str = field(validator=instance_of(str))


@frozen
class Feature:
    """
    Represents a feature with its offline expression and metadata.
    Online expressions are auto-generated from the offline schema.
    """

    name: str = field(validator=instance_of(str))
    entity: Entity = field(validator=instance_of(Entity))
    timestamp_column: str = field(validator=instance_of(str))
    offline_expr: Expr = field(validator=instance_of(Expr))
    description: str = field(validator=instance_of(str))
    ttl: Optional[timedelta] = field(
        validator=optional(instance_of(timedelta)), default=None
    )

    @property
    def schema(self):
        """Get the schema from the offline expression."""
        return self.offline_expr.schema()

    # def is_expired_expr(self, feature_timestamp_col, current_time: datetime = None):
    #     """Return an expression that checks if a feature is expired based on its TTL."""
    #     if self.ttl is None:
    #         return xo.literal(False)
    #     time_diff = xo.literal(current_time or datetime.now()) - feature_timestamp_col
    #     ttl_lit = xo.literal(self.ttl.total_seconds()).cast("interval")
    #     return time_diff > ttl_lit

    def clone(self, **kwargs):
        return type(self)(**self.__getstate__() | kwargs)

    def with_ttl(self, ttl):
        return self.clone(ttl=ttl)


@frozen
class FeatureView:
    """
    Groups multiple features for the same entity.
    Builds combined expressions by joining individual feature expressions.
    """

    name: str = field(validator=instance_of(str))
    entity: Entity = field(validator=instance_of(Entity))
    features: Tuple[Feature] = field(
        validator=deep_iterable(instance_of(Feature), instance_of(tuple))
    )
    ttl: Optional[timedelta] = field(
        validator=optional(instance_of(timedelta)), default=None
    )

    def __attrs_post_init__(self):
        self._validate_features()
        self._enforce_ttl()

    @property
    def timestamp_column(self):
        (timestamp_column, *rest) = set(
            feature.timestamp_column for feature in self.features
        )
        if rest:
            raise ValueError
        return timestamp_column

    @property
    def offline_expr(self):
        (expr, *others) = (feature.offline_expr for feature in self.features)
        for other in others:
            expr = expr.join(other, self.entity.key_column, how="full_outer")
        return expr

    @property
    def schema(self):
        return self.offline_expr.schema()

    @property
    def effective_ttl(self) -> Optional[timedelta]:
        """Get the minimum TTL among all features in the view."""
        ttls = [f.ttl for f in self.features if f.ttl is not None]
        return min(ttls, default=None)

    def _validate_features(self):
        # we must have features
        assert self.features
        # all features must have the same entity
        invalid_features = tuple(
            feature
            for feature in self.features
            if feature.entity.name != self.entity.name
        )
        if invalid_features:
            raise ValueError(
                f"Feature(s) {', '.join(feature.name for feature in invalid_features)} do not belong to {self.entity.name}"
            )
        # entities must not have schema conflicts
        # fixme: enforce this

    def _enforce_ttl(self):
        if self.ttl is not None and any(
            feature.ttl is None for feature in self.features
        ):
            features = tuple(
                feature.with_ttl(self.ttl) if feature.ttl is None else feature
                for feature in self.features
            )
            object.__setattr__(self, "features", features)


@define
class FeatureRegistry:
    """
    Registry of features
    """

    feature_mapping: Mapping[str, Feature] = field(factory=dict)

    @property
    def features(self):
        return tuple(self.feature_mapping.values())

    @property
    def entities(self):
        return tuple(set(feature.entity for feature in self.features))

    def register_feature(self, feature: Feature):
        self.feature_mapping[feature.name] = feature

    def get_entity_features(self, entity_name: str) -> List[Feature]:
        return [f for f in self.features if f.entity.name == entity_name]


@define
class FeatureStore:
    """
    Main entry: register views, materialize batch, serve & feed online.
    Auto-generates online expressions from offline schemas.
    """

    online_client: FlightClient = field(validator=instance_of(FlightClient))
    registry: FeatureRegistry = field(
        validator=instance_of(FeatureRegistry), factory=FeatureRegistry
    )
    views: Mapping[str, FeatureView] = field(factory=dict)

    def register_view(self, view: FeatureView):
        # what if we clobber a view and but retain its features in the registry
        for f in view.features:
            self.registry.register_feature(f)
        self.views[view.name] = view

    def _build_online_expr(self, view_name: str):
        if self.online_client is None:
            raise ValueError("No online client configured")
        # Hack: not sure how best to build bound expr without Backend
        # we probably need from_connection() implemented in Flight Backend
        fb = FlightBackend()
        fb.con = self.online_client

        # Extract column names from offline expression schema
        column_names = [field for field in self.views[view_name].schema]
        # why do we need to do a select if we are coordinating view name?
        online_expr = fb.tables[view_name].select(column_names)

        return online_expr

    def _parse_feature_references(self, references: List[str]) -> List[tuple]:
        """
        Parse feature references in the format "view_name:feature_name"
        Returns list of (view_name, feature_name) tuples
        """

        def validate_views_features(views_features):
            bad_references = tuple(
                reference for (reference, *rest) in views_features if not rest
            )
            if bad_references:
                raise ValueError
            bad_views = tuple(
                view for view, _ in views_features if view not in self.views
            )
            if bad_views:
                raise ValueError
            bad_views_features = tuple(
                (view, feature)
                for (view, feature) in views_features
                if feature not in self.views[view].features
            )
            if bad_views_features:
                raise ValueError

        views_features = tuple(reference.split(":", 1) for reference in references)
        validate_views_features(views_features)
        return views_features

    def _apply_ttl_filter_expr(
        self, expr, view: FeatureView, current_time: datetime = None
    ):
        """Apply TTL filtering to an expression to remove expired features."""
        (timestamp_column, effective_ttl) = (view.timestamp_column, view.effective_ttl)
        if timestamp_column not in expr.schema() or effective_ttl is None:
            return expr

        time_diff = xo.literal(current_time or datetime.now()) - xo._[timestamp_column]
        ttl_filter = time_diff > xo.literal(effective_ttl.total_seconds())
        return expr.filter(ttl_filter)

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

        # Apply TTL filtering using expressions
        filtered_expr = self._apply_ttl_filter_expr(
            view.offline_expr, view, current_time
        )

        # Get latest values per entity key using expressions
        # Sort by timestamp and get the last record for each entity
        # unclear why we only want one row
        latest_expr = (
            filtered_expr
            # should entity.key_column vary at all?
            .order_by([view.entity.key_column, view.timestamp_column])
            .mutate(
                row_number=xo.row_number().over(
                    group_by=view.entity.key_column,
                    order_by=xo.desc(view.timestamp_column),
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

        def make_keys_df():
            if (entity_df is not None) ^ (rows is not None):
                if entity_df is not None:
                    return entity_df
                else:
                    return pd.DataFrame(rows)
            else:
                raise ValueError(
                    "Exactly one of 'rows' or 'entity_df' must be provided"
                )

        keys_df = make_keys_df()
        view = self.views[view_name]
        key_col = view.entity.key_column
        if key_col not in keys_df.columns:
            raise ValueError(f"Entity key '{key_col}' not found in input data")

        # Build online expression and filter by entity keys
        filtered_expr = self._build_online_expr(view_name).filter(
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
