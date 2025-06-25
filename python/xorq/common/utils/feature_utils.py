import functools
import operator
from datetime import datetime, timedelta
from typing import List, Mapping, Optional, Tuple

import pandas as pd
import pyarrow as pa
import toolz
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
from xorq.vendor.ibis.expr.datatypes import (
    DataType,
)
from xorq.vendor.ibis.expr.types.core import (
    Expr,
)


EVENT_TIMESTAMP = "event_timestamp"


@frozen
class Entity:
    """
    Acts like a primary key for joins and feature grouping.
    """

    name: str = field(validator=instance_of(str))
    key_column: str = field(validator=instance_of(str))
    description: str = field(validator=instance_of(str), default="")

    def __attrs_post_init__(self):
        assert all(getattr(self, name) for name in ("name", "key_column"))


@frozen
class Feature:
    """
    Represents a feature
    """

    name: str = field(validator=instance_of(str))
    dtype: DataType = field(validator=instance_of(DataType))
    description: str = field(validator=instance_of(str), default="")
    tags: tuple = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple)), default=()
    )

    def __attrs_post_init__(self):
        assert all(getattr(self, name) for name in ("name",))


@frozen
class FeatureView:
    """
    Groups multiple features for the same entity.
    Builds combined expressions by joining individual feature expressions.
    """

    name: str = field(validator=instance_of(str))
    features: Tuple[Feature] = field(
        validator=deep_iterable(instance_of(Feature), instance_of(tuple))
    )
    offline_expr: Expr = field(validator=instance_of(Expr))
    entities: Tuple[Entity] = field(
        validator=deep_iterable(instance_of(Entity), instance_of(tuple)),
        default=(),
    )
    timestamp_column: str = field(
        validator=optional(instance_of(str)),
        default=None,
    )
    ttl: Optional[timedelta] = field(
        validator=optional(instance_of(timedelta)),
        default=None,
    )
    tags: tuple = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple)),
        default=(),
    )

    def __attrs_post_init__(self):
        assert all(getattr(self, name) for name in ("name",))
        assert self.timestamp_column is None or self.timestamp_column
        assert not self.tags or all(map(all, zip(*self.tags)))
        self._validate_features()

    @property
    def key_columns(self):
        return tuple(entity.key_column for entity in self.entities)

    @property
    def entity_names(self):
        return tuple(entity.name for entity in self.entities)

    @property
    def feature_names(self):
        return tuple(feature.name for feature in self.features)

    @property
    def columns(self):
        return self.key_columns + self.feature_names

    @property
    def schema(self):
        return self.offline_expr.schema()

    def _validate_features(self):
        # we must have features
        if not self.features:
            raise ValueError
        # everything must have a unique name,key
        if len(self.key_columns) != len(set(self.key_columns)):
            raise ValueError
        if len(self.entity_names) != len(set(self.entity_names)):
            raise ValueError
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError
        if len(self.columns) != len(set(self.columns)):
            raise ValueError
        # we must have all columns in our schema
        if set(self.columns).difference(self.schema):
            raise ValueError


@frozen
class FeatureService:
    def __attrs_post_init__(self):
        raise NotImplementedError


@frozen
class FeatureRegistry:
    """
    Registry of FeatureViews, FeatureServices
    """

    views: Tuple[FeatureView] = field(
        validator=deep_iterable(instance_of(FeatureView), instance_of(tuple)),
    )
    services: Tuple[FeatureService] = field(
        validator=deep_iterable(instance_of(FeatureService), instance_of(tuple)),
    )

    @property
    @functools.cache
    def entities(self):
        return tuple(set(entity for view in self.views for entity in view.entities))

    @property
    def entity_names(self):
        return tuple(entity.name for entity in self.entities)

    @property
    @functools.cache
    def features(self):
        return tuple(set(feature for view in self.views for feature in view.features))

    def feature_names(self):
        return tuple(feature.name for feature in self.features)

    @property
    @functools.cache
    def view_names(self):
        return tuple(view.name for view in self.views)

    @property
    @functools.cache
    def service_names(self):
        return tuple(service.name for service in self.services)

    def get_feature_view(self, view_name):
        return next(view for view in self.views if view.name == view_name)

    def get_feature_service(self, service_name):
        return next(
            service for service in self.services if service.name == service_name
        )

    def get_entity_features(self, entity_name: str) -> List[Feature]:
        return [f for f in self.features if f.entity.name == entity_name]


@define
class FeatureStore:
    """
    Main entry: register views, materialize batch, serve & feed online.
    Auto-generates online expressions from offline schemas.
    """

    views: Mapping[str, FeatureView] = field(factory=dict)
    online_client: FlightClient = field(
        validator=optional(instance_of(FlightClient)), default=None
    )

    @property
    def registry(self):
        raise NotImplementedError

    def register_view(self, view: FeatureView):
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
                if feature not in self.views[view].feature_names
            )
            if bad_views_features:
                raise ValueError

        views_features = tuple(reference.split(":", 1) for reference in references)
        validate_views_features(views_features)
        return views_features

    def _apply_ttl_filter_expr(
        self, expr, view: FeatureView, current_time: datetime = None
    ):
        (timestamp_column, ttl) = (view.timestamp_column, view.ttl)
        if not timestamp_column or not ttl or timestamp_column not in expr.schema():
            return expr

        # timestamp_column is assumed to be a string?
        timestamp_expr = expr[timestamp_column].as_timestamp("%Y-%m-%dT%H:%M:%S.%f%z")
        now = xo.literal(current_time) if current_time else xo.now()
        cutoff_time = now - xo.interval(seconds=ttl.total_seconds())
        feature_valid = timestamp_expr >= cutoff_time
        return expr.filter(feature_valid)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
    ):
        features_by_view = toolz.groupby(
            operator.itemgetter(0),
            self._parse_feature_references(features),
        )
        if EVENT_TIMESTAMP not in entity_df.columns:
            raise ValueError(f"entity_df must contain '{EVENT_TIMESTAMP}' column")

            for view_name, feature_names in features_by_view.items():
                view = self.views[view_name]

        # should we be checking entities consistency among views and entity_df
        con = xo.duckdb.connect()
        for view_name, _feature_names in features_by_view.items():
            view = self.views[view_name]
            feature_names = tuple(feature_name for _, feature_name in _feature_names)
            # what if the view does not have a timestamp column?
            feature_timestamp_col = view.timestamp_column
            assert feature_timestamp_col
            key_columns = view.key_columns
            feature_expr = (
                view.offline_expr.into_backend(con=con)
                .mutate(
                    **{
                        # propagate to EVENT_TIMESTAMP so we can join on it
                        EVENT_TIMESTAMP: xo._[feature_timestamp_col].cast("timestamp"),
                    }
                )
                .select([EVENT_TIMESTAMP] + list(key_columns) + list(feature_names))
            )
            # Point-in-time join: get latest feature <= entity timestamp
            result_expr = xo.memtable(entity_df).into_backend(con=con)
            columns = result_expr.columns
            suffix = "_right"
            result_expr = result_expr.asof_join(
                feature_expr,
                on=EVENT_TIMESTAMP,
                predicates=key_columns,
                rname="{name}" + suffix,
            )
            if view.ttl:
                result_expr = result_expr.filter(
                    xo._[f"{EVENT_TIMESTAMP}{suffix}"]
                    >= (xo._[EVENT_TIMESTAMP] - view.ttl)
                )
            result_expr = result_expr.select(
                columns
                + [column for column in feature_expr.columns if column not in columns]
            )
        return result_expr

    def materialize_online(self, features: List[str], current_time: datetime = None):
        features_by_view = toolz.groupby(
            operator.itemgetter(0),
            self._parse_feature_references(features),
        )

        for view_name, _feature_names in features_by_view.items():
            view = self.views[view_name]
            filtered_expr = self._apply_ttl_filter_expr(
                view.offline_expr, view, current_time
            )
            key_col = view.entities[0].key_column
            latest_expr = (
                filtered_expr.order_by([key_col, view.timestamp_column])
                .mutate(
                    row_number=xo.row_number().over(
                        group_by=key_col,
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

            if self.online_client is None:
                raise ValueError("No online client configured")

            tbl = pa.Table.from_pandas(latest_df)
            self.online_client.upload_table(view_name, tbl, overwrite=True)

            print(
                f"Materialized {len(latest_df)} non-expired feature records for view '{view_name}'"
            )

    def get_online_features(
        self,
        features: List[str],
        entity_df: pd.DataFrame,
        apply_ttl: bool = True,
        current_time: datetime = None,
    ):
        con = xo.duckdb.connect()
        result_expr = xo.memtable(entity_df).into_backend(con=con)

        features_by_view = toolz.groupby(
            operator.itemgetter(0),
            self._parse_feature_references(features),
        )

        for view_name, _feature_names in features_by_view.items():
            view = self.views[view_name]
            feature_names = tuple(feature_name for _, feature_name in _feature_names)
            key_col = view.entities[0].key_column

            online_expr = self._build_online_expr(view_name)

            feature_expr = online_expr.select([key_col] + list(feature_names))

            if apply_ttl:
                feature_expr = self._apply_ttl_filter_expr(
                    feature_expr, view, current_time
                )

            feature_expr = feature_expr.into_backend(con=con)

            columns = result_expr.columns

            result_expr = result_expr.join(feature_expr, key_col, how="left")

            result_expr = result_expr.select(
                columns
                + [column for column in feature_expr.columns if column not in columns]
            )

        return result_expr
