import pandas as pd
import pyarrow as pa
import xorq as xo
import threading
from xorq.flight.client import FlightClient
from typing import List, Mapping, Any, Callable


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


class DataSource:
    """
    Abstraction for batch or online table.
    con: supports table(name) for Ibis connections or FlightClient for lookups.
    """
    def __init__(self, name: str, con: Any, table_name: str, schema: pa.Schema):
        self.name = name
        self.con = con
        self.table_name = table_name
        self.schema = schema

    @property
    def table(self):
        if hasattr(self.con, 'table'):
            return self.con.table(self.table_name)
        raise ValueError("Unsupported connector for table retrieval")


class FeatureView:
    """
    Bundles entity, sources, and features for batch and online.
    """
    def __init__(
        self,
        name: str,
        offline_source: DataSource,
        online_source: DataSource,
        entity: Entity,
        features: List[Feature]
    ):
        self.name = name
        self.offline_source = offline_source
        self.online_source = online_source
        self.entity = entity
        self.features = features

    def offline_expr(self):
        base = self.offline_source.table
        mapping = {f.name: f.expr for f in self.features}
        cols = [self.entity.key_column, self.entity.timestamp_column] + list(mapping.keys())
        return base.mutate(**mapping).select(cols)

    def online_expr(self):
        # base = Flight-backed table containing only the precomputed features
        base = self.online_source.table
        mapping = {
            f.name: f.expr.unbind()  # “unbind” → “bind to this table”
            for f in self.features
        }

        cols = [self.entity.key_column,
                self.entity.timestamp_column] + list(mapping.keys())

        return base.select(cols)


class FeatureStore:
    """
    Main entry: register sources/views, materialize batch, serve & feed online.
    """
    def __init__(self):
        self.registry = FeatureRegistry()
        self.sources: Mapping[str, DataSource] = {}
        self.views: Mapping[str, FeatureView] = {}

    def register_source(self, src: DataSource):
        self.sources[src.name] = src

    def register_view(self, view: FeatureView):
        if view.entity.name not in self.registry.entities:
            raise ValueError("Entity not registered before view")
        for f in view.features:
            self.registry.register_feature(f)
        self.views[view.name] = view

    def materialize_online(self, view_name: str):
        view = self.views[view_name]

        batch_df = (
                view.offline_expr().execute()
        )

        latest = (
            batch_df
              .sort_values(view.entity.timestamp_column)
              .groupby(view.entity.key_column)
              .tail(1)
        )

        client: FlightClient = self.sources["online"].con.con
        tbl = pa.Table.from_pandas(latest)
        client.upload_data(
            view.online_source.table_name,
            tbl,
            overwrite=True
        )

    def get_online_features(
        self,
        view_name: str,
        rows: List[dict],
    ) -> pd.DataFrame:
        view = self.views[view_name]
        key_col = view.entity.key_column

        keys_df = pd.DataFrame(rows)

        expr = (
            view
              .online_expr()
              .filter(xo._[key_col].isin(keys_df[key_col].tolist()))
        )

        # run it straight through the FlightClient / duckdb Flight
        client = self.sources["online"].con.con
        return client.execute(expr)

#    def get_online_features(self, view_name: str, rows: List[dict]) -> pd.DataFrame:
#        view = self.views[view_name]
#        key_col = view.entity.key_column
#
#        df_keys = pd.DataFrame(rows)
#
#        table_name = view.offline_source.table_name
#        offline_df = xo.memtable(view.offline_source.con.tables[table_name].execute())
#
#        client = self.sources['online'].con
#        table_expr = client.tables[view.online_source.table_name]
#        online_df = xo.memtable(table_expr.execute())
#
#        mappings = {f.name: f.expr.unbind() for f in view.features}
#        cols = [view.entity.key_column, view.entity.timestamp_column] + list(mappings.keys())
#        expr = table_expr.unbind()
#        expr = expr.mutate(**mappings).filter(xo._[key_col].isin(df_keys[key_col].tolist())).select(cols)
#
#        # temporary fix since we cannot do into_backend with a flight backend
#        con = xo.duckdb.connect()
#        con.create_table("weather_history", online_df.union(offline_df))
#
#        return con.execute(expr)
