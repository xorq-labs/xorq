import contextlib
import functools
import operator
from datetime import datetime
from pathlib import (
    Path,
)

import dask
import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)


@dask.base.normalize_token.register(dask.utils.methodcaller)
def normalize_methodcaller(mc):
    return dask.base.normalize_token(
        (
            dask.utils.methodcaller,
            mc.method,
        )
    )


@frozen
class Store:
    path = field(validator=instance_of(Path), converter=Path)

    def __attrs_post_init__(self):
        assert self.path.exists()

    @property
    @functools.cache
    def store(self):
        import feast

        return feast.FeatureStore(self.path)

    @property
    def config(self):
        return self.store.config

    @property
    def provider(self):
        return self.store._get_provider()

    @property
    @functools.cache
    def repo_contents(self):
        import feast.repo_operations

        with contextlib.chdir(self.path):
            return feast.repo_operations._get_repo_contents(
                self.path, self.project_name
            )

    @property
    def registry(self):
        return self.store._registry

    @property
    def entities(self):
        return self.store.list_entities()

    @property
    def project_name(self):
        return self.config.project

    def apply(self, skip_source_validation=False):
        import feast.repo_operations

        with contextlib.chdir(self.path):
            return feast.repo_operations.apply_total(
                self.config, self.path, skip_source_validation=skip_source_validation
            )

    def teardown(self):
        return self.store.teardown()

    def list_on_demand_feature_view_names(self):
        return tuple(el.name for el in self.repo_contents.on_demand_feature_views)

    def get_on_demand_feature_view(self, on_demand_feature_view_name):
        return self.registry.get_on_demand_feature_view(
            on_demand_feature_view_name, self.store.project
        )

    def list_feature_view_names(self):
        return tuple(el.name for el in self.repo_contents.feature_views)

    def get_feature_view(self, feature_view_name):
        return self.registry.get_feature_view(feature_view_name, self.store.project)

    def get_feature_refs(self, features):
        import feast.utils as utils

        return utils._get_features(self.registry, self.store.project, list(features))

    def get_feature_views_to_use(self, features):
        import feast.utils as utils

        (all_feature_views, all_on_demand_feature_views) = (
            utils._get_feature_views_to_use(
                self.registry,
                self.store.project,
                list(features),
            )
        )
        return (all_feature_views, all_on_demand_feature_views)

    def get_grouped_feature_views(self, features):
        import feast.utils as utils

        feature_refs = self.get_feature_refs(features)
        (all_feature_views, all_on_demand_feature_views) = (
            self.get_feature_views_to_use(features)
        )
        fvs, odfvs = utils._group_feature_refs(
            feature_refs,
            all_feature_views,
            all_on_demand_feature_views,
        )
        (feature_views, on_demand_feature_views) = (
            tuple(view for view, _ in gen) for gen in (fvs, odfvs)
        )
        return feature_views, on_demand_feature_views

    def validate_entity_expr(self, entity_expr, features, full_feature_names=False):
        import feast.utils as utils

        (_, on_demand_feature_views) = self.get_grouped_feature_views(features)
        if self.store.config.coerce_tz_aware:
            # FIXME: pass entity_expr back out
            # entity_df = utils.make_df_tzaware(typing.cast(pd.DataFrame, entity_df))
            pass
        bad_pairs = (
            (feature_name, odfv.name)
            for odfv in on_demand_feature_views
            for feature_name in odfv.get_request_data_schema().keys()
            if feature_name not in entity_expr.columns
        )
        if pair := next(bad_pairs, None):
            from feast.feature_store import RequestDatanotFoundInEntityDfException

            (feature_name, feature_view_name) = pair
            raise RequestDatanotFoundInEntityDfException(
                feature_name=feature_name,
                feature_view_name=feature_view_name,
            )
        utils._validate_feature_refs(
            self.get_feature_refs(features),
            full_feature_names,
        )

    def get_historical_features(self, entity_expr, features, full_feature_names=False):
        self.validate_entity_expr(
            entity_expr, features, full_feature_names=full_feature_names
        )
        (odfv_dct, fv_dct) = group_features(self, features)
        entity_expr, all_join_keys = process_all_feature_views(
            self, entity_expr, fv_dct
        )
        expr = process_odfvs(entity_expr, odfv_dct)
        return expr

    def get_historical_features_feast(
        self, entity_df, features, full_feature_names=False
    ):
        return self.store.get_historical_features(
            entity_df=entity_df,
            features=features,
            full_feature_names=full_feature_names,
        )

    def get_online_features(self, features, entity_rows):
        return self.store.get_online_features(
            features=features,
            entity_rows=entity_rows,
        ).to_dict()

    def list_feature_service_names(self):
        return tuple(el.name for el in self.store.list_feature_services())

    def get_feature_service(self, feature_service_name):
        return self.store.get_feature_service(feature_service_name)

    def list_data_source_names(self):
        return tuple(
            el.name for el in self.registry.list_data_sources(self.project_name)
        )

    def get_data_source(self, data_source_name):
        return self.registry.get_data_source(data_source_name, self.project_name)

    @classmethod
    def make_applied_materialized(cls, path, end_date=None):
        end_date = end_date or datetime.now()
        store = cls(path)
        store.apply()
        store.store.materialize_incremental(end_date=end_date)
        return store


def process_one_feature_view(
    entity_expr, store, feature_view, feature_names, all_join_keys
):
    from feast.infra.offline_stores.offline_utils import (
        DEFAULT_ENTITY_DF_EVENT_TIMESTAMP_COL,
    )

    import xorq.api as xo

    def _read_mapped(
        con,
        store,
        feature_view,
        feature_names,
        right_entity_key_columns,
        ets,
        ts,
        full_feature_names=False,
    ):
        def maybe_rename(expr, dct):
            return (
                expr.rename({to_: from_ for from_, to_ in dct.items() if from_ in expr})
                if dct
                else expr
            )

        if full_feature_names:
            raise ValueError
        expr = (
            xo.deferred_read_parquet(
                store.config.repo_path.joinpath(feature_view.batch_source.path), con=con
            )
            .pipe(maybe_rename, feature_view.batch_source.field_mapping)
            .pipe(maybe_rename, feature_view.projection.join_key_map)
            .select(list(right_entity_key_columns) + list(feature_names))
        )
        if ts == ets:
            new_ts = f"__{ts}"
            expr, ts = expr.pipe(maybe_rename, {ts: new_ts}), new_ts
        return expr, ts

    def _merge(entity_expr, feature_expr, join_keys):
        return entity_expr.join(
            feature_expr, predicates=join_keys, how="left", rname="{name}__"
        )

    def _normalize_timestamp(expr, *tss):
        casts = {
            ts: xo.expr.datatypes.Timestamp(timezone="UTC")
            for ts in tss
            if ts in expr and expr[ts].type().timezone is None
        }
        return expr.cast(casts) if casts else expr

    def _filter_ttl(expr, ttl, ets, ts):
        isna_condition = expr[ts].isnull()
        le_condition = expr[ts] <= expr[ets]
        if ttl and ttl.total_seconds() != 0:
            ge_condition = (
                expr[ets] - xo.interval(seconds=ttl.total_seconds())
            ) <= expr[ts]
            time_condition = ge_condition & le_condition
        else:
            time_condition = le_condition
        condition = isna_condition | time_condition
        return expr[condition]

    def _drop_duplicates(expr, join_keys, ets, ts, cts):
        order_by = tuple(
            expr[ts].desc(nulls_first=False)
            # cts desc first: most recent update
            # ts desc: closest to the event ts
            for ts in (cts, ts)
            if ts in expr
        )
        ROW_NUM = "row_num"
        expr = (
            expr.mutate(
                **{
                    ROW_NUM: (
                        xo.row_number().over(
                            group_by=list(join_keys) + [ets],
                            order_by=order_by,
                        )
                    ),
                }
            )
            .filter(xo._[ROW_NUM] == 0)
            .drop(ROW_NUM)
        )
        return expr

    ets = DEFAULT_ENTITY_DF_EVENT_TIMESTAMP_COL
    assert ets in entity_expr
    con = entity_expr._find_backend()

    ts, cts = (
        feature_view.batch_source.timestamp_field,
        feature_view.batch_source.created_timestamp_column,
    )
    join_keys = tuple(
        feature_view.projection.join_key_map.get(entity_column.name, entity_column.name)
        for entity_column in feature_view.entity_columns
    )
    all_join_keys = all_join_keys + [
        join_key for join_key in join_keys if join_key not in all_join_keys
    ]
    right_entity_key_columns = list(filter(None, [ts, cts] + list(join_keys)))

    entity_expr = _normalize_timestamp(entity_expr, ets)

    feature_expr, ts = _read_mapped(
        con, store, feature_view, feature_names, right_entity_key_columns, ets, ts
    )
    expr = _merge(entity_expr, feature_expr, join_keys)
    expr = _normalize_timestamp(expr, ts, cts)
    expr = _filter_ttl(expr, feature_view.ttl, ets, ts)
    expr = _drop_duplicates(expr, all_join_keys, ets, ts, cts)
    return expr, all_join_keys


def process_all_feature_views(store, entity_expr, fv_dct):
    all_join_keys = []
    for feature_view, feature_names in fv_dct.items():
        entity_expr, all_join_keys = process_one_feature_view(
            entity_expr, store, feature_view, feature_names, all_join_keys
        )
    return entity_expr, all_join_keys


@toolz.curry
def apply_odfv_dct(df, odfv_udfs):
    for other in (udf(df) for udf in odfv_udfs):
        df = df.join(other)
    return df


def make_uniform_timestamps(expr, timezone="UTC", scale=6):
    import xorq.vendor.ibis.expr.datatypes as dt

    casts = {
        name: dt.Timestamp(timezone=timezone, scale=scale)
        for name, typ in expr.schema().items()
        if isinstance(typ, dt.Timestamp)
    }
    return expr.cast(casts) if casts else expr


def calc_odfv_schema_append(odfv_dct):
    fields = (field for odfv in odfv_dct for field in odfv.features)
    schema_append = {field.name: field.dtype.name for field in fields}
    return schema_append


def process_odfvs(entity_expr, odfv_dct, full_feature_names=False):
    import xorq.expr.relations as rel

    if full_feature_names:
        raise ValueError
    entity_expr = make_uniform_timestamps(entity_expr)
    odfv_udfs = tuple(odfv.feature_transformation.udf for odfv in odfv_dct.keys())
    schema_in = entity_expr.schema()
    schema_append = calc_odfv_schema_append(odfv_dct)
    udxf = rel.flight_udxf(
        process_df=apply_odfv_dct(odfv_udfs=odfv_udfs),
        maybe_schema_in=schema_in,
        maybe_schema_out=schema_in | schema_append,
        name="process_odfvs",
    )
    return udxf(entity_expr)


def group_features(store, feature_names):
    import feast

    splat = tuple(feature_name.split(":") for feature_name in feature_names)
    if bad_feature_splats := tuple(el for el in splat if len(el) != 2):
        raise ValueError(
            f"got invalid feature names: {tuple(':'.join(el) for el in bad_feature_splats)}"
        )
    name_to_use_to_view = {
        view.projection.name_to_use(): view
        for view in store.store.list_all_feature_views()
    }
    dct = toolz.groupby(
        operator.itemgetter(0),
        splat,
    )
    view_to_feature_names = {
        name_to_use_to_view[feature_view_name]: tuple(
            feature_name for _, feature_name in pairs
        )
        for feature_view_name, pairs in dct.items()
    }
    is_odfv = toolz.flip(isinstance)(feast.OnDemandFeatureView)
    (odfv_dct, fv_dct) = (
        toolz.keyfilter(f, view_to_feature_names)
        for f in (is_odfv, toolz.complement(is_odfv))
    )
    return odfv_dct, fv_dct
