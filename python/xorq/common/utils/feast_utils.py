import functools
from datetime import (
    datetime,
    timedelta,
)
from operator import methodcaller
from pathlib import Path

import toolz
from attrs import (
    field,
    frozen,
)
from attrs.validators import (
    deep_iterable,
    instance_of,
    is_callable,
    optional,
)
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    OnDemandFeatureView,
    Project,
    PushSource,
    RequestSource,
)
from feast.data_format import FileFormat
from feast.feature_logging import (
    LoggingConfig,
)
from feast.feature_view import (
    DUMMY_ENTITY_NAME,
)
from feast.transformation.base import (
    Transformation,
)
from feast.types import (
    # this is the only "leakage" from feast classes
    FeastType,
    ValueType,
)

import xorq.api as xo
import xorq.expr.datatypes as dt


EVENT_TIMESTAMP = "event_timestamp"


def dct_converter(maybe_dct):
    # toolz.compose(tuple, sorted, operator.methodcaller("items"), dict)
    return tuple(sorted(dict(maybe_dct).items()))


def gen_attr_names(has_attrs_attrs):
    # toolz.compose(partial(map, operator.attrgetter("name")), operator.attrgetter("__attrs_attrs__"))
    # operator.methodcaller("__getstate__")
    yield from (attr.name for attr in has_attrs_attrs.__attrs_attrs__)


def getattrs(attrs, obj, *args):
    yield from ((attr, getattr(obj, attr, *args)) for attr in attrs)


def apply_conversions(conversions, kwargs):
    conversions = dict(conversions)
    converted = {k: conversions.get(k, toolz.identity)(v) for k, v in kwargs.items()}
    return converted


@toolz.curry
def _to_feast(feast_cls, xorq_obj, conversions=(), post_process=None):
    kwargs = apply_conversions(
        conversions,
        dict(getattrs(gen_attr_names(xorq_obj), xorq_obj)),
    )
    if post_process:
        feast_obj = post_process(xorq_obj, feast_cls, kwargs)
    else:
        feast_obj = feast_cls(**kwargs)
    return feast_obj


@toolz.curry
def _from_feast(xorq_cls, feast_obj, conversions=(), post_process=None):
    kwargs = apply_conversions(
        conversions,
        dict(getattrs(gen_attr_names(xorq_cls), feast_obj, None)),
    )
    if post_process:
        xorq_obj = post_process(xorq_cls, feast_obj, kwargs)
    else:
        xorq_obj = xorq_cls(**kwargs)
    return xorq_obj


def from_feast(obj):
    match obj:
        case None:
            return None
        case tuple() | list():
            return tuple(from_feast(el) for el in obj)
        case _:
            typ = next((typ for typ in typs if isinstance(obj, typ.feast_cls)), None)
            if typ:
                return typ.from_feast(obj)
            else:
                raise NotImplementedError(f"don't know how to convert type {type(obj)}")


list_map_to_feast = toolz.compose(list, toolz.partial(map, methodcaller("to_feast")))


@frozen
class FeastProject:
    name = field(validator=instance_of(str))
    description = field(validator=instance_of(str), default="")
    tags = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple)),
        converter=dct_converter,
        default=(),
    )
    owner = field(validator=instance_of(str), default="")
    created_timestamp = field(validator=optional(instance_of(datetime)), default=None)
    last_updated_timestamp = field(
        validator=optional(instance_of(datetime)), default=None
    )
    #
    feast_cls = Project

    to_feast = _to_feast(feast_cls, conversions=(("tags", dict),))

    from_feast = classmethod(_from_feast)


@frozen
class FeastField:
    name = field(validator=instance_of(str))
    dtype = field(validator=instance_of(FeastType))
    description = field(validator=instance_of(str), default="")
    tags = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple)),
        converter=dct_converter,
        default=(),
    )
    vector_index = field(validator=instance_of(bool), default=False)
    vector_length = field(validator=instance_of(int), default=0)
    vector_search_metric = field(validator=optional(instance_of(str)), default=None)
    #
    feast_cls = Field

    def __attrs_post_init__(self):
        if (self.vector_index, self.vector_length, self.vector_search_metric) != (
            False,
            0,
            None,
        ):
            raise ValueError

    to_feast = _to_feast(feast_cls, conversions=(("tags", dict),))

    from_feast = classmethod(_from_feast)


@frozen
class FeastEntity:
    name = field(validator=instance_of(str))
    join_keys = field(validator=deep_iterable(instance_of(str), instance_of(tuple)))
    value_type = field(validator=optional(instance_of(ValueType)), default=None)
    description = field(validator=instance_of(str), default="")
    tags = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple)),
        converter=dct_converter,
        default=(),
    )
    owner = field(validator=instance_of(str), default="")
    created_timestamp = field(validator=optional(instance_of(datetime)), default=None)
    last_updated_timestamp = field(
        validator=optional(instance_of(datetime)), default=None
    )
    #
    feast_cls = Entity

    def __attrs_post_init__(self):
        # this is only every created once
        assert len(self.join_keys) == 1

    def to_feast_post_process(self, feast_cls, kwargs):
        setattrs = ("created_timestamp", "last_updated_timestamp")
        to_setattr = ((k, kwargs[k]) for k in setattrs)
        feast_obj = feast_cls(**toolz.dissoc(kwargs, *setattrs))
        for k, v in to_setattr:
            setattr(feast_obj, k, v)
        return feast_obj

    @staticmethod
    def from_feast_post_process(cls, feast_obj, kwargs):
        xorq_obj = cls(
            **kwargs
            | {
                "join_keys": (feast_obj.join_key,),
            }
        )
        return xorq_obj

    to_feast = _to_feast(
        feast_cls,
        conversions=(
            ("tags", dict),
            ("join_keys", list),
        ),
        post_process=to_feast_post_process,
    )

    from_feast = classmethod(_from_feast(post_process=from_feast_post_process))


@frozen
class FeastFeatureViewProjection:
    # feast_cls = FeatureViewProjection

    def __attrs_post_init__(self):
        raise NotImplementedError


@frozen
class FeastDataSource:
    # feast_cls = DataSource

    def __attrs_post_init__(self):
        raise NotImplementedError("we only have FeastFileSource for now")


@frozen
class FeastFileSource:
    path = field(validator=instance_of(Path), converter=Path)
    name = field(validator=instance_of(str), default="")
    event_timestamp_column = field(validator=optional(instance_of(str)), default=None)
    file_format = field(validator=optional(instance_of(FileFormat)), default=None)
    created_timestamp_column = field(validator=instance_of(str), default="")
    s3_endpoint_override = field(validator=optional(instance_of(str)), default=None)
    description = field(validator=instance_of(str), default="")
    tags = field(
        validator=optional(
            deep_iterable(instance_of(tuple), instance_of(tuple)),
        ),
        converter=dct_converter,
        default=(),
    )
    owner = field(validator=instance_of(str), default="")
    timestamp_field = field(validator=instance_of(str), default="")
    #
    feast_cls = FileSource

    def __attrs_post_init__(self):
        def validate_timestamp_field(field):
            if field:
                assert field in self.expr.schema()
                typ = type(self.expr[field].type())
                assert dt.Temporal in typ.mro()

        assert self.name
        assert self.path.exists()
        validate_timestamp_field(self.timestamp_field)
        validate_timestamp_field(self.created_timestamp_column)

    @property
    def expr(self):
        match self.path.suffix:
            case ".parquet":
                return xo.deferred_read_parquet(self.path, xo.connect())
            case ".csv":
                return xo.deferred_read_csv(self.path, xo.connect())
            case _:
                raise ValueError(
                    f"don't know how to deal with suffix {self.path.suffix}"
                )

    to_feast = _to_feast(
        feast_cls,
        conversions=(
            ("batch_source", methodcaller("to_feast")),
            ("tags", dict),
            ("path", str),
        ),
    )

    from_feast = classmethod(_from_feast)


@frozen
class FeastPushSource:
    name = field(validator=instance_of(str))
    batch_source = field(validator=instance_of((FeastDataSource, FeastFileSource)))
    description = field(validator=instance_of(str), default="")
    tags = field(
        validator=optional(
            deep_iterable(instance_of(tuple), instance_of(tuple)),
        ),
        converter=dct_converter,
        default=(),
    )
    owner = field(validator=instance_of(str), default="")
    #
    feast_cls = PushSource

    to_feast = _to_feast(
        feast_cls,
        conversions=(
            ("batch_source", methodcaller("to_feast")),
            ("tags", dict),
        ),
    )

    from_feast = classmethod(_from_feast(conversions=(("batch_source", from_feast),)))


# every field of schema either goes into entity_columns or into features
# entity_columns if its in join keys
# features if not


@frozen
class FeastFeatureView:
    name = field(validator=instance_of(str))
    source = field(
        validator=instance_of((FeastFileSource, FeastDataSource, FeastPushSource)),
    )
    schema = field(
        validator=deep_iterable(instance_of(FeastField), instance_of(tuple)),
        default=(),
    )
    # we only get strings, but would like to have the join keys
    entities = field(
        # validator=deep_iterable(instance_of(FeastEntity), instance_of(tuple)),
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
        converter=tuple,
        default=(DUMMY_ENTITY_NAME,),
    )
    # FIXME: default should be datetime.timedelta(0)
    ttl = field(validator=optional(instance_of(timedelta)), default=None)
    online = field(validator=instance_of(bool), default=False)
    offline = field(validator=instance_of(bool), default=True)
    description = field(validator=instance_of(str), default="")
    tags = field(
        validator=optional(
            deep_iterable(instance_of(tuple), instance_of(tuple)),
        ),
        converter=dct_converter,
        default=(),
    )
    owner = field(validator=instance_of(str), default="")
    #
    # how we handle getitem on feature view that results in a mutation of self.projection
    # do we have to inspect the FeatureView to see if its already been sliced?
    # it mutates projection.{features,desired_features}
    item = field(
        validator=deep_iterable(instance_of(FeastField), instance_of(tuple)),
        default=(),
    )
    #
    feast_cls = FeatureView

    def __getitem__(self, item):
        return from_feast(self.instance[item])

    @property
    @functools.cache
    def instance(self):
        obj = self.to_feast()
        if self.item:
            # obj.projection.features = list(self.item)
            obj = obj[list(self.item)]
        return obj

    # def __attrs_post_init__(self):
    #     # try to attain feast order for equals testing
    #     object.__setattr__(self, "schema", tuple(set(self.schema)))
    @staticmethod
    def from_feast_post_process(cls, feast_obj, kwargs):
        if feast_obj.projection.desired_features:
            raise ValueError
        return cls(
            **kwargs
            | {
                "item": from_feast(feast_obj.projection.features),
            }
        )

    def to_feast_post_process(self, feast_cls, kwargs):
        feast_obj = feast_cls(**toolz.dissoc(kwargs, "item"))
        if item := kwargs.get("item"):
            # feast_obj = feast_obj[list(item)]
            feast_obj.projection.features = list(el.to_feast() for el in item)
        return feast_obj

    to_feast = _to_feast(
        feast_cls,
        conversions=(
            ("schema", list_map_to_feast),
            ("tags", dict),
            ("entities", lambda x: [Entity(name=el, join_keys=[]) for el in x]),
            ("source", methodcaller("to_feast")),
        ),
        post_process=to_feast_post_process,
    )

    from_feast = classmethod(
        _from_feast(
            conversions=(
                ("source", from_feast),
                ("schema", from_feast),
            ),
            post_process=from_feast_post_process,
        )
    )


@frozen
class FeastRequestSource:
    name = field()
    schema = field(
        validator=deep_iterable(instance_of(FeastField), instance_of(tuple)),
    )
    description = field(validator=instance_of(str), default="")
    tags = field(
        validator=optional(
            deep_iterable(instance_of(tuple), instance_of(tuple)),
        ),
        converter=dct_converter,
        default=(),
    )
    owner = field(validator=instance_of(str), default="")
    #
    feast_cls = RequestSource

    to_feast = _to_feast(
        feast_cls, conversions=(("schema", list_map_to_feast), ("tags", dict))
    )

    from_feast = classmethod(
        _from_feast(
            conversions=(("schema", from_feast),),
        )
    )


@frozen
class FeastOnDemandFeatureView:
    name = field(validator=instance_of(str))
    # we would like to be able to infer / retain the fields
    entities = field(
        # validator=deep_iterable(instance_of(FeastEntity), instance_of(tuple)),
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
        converter=tuple,
        default=(DUMMY_ENTITY_NAME,),
    )
    schema = field(
        validator=deep_iterable(instance_of(FeastField), instance_of(tuple)),
        default=(),
    )
    sources = field(
        validator=deep_iterable(
            instance_of(
                (FeastFeatureView, FeastRequestSource, FeastFeatureViewProjection)
            ),
            instance_of(tuple),
        ),
        converter=tuple,
        default=(),
    )
    udf = field(validator=optional(is_callable()), default=None)
    udf_string = field(validator=instance_of(str), default="")
    feature_transformation = field(
        validator=optional(instance_of(Transformation)), default=None
    )
    mode = field(validator=instance_of(str), default="pandas")
    description = field(validator=instance_of(str), default="")
    tags = field(
        validator=optional(
            deep_iterable(instance_of(tuple), instance_of(tuple)),
        ),
        converter=dct_converter,
        default=(),
    )
    owner = field(validator=instance_of(str), default="")
    write_to_online_store = field(validator=instance_of(bool), default=False)
    singleton = field(validator=instance_of(bool), default=False)
    #
    feast_cls = OnDemandFeatureView

    @classmethod
    def on_demand_feature_view(cls, name=None, **kwargs):
        def decorator(user_function):
            import dill

            user_function.__module__ = "__main__"
            udf_string = dill.source.getsource(user_function)
            obj = cls(
                name=name or user_function.__name__,
                udf=user_function,
                udf_string=udf_string,
                **kwargs,
            )
            functools.update_wrapper(wrapper=obj, wrapped=user_function)
            return obj

        return decorator

    to_feast = _to_feast(
        feast_cls,
        conversions=(
            ("schema", list_map_to_feast),
            ("tags", dict),
            ("entities", lambda x: [Entity(name=el, join_keys=[]) for el in x]),
            # ("sources", methodcaller("to_feast")),
            ("sources", list_map_to_feast),
        ),
    )

    from_feast = classmethod(
        _from_feast(
            conversions=(
                ("sources", from_feast),
                ("schema", from_feast),
            ),
        )
    )


@frozen
class FeastFeatureService:
    name = field(validator=instance_of(str))
    features = field(
        validator=deep_iterable(
            instance_of((FeastFeatureView, FeastOnDemandFeatureView)),
            instance_of(tuple),
        )
    )
    tags = field(
        validator=optional(
            deep_iterable(instance_of(tuple), instance_of(tuple)),
        ),
        converter=dct_converter,
        default=(),
    )
    description = field(validator=instance_of(str), default="")
    owner = field(validator=instance_of(str), default="")
    logging_config = field(validator=optional(instance_of(LoggingConfig)), default=None)
    #
    feast_cls = FeatureService

    @staticmethod
    def from_feast_post_process(cls, feast_obj, kwargs):
        xorq_obj = cls(**kwargs | {"features": from_feast(feast_obj._features)})
        return xorq_obj

    to_feast = _to_feast(
        feast_cls,
        conversions=(
            ("features", list_map_to_feast),
            ("tags", dict),
        ),
    )

    # from_feast = classmethod(_from_feast(conversions=(("features", from_feast),)))
    from_feast = classmethod(_from_feast(post_process=from_feast_post_process))


typs = (
    FeastProject,
    FeastField,
    FeastEntity,
    FeastFeatureView,
    # FeastFeatureViewProjection,
    FeastFileSource,
    # FeastDataSource,
    FeastPushSource,
    FeastRequestSource,
    FeastOnDemandFeatureView,
    FeastFeatureService,
)
