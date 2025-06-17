from datetime import timedelta
from pathlib import Path

import toolz
from attrs import (
    field,
    frozen,
)
from attrs.validators import (
    deep_iterable,
    instance_of,
    optional,
)
from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    RequestSource,
)

# this is the only "leakage" from feast classes
# and we need a converter
# from feast.types import FeastType
import xorq as xo
import xorq.expr.datatypes as dt


FeastType = None


EVENT_TIMESTAMP = "event_timestamp"


def dct_converter(maybe_dct):
    return tuple(sorted(dict(maybe_dct).items()))


@toolz.curry
def _to_feast(feast_cls, xorq_obj):
    return feast_cls(
        **{attr.name: getattr(xorq_obj, attr.name) for attr in xorq_obj.__attrs_attrs__}
    )


def _from_feast(xorq_cls, feast_obj):
    return xorq_cls(
        **{
            attr.name: getattr(feast_obj, attr.name)
            for attr in xorq_cls.__attrs_attrs__
        }
    )


@frozen
class FeastProject:
    name = field(validator=instance_of(str))
    description = field(validator=optional(instance_of(str)))

    to_feast = _to_feast(Project)

    from_feast = classmethod(_from_feast)


@frozen
class FeastField:
    name = field(validator=instance_of(str))
    dtype = field(validator=instance_of(FeastType))
    description = field(validator=optional(instance_of(str)))
    tags = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple)),
        converter=dct_converter,
    )

    def __attrs_post_init__(self):
        pass

    to_feast = _to_feast(Field)

    from_feast = classmethod(_from_feast)


@frozen
class FeastEntity:
    name = field(validator=instance_of(str))
    join_keys = field(validator=deep_iterable(instance_of(str), instance_of(tuple)))

    to_feast = _to_feast(Entity)

    from_feast = classmethod(_from_feast)


@frozen
class FeastFileSource:
    name = field(validator=instance_of(str))
    path = field(validator=instance_of(str), converter=Path)
    timestamp_field = field(validator=optional(instance_of(str)), default=None)
    created_timestamp_column = field(validator=optional(instance_of(str)), default=None)

    def __attrs_post_init__(self):
        def validate_timestamp_field(field):
            if field:
                assert field in self.expr.schema()
                typ = self.expr[field].type()
                assert dt.Temporal in typ.mro()

        assert self.name
        assert self.path.exists()
        validate_timestamp_field(self.timestamp_field)
        validate_timestamp_field(self.created_timestamp_column)

    @property
    def expr(self):
        match self.path.suffix:
            case ".parquet":
                return xo.deferred_read_parquet(xo.connect(), self.path)
            case ".csv":
                return xo.deferred_read_csv(xo.connect(), self.path)
            case _:
                raise ValueError(
                    f"don't know how to deal with suffix {self.path.suffix}"
                )

    to_feast = _to_feast(FileSource)

    from_feast = classmethod(_from_feast)


@frozen
class FeastDataSource:
    def __attrs_post_init__(self):
        raise NotImplementedError("we only have FeastFileSource for now")


@frozen
class FeastFeatureView:
    name = field(validator=instance_of(str))
    entities = field(
        validator=deep_iterable(instance_of(FeastEntity), instance_of(tuple))
    )
    schema = field(deep_iterable(instance_of(FeastField), instance_of(tuple)))
    online = field(validator=instance_of(bool), default=False)
    source = field(validator=instance_of(FeastDataSource))
    ttl = field(validator=optional(instance_of(timedelta)), default=None)
    tags = field(
        validator=optional(
            deep_iterable(instance_of(tuple), instance_of(tuple)),
            converter=dct_converter,
        )
    )

    to_feast = _to_feast(FeatureView)

    from_feast = classmethod(_from_feast)


@frozen
class FeastRequestSource:
    name = field()
    schema = field(deep_iterable(instance_of(FeastField), instance_of(tuple)))
    description = field(validator=optional(instance_of(str)))
    tags = field(
        validator=optional(
            deep_iterable(instance_of(tuple), instance_of(tuple)),
            converter=dct_converter,
        )
    )
    owner = field(validator=instance_of(str))

    to_feast = _to_feast(RequestSource)

    from_feast = classmethod(_from_feast)


@frozen
class FeastFeatureService:
    owner = field(validator=instance_of(str))
    features = field()

    to_feast = _to_feast(FeatureService)

    from_feast = classmethod(_from_feast)
