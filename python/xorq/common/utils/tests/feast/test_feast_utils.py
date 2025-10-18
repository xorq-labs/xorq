from abc import abstractmethod
from typing import (
    Dict,
    List,
    Optional,
)

import feast
import pytest
from feast import (
    Field,
)
from feast.data_source import (
    DataSource,
)
from feast.feature_view_projection import FeatureViewProjection

from xorq.common.utils.feast_utils import (
    from_feast,
)
from xorq.common.utils.import_utils import (
    import_from_gist,
)


@abstractmethod
def my_base_feature_view_init(
    self,
    *,
    name: str,
    features: Optional[List[Field]] = None,
    description: str = "",
    tags: Optional[Dict[str, str]] = None,
    owner: str = "",
    source: Optional[DataSource] = None,
):
    """
    Creates a BaseFeatureView object.

    Args:
        name: The unique name of the base feature view.
        features (optional): The list of features defined as part of this base feature view.
        description (optional): A human-readable description.
        tags (optional): A dictionary of key-value pairs to store arbitrary metadata.
        owner (optional): The owner of the base feature view, typically the email of the
            primary maintainer.
        source (optional): The source of data for this group of features. May be a stream source, or a batch source.
            If a stream source, the source should contain a batch_source for backfills & batch materialization.
    Raises:
        ValueError: A field mapping conflicts with an Entity or a Feature.
    """
    assert name is not None
    self.name = name
    self.features = features or []
    self.description = description
    self.tags = tags or {}
    self.owner = owner
    self.created_timestamp = None
    self.last_updated_timestamp = None

    if source:
        self.source = source

    # above is what is in feast.base_feature_view.BaseFeatureView.__init__
    # below is things we have to modify to ensure replicability
    for attr in ("features", "entities", "entity_columns"):
        if value := getattr(self, attr, None):
            setattr(self, attr, sorted(value))
    self.projection = FeatureViewProjection.from_definition(self)


def do_rountdrips(cls, instance, n=1):
    for _ in range(n):
        instance = cls.from_feast(instance).to_feast()
    return instance


def assert_roundtrip(instance):
    other = from_feast(instance).to_feast()
    if other != instance:
        diff = diff_dunder_dict(other, instance)
        raise ValueError(f"roundtrip failed: {diff}")


def diff_dunder_dict(obj0, obj1):
    import operator

    import toolz

    is_not_dunder = toolz.complement(operator.methodcaller("startswith", "__"))
    dct0, dct1 = (
        toolz.keyfilter(is_not_dunder, getattr(obj, "__dict__")) for obj in (obj0, obj1)
    )
    assert type(obj0) is type(obj1) and set(dct0) == set(dct1)
    diff = tuple(
        (k, (v0, v1))
        for ((k, v0), (_, v1)) in zip(*(sorted(dct.items()) for dct in (dct0, dct1)))
        if v0 != v1
    )
    return diff


user, gist = "dlovell", "b719dc08a4cffb232e3f92bea3bd75aa"
fer = import_from_gist(user, gist)


instances = (
    fer.project,
    fer.driver,
    # how to materialize path to this working dir? currently just symlinked it
    fer.driver_stats_source,
    fer.input_request,
    fer.driver_stats_fv,
    fer.transformed_conv_rate,
    fer.transformed_conv_rate_fresh,
    fer.driver_activity_v1,
)


@pytest.mark.parametrize("instance", (instances))
# as a hack, we can change into the fer dir?
def test_roundtrip(instance, store_applied, monkeypatch):
    # FeastFileSource.path.exists() must be true
    monkeypatch.chdir(store_applied.path)
    # ordering of features must match instance
    monkeypatch.setattr(
        feast.feature_view.BaseFeatureView, "__init__", my_base_feature_view_init
    )
    assert_roundtrip(instance)
