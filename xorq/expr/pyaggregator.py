import pickle
from abc import (
    ABC,
    abstractmethod,
)

import pyarrow as pa

from xorq.common.utils import classproperty
from xorq.internal import Accumulator


def make_struct_type(names, arrow_types):
    return pa.struct(
        (
            pa.field(
                field_name,
                arrow_type,
            )
            for field_name, arrow_type in zip(names, arrow_types)
        )
    )


class PyAggregator(Accumulator, ABC):
    """Variadic aggregator for UDAFs"""

    def __init__(self):
        self._states = []

    def pystate(self):
        return pa.concat_arrays(map(pickle.loads, self._states))

    def state(self):
        value = pa.array(
            [self._states],
            type=self.state_type,
        )
        return value

    @abstractmethod
    def py_evaluate(self):
        pass

    def evaluate(self):
        return pa.scalar(
            self.py_evaluate(),
            type=self.return_type,
        )

    def update(self, *arrays) -> None:
        state = pa.StructArray.from_arrays(
            arrays,
            names=self.names,
        )
        self._states.append(pickle.dumps(state))

    def merge(self, states: pa.Array) -> None:
        for state in states.to_pylist():
            self._states.extend(state)

    @classproperty
    def state_type(cls):
        return pa.list_(pa.large_binary())

    @classproperty
    def names(cls):
        return tuple(field.name for field in cls.struct_type)

    @classproperty
    @abstractmethod
    def struct_type(cls):
        pass

    @classproperty
    def volatility(cls):
        return "stable"
