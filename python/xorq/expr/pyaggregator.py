from __future__ import annotations

import pickle
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import Iterable
from typing import Any

import pyarrow as pa

from xorq.common.utils import classproperty
from xorq.internal import Accumulator


def make_struct_type(
    names: Iterable[str], arrow_types: Iterable[pa.DataType]
) -> pa.StructType:
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

    def __init__(self) -> None:
        self._states: list[bytes] = []

    def pystate(self) -> pa.Array:
        return pa.concat_arrays(map(pickle.loads, self._states))

    def state(self) -> pa.Array:
        value = pa.array(
            [self._states],
            type=self.state_type,
        )
        return value

    @abstractmethod
    def py_evaluate(self) -> Any:
        pass

    def evaluate(self) -> pa.Scalar:
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
    def state_type(cls) -> pa.DataType:
        return pa.list_(pa.large_binary())

    @classproperty
    def names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in cls.struct_type)

    @classproperty
    @abstractmethod
    def struct_type(cls) -> pa.StructType:
        pass

    @classproperty
    def volatility(cls) -> str:
        return "stable"
