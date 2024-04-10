from abc import ABCMeta, abstractmethod
from typing import List

import pyarrow as pa

from ._internal import (
    ScalarUDF,
    AggregateUDF,
    SessionContext,  # noqa: F401
    SessionConfig,  # noqa: F401
    ContextProvider,
    OptimizerContext,
    LogicalPlan,
    LogicalPlanBuilder,
)

__all__ = [
    "SessionContext",
    "ContextProvider",
    "OptimizerContext",
    "LogicalPlan",
    "LogicalPlanBuilder",
    "SessionConfig",
    "ScalarUDF",
    "AggregateUDF",
]


class Accumulator(metaclass=ABCMeta):
    @abstractmethod
    def state(self) -> List[pa.Scalar]:
        pass

    @abstractmethod
    def update(self, values: pa.Array) -> None:
        pass

    @abstractmethod
    def merge(self, states: pa.Array) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> pa.Scalar:
        pass


def udf(func, input_types, return_type, volatility, name=None):
    """
    Create a new User Defined Function
    """
    if not callable(func):
        raise TypeError("`func` argument must be callable")
    if name is None:
        name = func.__qualname__.lower()
    return ScalarUDF(
        name=name,
        func=func,
        input_types=input_types,
        return_type=return_type,
        volatility=volatility,
    )


def udaf(accum, input_type, return_type, state_type, volatility, name=None):
    """
    Create a new User Defined Aggregate Function
    """
    if not issubclass(accum, Accumulator):
        raise TypeError("`accum` must implement the abstract base class Accumulator")
    if name is None:
        name = accum.__qualname__.lower()
    return AggregateUDF(
        name=name,
        accumulator=accum,
        input_type=input_type,
        return_type=return_type,
        state_type=state_type,
        volatility=volatility,
    )
