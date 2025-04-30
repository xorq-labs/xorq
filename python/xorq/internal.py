from abc import ABCMeta, abstractmethod
from typing import List

import pyarrow as pa
from xorq_datafusion._internal import (
    AggregateUDF,
    ContextProvider,
    DataFrame,
    LogicalPlan,
    LogicalPlanBuilder,
    OptimizerContext,
    OptimizerRule,
    ScalarUDF,
    SessionConfig,  # noqa: F401
    SessionContext,  # noqa: F401
    SessionState,  # noqa: F401
    Table,
    TableProvider,
    WindowUDF,
)


__all__ = [
    "SessionContext",
    "SessionState",
    "SessionConfig",
    "ContextProvider",
    "OptimizerContext",
    "LogicalPlan",
    "LogicalPlanBuilder",
    "ScalarUDF",
    "AggregateUDF",
    "OptimizerRule",
    "OptimizationRule",
    "TableProvider",
    "AbstractTableProvider",
    "Table",
    "Accumulator",
    "DataFrame",
    "WindowUDF",
    "WindowEvaluator",
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


class OptimizationRule(metaclass=ABCMeta):
    @abstractmethod
    def try_optimize(self, plan: LogicalPlan) -> LogicalPlan:
        pass


class AbstractTableProvider(metaclass=ABCMeta):
    @abstractmethod
    def schema(self):
        pass

    @abstractmethod
    def scan(self, filters=None):
        pass


class WindowEvaluator(metaclass=ABCMeta):
    def memoize(self) -> None:
        pass

    def get_range(self, idx: int, num_rows: int) -> tuple[int, int]:
        return idx, idx + 1

    def is_causal(self) -> bool:
        return False

    def evaluate_all(self, values: list[pa.Array], num_rows: int) -> pa.Array:
        pass

    def evaluate(
        self, values: list[pa.Array], eval_range: tuple[int, int]
    ) -> pa.Scalar:
        pass

    def evaluate_all_with_rank(
        self, num_rows: int, ranks_in_partition: list[tuple[int, int]]
    ) -> pa.Array:
        pass

    def supports_bounded_execution(self) -> bool:
        return False

    def uses_window_frame(self) -> bool:
        return False

    def include_rank(self) -> bool:
        return False


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


def udwf(
    func: WindowEvaluator,
    input_types: pa.DataType | list[pa.DataType],
    return_type: pa.DataType,
    volatility: str,
    name: str | None = None,
) -> WindowUDF:
    """Create a new User-Defined Window Function.

    Args:
        func: The python function.
        input_types: The data types of the arguments to ``func``.
        return_type: The data type of the return value.
        volatility: See :py:class:`Volatility` for allowed values.
        name: A descriptive name for the function.

    Returns:
        A user-defined window function.
    """
    if not isinstance(func, WindowEvaluator):
        raise TypeError("`func` must implement the abstract base class WindowEvaluator")
    if name is None:
        name = func.__class__.__qualname__.lower()
    if isinstance(input_types, pa.DataType):
        input_types = [input_types]

    return WindowUDF(name, func, input_types, return_type, str(volatility))
